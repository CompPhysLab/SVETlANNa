from typing import Literal
import torch

from .element import Element
from ..simulation_parameters import SimulationParameters
from ..parameters import OptimizableFloat
from ..wavefront import Wavefront
from ..axes_math import tensor_dot


class FreeSpace(Element):
    """A class that describes a propagation of the field in free space
    before two optical elements
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        distance: OptimizableFloat,
        method: Literal['auto', 'fresnel', 'AS']
    ):
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Class exemplar, that describes optical system
        distance : float
            distance between two optical elements
        method : Literal['auto', 'fresnel', 'AS']
            Method describing propagation in free space
                (1) 'AS' - angular spectrum method,
                (2) 'fresnel' - fresnel approximation,
                (3) 'auto' - auto mode
        """
        super().__init__(simulation_parameters)

        self.distance = distance
        self.method = method

        # params extracted from SimulationParameters
        device = self.simulation_parameters.device

        self._w_index = self.simulation_parameters.axes.index('W')
        self._h_index = self.simulation_parameters.axes.index('H')

        # TODO: Why we need `_` before a variable name? Anyway it is not an attribute of the class
        # TODO: Is there some convention for `_` and `__` in a variable name?
        _x_linear = self.simulation_parameters.axes.W
        _y_linear = self.simulation_parameters.axes.H

        x_nodes = _x_linear.shape[0]
        y_nodes = _y_linear.shape[0]

        # spatial frequencies calculation
        dx = (_x_linear[1] - _x_linear[0]) if x_nodes > 1 else 1.
        dy = (_y_linear[1] - _y_linear[0]) if y_nodes > 1 else 1.

        kx_linear = 2 * torch.pi * torch.fft.fftfreq(
            x_nodes,
            dx,
            device=device
        )
        ky_linear = 2 * torch.pi * torch.fft.fftfreq(
            y_nodes,
            dy,
            device=device
        )

        # spatial frequencies mesh
        kx_grid = kx_linear[None, :]
        ky_grid = ky_linear[:, None]

        # Calculate (kx^2+ky^2) / k^2 relation
        # 1) float or torch.Tensor of 'wavelength' axis size
        k = 2 * torch.pi / self.simulation_parameters.axes.wavelength
        # 2) calculate (kx^2+ky^2) tensor of size ('H', 'W')
        kx2ky2 = kx_grid ** 2 + ky_grid ** 2
        # 3) relation
        relation, relation_axes = tensor_dot(
            1 / (k ** 2),
            kx2ky2,
            'wavelength',
            ('H', 'W')
        )

        # TODO: What is that? Is this code for some future features?
        # TODO: Must not a `use_filter` be an attribute of the Class?
        # low pass filter, (kx^2 + ky^2) <= k^2
        self._low_pass_filter: torch.Tensor | int  # <- Registering Buffer for _low_pass_filter
        use_filter = False
        if use_filter:
            # TODO: Shouldn't the 88'th string be here?
            self.register_buffer(
                '_low_pass_filter',
                (relation <= 1).to(kx_grid),  # TODO: I did not understand this condition...
                persistent=False
            )
        else:
            self._low_pass_filter = 1

        self._wave_number: torch.Tensor
        self.register_buffer(
            '_wave_number',
            k[..., None, None],
            persistent=False
        )  # TODO: Does a Buffer works even if `k` is not a Class attribute (not self.k)?

        self._calc_axes = relation_axes  # axes tuple used during calculations
        _wave_number_x2y2 = self._low_pass_filter * kx2ky2

        # kz
        if use_filter:  # TODO: what the difference between `if` and `else`?
            # kz
            #    = sqrt(k^2 - (kx^2 + ky^2)), if (kx^2 + ky^2) <= k^2;
            #    = |k| otherwise.
            wave_number_z = torch.sqrt(
                self._wave_number ** 2 - _wave_number_x2y2  # TODO: `+ 0j`?
            )
        else:  #
            wave_number_z = torch.sqrt(
                self._wave_number ** 2 - _wave_number_x2y2 + 0j
            )

        self._wave_number_z: torch.Tensor  # <- Registering Buffer for _wave_number_z
        self.register_buffer(
            '_wave_number_z',
            wave_number_z,
            persistent=False
        )

        # kz taylored, used by Fresnel approximation
        # TODO: Do we need this if `method` is not equal to 'fresnel'?
        self._wave_number_z_eff_fresnel: torch.Tensor  # <- Registering Buffer for _wave_number_z_eff_fresnel
        self.register_buffer(
            '_wave_number_z_eff_fresnel',
            - 0.5 * _wave_number_x2y2 / self._wave_number,
            persistent=False
        )
        # TODO: Maybe add a separate method to define all necessary Buffers?

    def impulse_response_angular_spectrum(self) -> torch.Tensor:
        """Create the impulse response function for angular spectrum method

        Returns
        -------
        torch.Tensor
            2d impulse response function for angular spectrum method
        """

        # Fourier image of impulse response function,
        # 0 if k^2 < (kx^2 + ky^2) [if use_filter]
        # TODO: there is still no information in docstrings about a filter:(
        return self._low_pass_filter * torch.exp(
            (1j * self.distance) * self._wave_number_z
        )  # Comment: Here we use the following exponent: `exp(+ i * d * k)`

    def impulse_response_fresnel(self) -> torch.Tensor:
        """Create the impulse response function for fresnel approximation

        Returns
        -------
        torch.Tensor
            2d impulse response function for fresnel approximation
        """

        # Fourier image of impulse response function
        return self._low_pass_filter * torch.exp(
            (1j * self.distance) * self._wave_number_z_eff_fresnel
        ) * torch.exp(
            (1j * self.distance) * self._wave_number
        )

    def _impulse_response(self, tol: float = 1e-3) -> torch.Tensor:
        # TODO: No docstring!

        if self.method == 'AS':
            return self.impulse_response_angular_spectrum()

        elif self.method == 'fresnel':
            return self.impulse_response_fresnel()

        # TODO: fix auto mod
        # TODO: Didn't we plan to exclude `auto` method at all?
        elif self.method == 'auto':

            # TODO: Somewhere we use `** 2`, somewhere `torch.pow(_, 2)`
            radius_squared = torch.pow(self._x_grid, 2) + torch.pow(self._y_grid, 2)

            # criterion for Fresnel approximation
            fresnel_criterion = (
                torch.pi *
                torch.max(
                    torch.pow(radius_squared, 2)
                ) /
                (4 * self._wavelength * (self.distance ** 3))
            )

            if fresnel_criterion <= tol:
                return self.impulse_response_fresnel()
            else:
                return self.impulse_response_angular_spectrum()

        raise ValueError("Unknown forward propagation method")

    # TODO: ask for tol parameter
    def forward(
        self,
        input_field: Wavefront
    ) -> Wavefront:
        """Method that calculates the field after propagating in the free space

        Parameters
        ----------
        input_field : Wavefront
            Field before propagation in free space
        tol : float, optional
            tolerance for Fresnel approximation, by default 1e-3

        Returns
        -------
        Wavefront
            Field after propagation in free space

        Raises
        ------
        ValueError
            Occurs when a non-existent direct distribution method is chosen
        """

        input_field_fft = torch.fft.fft2(
            input_field,
            dim=(self._h_index, self._w_index)
        )

        impulse_response_fft = self._impulse_response()

        # Fourier image of output field
        output_field_fft, _ = tensor_dot(
            input_field_fft,  # TODO: It's really hard to track dimensions... Maybe add some comments for some tensors?
            impulse_response_fft,
            self.simulation_parameters.axes.names,  # TODO: Will it be better to use named arguments here?
            self._calc_axes,
            preserve_a_axis=True
        )

        output_field = torch.fft.ifft2(
            output_field_fft,
            dim=(self._h_index, self._w_index)
        )

        return output_field

    def reverse(self, transmission_field: torch.Tensor) -> Wavefront:
        # TODO: Check the description...
        """Method that calculates the field after propagating in the free space
        in back propagation

        Parameters
        ----------
        transmission_field : torch.Tensor
            Field before propagation in free space in back propagation  # TODO: Check!

        Returns
        -------
        torch.Tensor
            Field after propagation in free space in back propagation
        """

        transmission_field_fft = torch.fft.fft2(
            transmission_field,
            dim=(self._h_index, self._w_index)
        )

        impulse_response_fft = self._impulse_response().conj()

        # Fourier image of output field
        incident_field_fft, _ = tensor_dot(
            transmission_field_fft,
            impulse_response_fft,
            self.simulation_parameters.axes.names,  # TODO: Named arguments
            self._calc_axes,
            preserve_a_axis=True
        )

        incident_field = torch.fft.ifft2(
            incident_field_fft,
            dim=(self._h_index, self._w_index)
        )

        return incident_field
