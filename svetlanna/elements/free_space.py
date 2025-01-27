from typing import Literal
import torch

from .element import Element
from ..simulation_parameters import SimulationParameters
from ..parameters import OptimizableFloat
from ..wavefront import Wavefront
from ..axes_math import tensor_dot
from warnings import warn


class FreeSpace(Element):
    """A class that describes a propagation of the field in free space
    between two optical elements
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        distance: OptimizableFloat,
        method: Literal['auto', 'fresnel', 'AS']
    ):
        """Free space element.

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            An instance describing the optical system's simulation parameters.
        distance : float
            The distance of the free space propagation.
        method : Literal['auto', 'fresnel', 'AS']
            Method describing propagation in free space
                (1) 'AS' - angular spectrum method,
                (2) 'fresnel' - fresnel approximation,
                (3) 'auto' - auto mode
        """
        super().__init__(simulation_parameters)

        self.distance = self.process_parameter('distance', distance)
        self.method = self.process_parameter('method', method)

        # params extracted from SimulationParameters
        device = self.simulation_parameters.device

        self._w_index = self.simulation_parameters.axes.index('W')
        self._h_index = self.simulation_parameters.axes.index('H')

        x_linear = self.simulation_parameters.axes.W
        y_linear = self.simulation_parameters.axes.H

        x_nodes = x_linear.shape[0]
        y_nodes = y_linear.shape[0]

        # Compute spatial grid spacing
        dx = (x_linear[1] - x_linear[0]) if x_nodes > 1 else 1.
        dy = (y_linear[1] - y_linear[0]) if y_nodes > 1 else 1.

        # Compute wave vectors
        kx_linear = 2 * torch.pi * torch.fft.fftfreq(x_nodes, dx, device=device)
        ky_linear = 2 * torch.pi * torch.fft.fftfreq(y_nodes, dy, device=device)

        # Compute wave vectors grids
        kx_grid = kx_linear[None, :]  # shape: (1, 'W')
        ky_grid = ky_linear[:, None]  # shape: ('H', 1)

        # Calculate (kx^2+ky^2) / k^2 relation
        # 1) Calculate wave vector of shape ('wavelength') or ()
        k = 2 * torch.pi / self.simulation_parameters.axes.wavelength

        # 2) Calculate (kx^2+ky^2) tensor
        kx2ky2 = kx_grid ** 2 + ky_grid ** 2  # shape: ('H', 'W')

        # 3) Calculate (kx^2+ky^2) / k^2
        relation, relation_axes = tensor_dot(
            a=1 / (k ** 2),
            b=kx2ky2,
            a_axis='wavelength',
            b_axis=('H', 'W')
        )  # shape: ('wavelength', 'H', 'W') or ('H', 'W') depending on k shape

        # TODO: Remove legacy filter
        use_legacy_filter = False

        # Legacy low pass filter, (kx^2+ky^2) / k^2 <= 1
        # The filter removes contribution of evanescent waves
        if use_legacy_filter:
            # TODO: Shouldn't the 88'th string be here?
            condition = (relation <= 1)  # calculate the low pass filter condition
            condition = condition.to(kx_grid)  # cast bool to float

            # Registering Buffer for _low_pass_filter
            self._low_pass_filter = self.make_buffer(
                '_low_pass_filter', condition
            )
        else:
            self._low_pass_filter = 1

        # Reshape wave vector for further calculations
        wave_number = k[..., None, None]  # shape: ('wavelength', 1, 1) or (1, 1)

        # Registering Buffer for _wave_number
        self._wave_number = self.make_buffer(
            '_wave_number', wave_number
        )

        self._calc_axes = relation_axes  # axes tuple used during calculations

        # Calculate kz
        if use_legacy_filter:
            # kz = sqrt(k^2 - (kx^2 + ky^2)), if (kx^2 + ky^2) / k^2 <= 1
            #    or
            # kz = |k| otherwise
            wave_number_z = torch.sqrt(
                self._wave_number ** 2 - self._low_pass_filter * kx2ky2
            )
        else:
            # kz = sqrt(k^2 - (kx^2 + ky^2))
            wave_number_z = torch.sqrt(
                self._wave_number ** 2 - kx2ky2 + 0j
            )  # 0j is required to convert argument to complex

        # Registering Buffer for _wave_number_z
        self._wave_number_z = self.make_buffer(
            '_wave_number_z', wave_number_z
        )

        # Calculate kz taylored, used by Fresnel approximation
        wave_number_z_eff_fresnel = - 0.5 * kx2ky2 / self._wave_number

        # Registering Buffer for _wave_number_z_eff_fresnel
        self._wave_number_z_eff_fresnel = self.make_buffer(
            '_wave_number_z_eff_fresnel', wave_number_z_eff_fresnel
        )

        # Warnings for fulfilling the method criteria
        # See (9.32), (9.36) in
        # Fourier Optics and Computational Imaging (2nd ed)
        # by Kedar Khare, Mansi Butola and Sunaina Rajor
        Lx = torch.abs(x_linear[-1] - x_linear[0])
        Ly = torch.abs(y_linear[-1] - y_linear[0])
        if method == 'AS':
            kx_max = torch.max(torch.abs(kx_linear))
            ky_max = torch.max(torch.abs(ky_linear))
            x_condition = kx_max >= k / torch.sqrt(1 + (2*distance / Lx)**2)
            y_condition = ky_max >= k / torch.sqrt(1 + (2*distance / Ly)**2)

            if not torch.all(x_condition):
                warn(
                    'Aliasing problems may occur in the AS method. '
                    'Consider reducing the distance or increasing the Nx*dx product.'
                )
            if not torch.all(y_condition):
                warn(
                    'Aliasing problems may occur in the AS method. '
                    'Consider reducing the distance or increasing the Ny*dy product.'
                )

        if method == 'fresnel':
            diagonal_squared = Lx**2 + Ly**2
            condition = distance**3 > k / 8 * (diagonal_squared)**2

            if not torch.all(condition):
                warn(
                    'The paraxial (near-axis) optics condition required for the Fresnel method is not satisfied. '
                    'Consider increasing the distance or decreasing the screen size.'
                )

    def impulse_response_angular_spectrum(self) -> torch.Tensor:
        """Creates the impulse response function for angular spectrum method

        Returns
        -------
        torch.Tensor
            2d impulse response function for angular spectrum method
        """

        # Fourier image of impulse response function,
        # 0 if k^2 < (kx^2 + ky^2) [if use_legacy_filter]
        # TODO: there is still no information in docstrings about a filter:(
        return self._low_pass_filter * torch.exp(
            (1j * self.distance) * self._wave_number_z
        )  # Comment: Here we use the following exponent: `exp(+ i * d * k)`

    def impulse_response_fresnel(self) -> torch.Tensor:
        """Creates the impulse response function for fresnel approximation

        Returns
        -------
        torch.Tensor
            2d impulse response function for fresnel approximation
        """

        # Fourier image of impulse response function
        # 0 if k^2 < (kx^2 + ky^2) [if use_legacy_filter]
        return self._low_pass_filter * torch.exp(
            (1j * self.distance) * self._wave_number_z_eff_fresnel
        ) * torch.exp(
            (1j * self.distance) * self._wave_number
        )

    def _impulse_response(self, tol: float = 1e-3) -> torch.Tensor:
        """Calculates the impulse response function based on selected method

        Parameters
        ----------
        tol : float, optional
            tolerance for auto method, by default 1e-3

        Returns
        -------
        torch.Tensor
            The impulse response function
        """

        if self.method == 'AS':
            return self.impulse_response_angular_spectrum()

        elif self.method == 'fresnel':
            return self.impulse_response_fresnel()

        # TODO: fix auto mod
        # TODO: Didn't we plan to exclude `auto` method at all?
        elif self.method == 'auto':

            radius_squared = self._x_grid**2 + self._y_grid**2

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

    # TODO: ask for tol parameter, maybe move it to init?
    def forward(
        self,
        input_field: Wavefront
    ) -> Wavefront:
        """Calculates the field after propagating in the free space

        Parameters
        ----------
        input_field : Wavefront
            Field before propagation in free space

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
            a=input_field_fft,  # example shape: (5, 'wavelength', 1, 'H', 'W')
            b=impulse_response_fft,  # example shape: ('wavelength', 'H', 'W')
            a_axis=self.simulation_parameters.axes.names,
            b_axis=self._calc_axes,
            preserve_a_axis=True  # check that the output has the input shape
        )  # example output shape: (5, 'wavelength', 1, 'H', 'W')

        output_field = torch.fft.ifft2(
            output_field_fft,
            dim=(self._h_index, self._w_index)
        )

        return output_field

    def reverse(self, transmission_field: Wavefront) -> Wavefront:
        # TODO: Check the description...
        """Calculate the field after it propagates in the free space
        in the backward direction.

        Parameters
        ----------
        transmission_field : Wavefront
            Field to be propagated in the backward direction

        Returns
        -------
        Wavefront
            Propagated in the backward direction field
        """

        transmission_field_fft = torch.fft.fft2(
            transmission_field,
            dim=(self._h_index, self._w_index)
        )

        impulse_response_fft = self._impulse_response().conj()

        # Fourier image of output field
        incident_field_fft, _ = tensor_dot(
            a=transmission_field_fft,  # example shape: (5, 'wavelength', 1, 'H', 'W')
            b=impulse_response_fft,  # example shape: ('wavelength', 'H', 'W')
            a_axis=self.simulation_parameters.axes.names,
            b_axis=self._calc_axes,
            preserve_a_axis=True  # check that the output has the first input shape
        )  # example output shape: (5, 'wavelength', 1, 'H', 'W')

        incident_field = torch.fft.ifft2(
            incident_field_fft,
            dim=(self._h_index, self._w_index)
        )

        return incident_field
