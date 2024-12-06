from typing import Literal
import torch

from .element import Element
from ..simulation_parameters import SimulationParameters
from ..parameters import OptimizableFloat
from ..wavefront import Wavefront
from ..axes_math import tensor_dot


class FreeSpace(Element):
    """A class that describes the propagating of the field in free space
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
        method : Literal[&#39;auto&#39;, &#39;fresnel&#39;, &#39;AS&#39;]
            Method describing propagation in free space(AS - angular
            spectrum method, fresnel - fresnel approximation, auto - auto mode)
        """

        super().__init__(simulation_parameters)

        self.distance = distance
        self.method = method

        device = self.simulation_parameters.device

        self._w_index = self.simulation_parameters.axes.index('W')
        self._h_index = self.simulation_parameters.axes.index('H')

        _x_linear = self.simulation_parameters.axes.W
        _y_linear = self.simulation_parameters.axes.H

        x_nodes = _x_linear.shape[0]
        y_nodes = _y_linear.shape[0]

        # spatial frequencies calculation
        dx = _x_linear[1] - _x_linear[0] if x_nodes > 1 else 1.
        dy = _y_linear[1] - _y_linear[0] if y_nodes > 1 else 1.

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

        # (kx^2+ky^2) / k^2 relation
        k = 2 * torch.pi / self.simulation_parameters.axes.wavelength
        kx2ky2 = kx_grid**2 + ky_grid**2
        relation, relation_axes = tensor_dot(
            1 / k**2,
            kx2ky2,
            'wavelength',
            ('H', 'W')
        )

        # low pass filter, (kx^2 + ky^2) <= k^2
        self._low_pass_filter: torch.Tensor | int
        use_filter = False
        if use_filter:
            self.register_buffer(
                '_low_pass_filter',
                (relation <= 1).to(kx_grid),
                persistent=False
            )
        else:
            self._low_pass_filter = 1

        self._wave_number: torch.Tensor
        self.register_buffer(
            '_wave_number',
            k[..., None, None],
            persistent=False
        )

        self._calc_axes = relation_axes  # axes tuple used during calculations
        _wave_number_x2y2 = self._low_pass_filter * kx2ky2

        # kz
        if use_filter:
            # kz =
            #      sqrt(k^2-(kx^2 + ky^2)) if (kx^2 + ky^2) <= k^2,
            #      kz=|k| otherwise
            wave_number_z = torch.sqrt(
                self._wave_number**2 - _wave_number_x2y2 * self._low_pass_filter  # noqa: E501
            )
        else:
            wave_number_z = torch.sqrt(
                self._wave_number**2 - _wave_number_x2y2 + 0j
            )
        self._wave_number_z: torch.Tensor
        self.register_buffer(
            '_wave_number_z',
            wave_number_z,
            persistent=False
        )

        # kz taylored, used by Fresnel approximation
        self._wave_number_z_eff_fresnel: torch.Tensor
        self.register_buffer(
            '_wave_number_z_eff_fresnel',
            - 0.5 * _wave_number_x2y2 / self._wave_number,
            persistent=False
        )

    def impulse_response_angular_spectrum(self) -> torch.Tensor:
        """Create the impulse response function for angular spectrum method

        Returns
        -------
        torch.Tensor
            2d impulse response function for angular spectrum method
        """

        # Fourier image of impulse response function,
        # 0 if k^2 < (kx^2 + ky^2) [if use_filter]
        return self._low_pass_filter * torch.exp(
            1j * self.distance * self._wave_number_z
        )

    def impulse_response_fresnel(self) -> torch.Tensor:
        """Create the impulse response function for fresnel approximation

        Returns
        -------
        torch.Tensor
            2d impulse response function for fresnel approximation
        """

        # Fourier image of impulse response function
        return self._low_pass_filter * torch.exp(
            1j * self.distance * self._wave_number_z_eff_fresnel
        ) * torch.exp(
            1j * self.distance * self._wave_number
        )

    def _impulse_response(self, tol: float = 1e-3) -> torch.Tensor:
        if self.method == 'AS':
            return self.impulse_response_angular_spectrum()

        elif self.method == 'fresnel':
            return self.impulse_response_fresnel()

        # TODO: fix auto mod
        elif self.method == 'auto':

            radius_squared = torch.pow(self._x_grid, 2) + torch.pow(
                self._y_grid, 2)

            # criterion for Fresnel approximation
            fresnel_criterion = torch.pi * torch.max(
                torch.pow(radius_squared, 2)
            ) / (
                4 * self._wavelength * (self.distance**3)
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
            input_field, dim=(self._h_index, self._w_index)
        )

        impulse_response_fft = self._impulse_response()

        # Fourier image of output field
        output_field_fft, _ = tensor_dot(
            input_field_fft,
            impulse_response_fft,
            self.simulation_parameters.axes.names,
            self._calc_axes,
            preserve_a_axis=True
        )

        output_field = torch.fft.ifft2(
            output_field_fft, dim=(self._h_index, self._w_index)
        )

        return output_field

    def reverse(self, transmission_field: torch.Tensor) -> Wavefront:
        """Method that calculates the field after propagating in the free space
        in back propagation

        Parameters
        ----------
        transmission_field : torch.Tensor
            Field before propagation in free space in back propagation

        Returns
        -------
        torch.Tensor
            Field after propagation in free space in back propagation
        """

        transmission_field_fft = torch.fft.fft2(
            transmission_field, dim=(self._h_index, self._w_index)
        )

        # TODO: check physics. Not sure that conj() is enough
        impulse_response_fft = self._impulse_response().conj()

        # Fourier image of output field
        incident_field_fft, _ = tensor_dot(
            transmission_field_fft,
            impulse_response_fft,
            self.simulation_parameters.axes.names,
            self._calc_axes,
            preserve_a_axis=True
        )

        incident_field = torch.fft.ifft2(
            incident_field_fft, dim=(self._h_index, self._w_index)
        )

        return incident_field
