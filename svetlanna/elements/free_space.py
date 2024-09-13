from .element import Element
from ..simulation_parameters import SimulationParameters
import torch
from typing import Literal


# TODO: check docstrings
class FreeSpace(Element):
    """A class that describes the propagating of the field in free space
    before two optical elements

    Parameters
    ----------
    Element : _type_
        _description_
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        distance: float,
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

        self._wave_number = 2 * torch.pi / self._wavelength

        # spatial frequencies
        self._kx_linear = torch.fft.fftfreq(self._x_nodes, torch.diff(
            self._x_linspace)[0]) * (2 * torch.pi)
        self._ky_linear = torch.fft.fftfreq(self._y_nodes, torch.diff(
            self._y_linspace)[0]) * (2 * torch.pi)
        self._kx_grid, self._ky_grid = torch.meshgrid(
            self._kx_linear, self._ky_linear, indexing='xy')

        # low-pass filter
        self.low_pass_filter = 1. * (torch.pow(self._kx_grid, 2) + torch.pow(
            self._ky_grid, 2) <= self._wave_number**2)

    def impulse_response_angular_spectrum(self) -> torch.Tensor:
        """Create the impulse response function for angular spectrum method

        Returns
        -------
        torch.Tensor
            2d impulse response function for angular spectrum method
        """

        wave_number_z = torch.sqrt(
                self._wave_number**2 - torch.pow(self._kx_grid, 2) - torch.pow(self._ky_grid, 2)  # noqa: E501
            )

        # Fourier image of impulse response function
        impulse_response_fft = self.low_pass_filter * torch.exp(
            1j * self.distance * wave_number_z
        )
        return impulse_response_fft

    def impulse_response_fresnel(self) -> torch.Tensor:
        """Create the impulse response function for fresnel approximation

        Returns
        -------
        torch.Tensor
            2d impulse response function for fresnel approximation
        """

        wave_number_in_plane = torch.pow(self._kx_grid, 2) + torch.pow(self._ky_grid, 2)  # noqa: E501

        # Fourier image of impulse response function
        impulse_response_fft = - self.low_pass_filter * torch.exp(
            1j * self.distance * (self._wave_number - self._wavelength / (4 * torch.pi) * wave_number_in_plane)  # noqa: E501
        )
        return impulse_response_fft

    # TODO: ask for tol parameter
    def forward(
        self,
        input_field: torch.Tensor,
        tol: float = 1e-3
    ) -> torch.Tensor:
        """Method that calculates the field after propagating in the free space

        Parameters
        ----------
        input_field : torch.Tensor
            Field before propagation in free space
        tol : float, optional
            tolerance for Fresnel approximation, by default 1e-3

        Returns
        -------
        torch.Tensor
            Field after propagation in free space

        Raises
        ------
        ValueError
            Occurs when a non-existent direct distribution method is chosen
        """

        input_field_fft = torch.fft.fft2(input_field)

        if self.method == 'AS':

            impulse_response_fft = self.impulse_response_angular_spectrum()

        elif self.method == 'fresnel':

            impulse_response_fft = self.impulse_response_fresnel()

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
                impulse_response_fft = self.impulse_response_fresnel()
            else:

                impulse_response_fft = self.impulse_response_angular_spectrum
        else:
            raise ValueError("Unknown forward propagation method")

        # Fourier image of output field
        output_field_fft = input_field_fft * impulse_response_fft

        output_field = torch.fft.ifft2(output_field_fft)

        return output_field

    def reverse(self, transmission_field: torch.Tensor) -> torch.Tensor:
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

        transmission_field_fft = torch.fft.fft2(transmission_field)

        # square of the modulus of the wave vector in the plane oXY
        wave_number_in_plane = torch.pow(self._kx_grid, 2) + torch.pow(self._ky_grid, 2)  # noqa: E501

        impulse_response = self.low_pass_filter * torch.exp(
            1j * self.distance * (
                -self._wave_number + wave_number_in_plane / (2 * self._wave_number)  # noqa: E501
            )
        )

        incident_field_fft = impulse_response * transmission_field_fft
        incident_field = torch.fft.ifft2(incident_field_fft)

        return incident_field
