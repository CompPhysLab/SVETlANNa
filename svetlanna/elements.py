from abc import ABCMeta, abstractmethod
from torch import nn
from torch import Tensor
from .simulation_parameters import SimulationParameters
from .specs import ReprRepr, ParameterSpecs
from typing import Iterable, Literal
from .parameters import BoundedParameter, Parameter
import torch

INNER_PARAMETER_SUFFIX = '_svtlnn_inner_parameter'


# TODO: check docstring
class Element(nn.Module, metaclass=ABCMeta):
    """A class that describes each element of the system

    Parameters
    ----------
    nn : _type_
        _description_
    metaclass : _type_, optional
        _description_, by default ABCMeta
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters
    ) -> None:
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Class exemplar that describes the optical system
        """

        super().__init__()

        self.simulation_parameters = simulation_parameters

        self._x_size = self.simulation_parameters.x_size
        self._y_size = self.simulation_parameters.y_size
        self._x_nodes = self.simulation_parameters.x_nodes
        self._y_nodes = self.simulation_parameters.y_nodes
        self._wavelength = self.simulation_parameters.wavelength

        self._x_linspace = torch.linspace(
            -self._x_size/2, self._x_size/2, self._x_nodes
        )
        self._y_linspace = torch.linspace(
            -self._y_size/2, self._y_size/2, self._y_nodes
        )
        self._x_grid, self._y_grid = torch.meshgrid(
            self._x_linspace, self._y_linspace, indexing='xy'
        )

    # TODO: check doctrings
    @abstractmethod
    def forward(self, Ein: Tensor) -> Tensor:

        """Forward propagation through the optical element"""

    def to_specs(self) -> Iterable[ParameterSpecs]:

        """Create specs"""

        for (name, parameter) in self.named_parameters():

            # BoundedParameter and Parameter support
            if name.endswith(INNER_PARAMETER_SUFFIX):
                name = name.removesuffix(INNER_PARAMETER_SUFFIX)
                parameter = self.__getattribute__(name)

            yield ParameterSpecs(
                name=name,
                representations=(ReprRepr(value=parameter),)
            )

    # TODO: create docstrings
    def __setattr__(self, name: str, value: Tensor | nn.Module) -> None:

        # BoundedParameter and Parameter are handled by pointing
        # auxiliary attribute on them with a name plus INNER_PARAMETER_SUFFIX
        if isinstance(value, (BoundedParameter, Parameter)):
            super().__setattr__(
                name + INNER_PARAMETER_SUFFIX, value.inner_parameter
            )

        return super().__setattr__(name, value)


# TODO: check docstring
class Aperture(Element):
    """Aperture of the optical element with transmission function, which takes
    the value 0 or 1

    Parameters
    ----------
    Element : _type_
        _description_
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        mask: torch.Tensor
    ):
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Class exemplar that describes the optical system
        mask : torch.Tensor
            Tensor that describes 2d transmission function
        """

        super().__init__(simulation_parameters)

        self.mask = mask

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Method that calculates the field after propagating through the
        aperture

        Parameters
        ----------
        input_field : torch.Tensor
            Field incident on the aperture

        Returns
        -------
        torch.Tensor
            The field after propagating through the aperture
        """

        return input_field * self.mask

    def get_transmission_function(self) -> torch.Tensor:
        """Method which returns the transmission function of
        the aperture

        Returns
        -------
        torch.Tensor
            transmission function of the aperture
        """

        return self.mask


# TODO" check docstring
class RectangularAperture(Element):
    """A rectangle-shaped aperture with a transmission function taking either
      a value of 0 or 1

    Parameters
    ----------
    Element : _type_
        _description_
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        height: float,
        width: float
    ):
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Class exemplar, that describes optical system
        height : float
            aperture height
        width : float
            aperture width
        """

        super().__init__(simulation_parameters)

        self.height = height
        self.width = width

        self.transmission_function = ((torch.abs(
            self._x_grid) <= self.width/2) * (torch.abs(
                self._y_grid) <= self.height/2)).float()

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Method that calculates the field after propagating through the
        rectangular aperture

        Parameters
        ----------
        input_field : torch.Tensor
            Field incident on the rectangular aperture

        Returns
        -------
        torch.Tensor
            The field after propagating through the rectangular aperture
        """

        return input_field * self.transmission_function

    def get_transmission_function(self) -> torch.Tensor:
        """Method which returns the transmission function of
        the rectangular aperture

        Returns
        -------
        torch.Tensor
            transmission function of the rectangular aperture
        """

        return self.transmission_function


# TODO: check docstrings
class RoundAperture(Element):
    """A round-shaped aperture with a transmission function taking either
      a value of 0 or 1

    Parameters
    ----------
    Element : _type_
        _description_
    """
    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        radius: float
    ):
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Class exemplar, that describes optical system
        radius : float
            Radius of the round-shaped aperture
        """

        super().__init__(simulation_parameters)

        self.radius = radius
        self.transmission_function = ((torch.pow(self._x_grid, 2) + torch.pow(
            self._y_grid, 2)) <= self.radius**2).float()

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Method that calculates the field after propagating through the
        round-shaped aperture

        Parameters
        ----------
        input_field : torch.Tensor
            Field incident on the round-shaped aperture

        Returns
        -------
        torch.Tensor
            The field after propagating through the round-shaped aperture
        """

        return input_field * self.transmission_function

    def get_transmission_function(self) -> torch.Tensor:
        """Method which returns the transmission function of
        the round-shaped aperture

        Returns
        -------
        torch.Tensor
            transmission function of the round-shaped aperture
        """

        return self.transmission_function


# TODO: check docstrings
class ThinLens(Element):
    """A class that described the field after propagating through the
    thin lens

    Parameters
    ----------
    Element : _type_
        _description_
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        focal_length: float,
        radius: float
    ):
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Class exemplar, that describes optical system
        focal_length : float
            focal length of the lens, greater than 0 for the collecting lens
        radius : float
            radius of the thin lens
        """

        super().__init__(simulation_parameters)

        self.focal_length = focal_length
        self.radius = radius

        self._wave_number = 2 * torch.pi / self._wavelength
        self._radius_squared = torch.pow(self._x_grid, 2) + torch.pow(
            self._y_grid, 2)

        self.transmission_function = torch.exp(1j * (-self._wave_number/(
            2 * self.focal_length) * self._radius_squared * (
                (self._radius_squared <= self.radius**2))))

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Method that calculates the field after propagating through the
        thin lens

        Parameters
        ----------
        input_field : torch.Tensor
            Field incident on the thin lens

        Returns
        -------
        torch.Tensor
            The field after propagating through the thin lens
        """

        return input_field * self.transmission_function

    def reverse(self, transmission_field: torch.Tensor) -> torch.tensor:
        """Method that calculates the field after passing the lens in back
        propagation

        Parameters
        ----------
        transmission_field : torch.Tensor
            Field incident on the lens in back propagation
            (transmitted field in forward propagation)

        Returns
        -------
        torch.tensor
            Field transmitted on the lens in back propagation
            (incident field in forward propagation)
        """
        return transmission_field * torch.conj(self.transmission_function)

    def get_transmission_function(self):
        """Method which returns the transmission function of
        the thin lens

        Returns
        -------
        torch.Tensor
            transmission function of the thin lens
        """

        return self.transmission_function


# TODO: check docstrings
class SpatialLightModulator(Element):
    """A class that described the field after propagating through the
    Spatial Light Modulator with a given phase mask

    Parameters
    ----------
    Element : _type_
        _description_
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        mask: torch.Tensor,
        number_of_levels: int = 256
    ):
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Class exemplar, that describes optical system
        mask : torch.Tensor
            Phase mask in grey format for the SLM, every element must be int
        number_of_levels : int, optional
            Number of phase quantization levels for the SLM, by default 256
        """

        super().__init__(simulation_parameters)

        self.mask = mask
        self.number_of_levels = number_of_levels

        self.transmission_function = torch.exp(
            1j * 2 * torch.pi / self.number_of_levels * self.mask
        )

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Method that calculates the field after propagating through the SLM

        Parameters
        ----------
        input_field : torch.Tensor
            Field incident on the SLM

        Returns
        -------
        torch.Tensor
            The field after propagating through the SLM
        """

        return input_field * self.transmission_function

    def reverse(self, transmission_field: torch.Tensor) -> torch.tensor:
        """Method that calculates the field after passing the SLM in back
        propagation

        Parameters
        ----------
        transmission_field : torch.Tensor
            Field incident on the SLM in back propagation
            (transmitted field in forward propagation)

        Returns
        -------
        torch.tensor
            Field transmitted on the SLM in back propagation
            (incident field in forward propagation)
        """

        return transmission_field * torch.conj(self.transmission_function)

    def get_transmission_function(self):
        """Method which returns the transmission function of
        the SLM

        Returns
        -------
        torch.Tensor
            transmission function of the SLM
        """

        return self.transmission_function


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
