from abc import ABCMeta, abstractmethod
from torch import nn
from torch import Tensor
from .simulation_parameters import SimulationParameters
from .specs import ReprRepr, ParameterSpecs
from typing import Iterable, Literal
from .parameters import BoundedParameter, Parameter
import torch


INNER_PARAMETER_SUFFIX = '_svtlnn_inner_parameter'


class Element(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        simulation_parameters: SimulationParameters
    ) -> None:
        super().__init__()

        self.simulation_parameters = simulation_parameters

        self._x_size = self.simulation_parameters._x_size
        self._y_size = self.simulation_parameters._y_size
        self._x_nodes = self.simulation_parameters._x_nodes
        self._y_nodes = self.simulation_parameters._y_nodes
        self._wavelength = self.simulation_parameters._wavelength

        self._x_linspace = torch.linspace(
            -self._x_size/2, self._x_size/2, self._x_nodes
        )
        self._y_linspace = torch.linspace(
            -self._y_size/2, self._y_size/2, self._y_nodes
        )
        self._x_grid, self._y_grid = torch.meshgrid(
            self._x_linspace, self._y_linspace, indexing='xy'
        )

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

    def __setattr__(self, name: str, value: Tensor | nn.Module) -> None:

        # BoundedParameter and Parameter are handled by pointing
        # auxiliary attribute on them with a name plus INNER_PARAMETER_SUFFIX
        if isinstance(value, (BoundedParameter, Parameter)):
            super().__setattr__(
                name + INNER_PARAMETER_SUFFIX, value.inner_parameter
            )

        return super().__setattr__(name, value)


class Aperture(Element):
    """Aperture of the optical element with transmissin function, which takes
    the value 0 or 1

    :param Element: Parent class for class Aperture
    :type Element: class
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        mask: torch.Tensor
    ):
        """Constructor method

        :param simulation_parameters: Class exemplar, that describes optical
        system
        :type simulation_parameters: SimulationParameters
        :param mask: Tensor that describes transmission function
        :type mask: torch.Tensor
        """

        super().__init__(simulation_parameters)
        self.mask = mask

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Method that callulates the field after propagating through the
        aperture

        :param input_field: The incident field in the plane on an aperture of
                            the system(field before the aperture)
        :type input_field: torch.Tensor
        :return: The field after propagating through the aperture
        :rtype: torch.Tensor
        """
        return input_field * self.mask


class RectangularAperture(Element):
    """A rectangle-shaped aperture with a transmittance function taking either
      a value of 0 or 1

    :param Element: Parent class for class RectangularAperture
    :type Element: class
    """
    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        length: float,
        width: float
    ):
        """Constructor method

        :param simulation_parameters: Class exemplar, that describes optical
        system
        :type simulation_parameters: SimulationParameters
        :param length: length of the rectangle
        :type length: float
        :param width: width of the rectangle
        :type width: float
        """

        super().__init__(simulation_parameters)
        self.length = length
        self.width = width
        self.transmission_function = ((torch.abs(
            self._x_grid) <= self.length/2) * (torch.abs(
                self._y_grid) <= self.width/2)).float()

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Method that callulates the field after propagating through the
        rectangular aperture

        :param input_field: The incident field in the plane on an aperture of
                            the system(field before the aperture)
        :type input_field: torch.Tensor
        :return: The field after propagating through the aperture
        :rtype: torch.Tensor
        """
        return input_field*self.transmission_function

    def get_transmission_function(self) -> torch.Tensor:
        """Method which returns the transmission function of
        the rectangular aperture

        :return: transmission function of the rectangular aperture
        :rtype: torch.Tensor
        """
        return self.transmission_function


class RoundAperture(Element):
    """A round-shaped aperture with a transmittance function taking either
      a value of 0 or 1

    :param Element: Parent class for class RoundAperture
    :type Element: class
    """
    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        radius: float
    ):
        """Constructor method

        :param simulation_parameters: Class exemplar, that describes optical
        system
        :type simulation_parameters: SimulationParameters
        :param radius: circle radius
        :type length: float
        """

        super().__init__(simulation_parameters)
        self.radius = radius
        self.transmission_function = ((torch.pow(self._x_grid, 2) + torch.pow(
            self._y_grid, 2)) <= self.radius**2).float()

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Method that calculates the field after propagating through the
        round aperture

        :param input_field: The incident field in the plane on an aperture of
                            the system(field before the aperture)
        :type input_field: torch.Tensor
        :return: The field after propagating through the aperture
        :rtype: torch.Tensor
        """
        return input_field*self.transmission_function

    def get_transmission_function(self) -> torch.Tensor:
        """Method which returns the transmission function of
        the round aperture

        :return: transmission function of the round aperture
        :rtype: torch.Tensor
        """
        return self.transmission_function


class ThinLens(Element):
    """ A class that described the field after propagating through the
    thin lens

    :param Element: Parent class for class Thin Lens
    :type Element: class
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        focal_length: float,
        radius: float
    ):
        """Constructor method

        :param simulation_parameters: Class exemplar, that describes optical
        system
        :type simulation_parameters: SimulationParameters
        :param focal_length: focal length of the lens, greater than 0 for the
                             collecting lens
        :type focal_length: float
        :param radius: radius of the thin lens
        :type radius: float
        """

        super().__init__(simulation_parameters)

        self.focal_length = focal_length
        self.radius = radius
        self._wave_number = 2 * torch.pi/self._wavelength
        self._radius_squared = torch.pow(self._x_grid, 2) + torch.pow(
            self._y_grid, 2)
        # TODO: check if .float is required
        self.transmission_function = torch.exp(1j * (-self._wave_number/(
            2 * self.focal_length) * self._radius_squared * (
                (self._radius_squared <= self.radius**2).float())))

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Method that calculates the field after propagating through the
        thin lens

        :param input_field: The incident field in the plane on a thin lens of
                            the system(field before the thin lens)
        :type input_field: torch.Tensor
        :return: The field after propagating through the thin lens
        :rtype: torch.Tensor
        """
        return input_field * self.transmission_function

    def get_transmission_function(self):
        """Method which returns the complex-valued transmission function of
        the thin lens

        :return: transmission function of the thin lens
        :rtype: torch.Tensor
        """
        return self.transmission_function


class SpatialLightModulator(Element):
    """A class that described the field after propagating through the
    Spatial Light Modulator with a given phase function

    :param Element: Parent class for the class SpatialLightModulator
    :type Element: class
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        mask: torch.Tensor,
        number_of_levels: int = 256
    ):
        """Constructor method

        :param simulation_parameters: Class exemplar, that describes optical
        system
        :type simulation_parameters: SimulationParameters
        :param mask: phase mask in grey format for the SLM
        :type mask: torch.Tensor
        :param number_of_levels: number of phase quantisation levels for the
                                 SLM, default=256
        :type number_of_levels: int
        """

        super().__init__(simulation_parameters)

        self.mask = mask
        self.number_of_levels = number_of_levels

        self.transmission_function = torch.exp(
            1j * 2 * torch.pi / self.number_of_levels * self.mask
        )

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Method that calculates the field after propagating through the SLM

        :param input_field: The incident field in the plane on a thin lens of
                            the system(field before the thin lens)
        :type input_field: torch.Tensor
        :return: The field after propagating through the SLM
        :rtype: torch.Tensor
        """
        return input_field * self.transmission_function

    def get_transmission_function(self):
        """Method which returns the complex-valued transmission function of
        the Spartial Light Modulator, which takes values from 0 to 2*pi

        :return: transmission function of the SLM
        :rtype: torch.Tensor
        """
        return self.transmission_function


class FreeSpace(Element):
    """A class that describes the propagating of the field in free space
    before two optical elements

    :param Element: Parent class for class FreeSpace
    :type Element: class
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        distance: float,
        method: Literal['auto', 'fresnel', 'AS']
    ):
        """Constructor method

        :param simulation_parameters: Class exemplar, that describes optical
        system
        :type simulation_parameters: SimulationParameters
        :param distance: distance between two optical elements
        :type distance: float
        :param wavelength: wavelength of the incident beam
        :type wavelength: float
        :param method: method for calculating the field after propagating,
                       default = 'fresnel'
        :type method: str
        """
        super().__init__(simulation_parameters)

        self.distance = distance
        self.method = method

        self._wave_number = 2*torch.pi/self._wavelength

        # spatial frequencies
        self._kx_linear = torch.fft.fftfreq(self._x_nodes, torch.diff(
            self._x_linspace)[0]) * (2 * torch.pi)
        self._ky_linear = torch.fft.fftfreq(self._y_nodes, torch.diff(
            self._y_linspace)[0]) * (2 * torch.pi)
        self._kx_grid, self._ky_grid = torch.meshgrid(
            self._kx_linear, self._ky_linear, indexing='xy')

        self.low_pass_filter = (torch.pow(self._kx_grid, 2) + torch.pow(
            self._ky_grid, 2) <= self._wave_number**2).float()

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Method that calculates the field after propagating in the free space

        :param input_field: The incident field in the plane on a thin lens of
                            the system(field before the thin lens)
        :type input_field: torch.Tensor
        :return: The field after propagating in the free space
        :rtype: torch.Tensor
        """

        input_field_fft = torch.fft.fft2(input_field)

        if self.method == 'AS':

            wave_number_z = torch.sqrt(
                self._wave_number**2 - torch.pow(self._kx_grid, 2) - torch.pow(self._ky_grid, 2)  # noqa: E501
            )
            # Fourier image of impulse response function
            impulse_response_fft = self.low_pass_filter * torch.exp(
                1j * self.distance * wave_number_z
            )

        elif self.method == 'fresnel':

            wave_number_in_plane = torch.pow(self._kx_grid, 2) + torch.pow(self._ky_grid, 2)  # noqa: E501

            # Fourier image of impulse response function
            impulse_response_fft = -self.low_pass_filter * torch.exp(
                1j * self.distance * (self._wave_number - self._wavelength / (4 * torch.pi) * wave_number_in_plane)  # noqa: E501
            )
        elif self.method == 'auto':

            radius_squared = torch.pow(self._x_grid, 2) + torch.pow(
                self._y_grid, 2)

            fresnel_criterion = torch.pi * torch.max(
                torch.pow(radius_squared, 2)
            ) / (
                4 * self._wavelength * (self.distance**3)
            )

            if fresnel_criterion <= 1e-3:

                impulse_response_fft = -torch.exp(1j * torch.tensor(
                    self._wave_number * self.distance)) * torch.exp(-1j * (
                     self._wavelength * self.distance/(4 * torch.pi)) * (
                        torch.pow(self._kx_grid, 2)+torch.pow(self._ky_grid, 2
                                                              )))*(
                            self.low_pass_filter)
            else:

                impulse_response_fft = self.low_pass_filter*torch.exp(
                    1j*self.distance*torch.sqrt(self._wave_number**2 -
                                                torch.pow(self._kx_grid, 2) -
                                                torch.pow(self._ky_grid, 2)))
        else:
            raise ValueError("Unknown forward propagation method")

        # Fourier image of output field
        output_field_fft = input_field_fft * impulse_response_fft

        output_field = torch.fft.ifft2(output_field_fft)

        return output_field
