from .element import Element
from ..simulation_parameters import SimulationParameters
import torch


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
