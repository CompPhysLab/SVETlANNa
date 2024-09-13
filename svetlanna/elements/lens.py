from .element import Element
from ..simulation_parameters import SimulationParameters
import torch


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

    def reverse(self, transmission_field: torch.Tensor) -> torch.Tensor:
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
