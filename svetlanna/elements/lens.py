import torch

from .element import Element
from ..simulation_parameters import SimulationParameters
from ..parameters import OptimizableFloat
from ..wavefront import Wavefront, mul


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
        focal_length: OptimizableFloat,
        radius: OptimizableFloat
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

        self._wave_number = 2 * torch.pi / self.simulation_parameters.__getitem__(  # noqa: E501
            axis='wavelength'
        )[..., None, None]

        self._x_linear = self.simulation_parameters.__getitem__(axis='W')
        self._y_linear = self.simulation_parameters.__getitem__(axis='H')

        # creating meshgrid
        self._x_grid = self._x_linear[None, :]
        self._y_grid = self._y_linear[:, None]

        self._x_grid = self._x_grid[None, ...]
        self._y_grid = self._y_grid[None, ...]

        self._radius_squared = torch.pow(self._x_grid, 2) + torch.pow(
            self._y_grid, 2)

        self.transmission_function = torch.exp(
            1j * (
                -self._wave_number / (2 * self.focal_length) * self._radius_squared * (     # noqaL E501
                    (self._radius_squared <= self.radius**2).to(
                        dtype=torch.get_default_dtype()
                    )
                )
            )
        )

    def get_transmission_function(self) -> torch.Tensor:
        """Method which returns the transmission function of
        the thin lens

        Returns
        -------
        torch.Tensor
            transmission function of the thin lens
        """

        return self.transmission_function

    def forward(self, input_field: Wavefront) -> Wavefront:
        """Method that calculates the field after propagating through the
        thin lens

        Parameters
        ----------
        input_field : Wavefront
            Field incident on the thin lens

        Returns
        -------
        Wavefront
            The field after propagating through the thin lens
        """

        # mul(
        #     input_field,
        #     self.transmission_function,
        #     ('wavelength', 'H', 'W'),
        #     self.simulation_parameters
        # )

        return input_field * self.transmission_function

    def reverse(self, transmission_field: torch.Tensor) -> Wavefront:
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
        return mul(
            transmission_field,
            torch.conj(self.transmission_function),
            ('H', 'W'),
            self.simulation_parameters
        )
