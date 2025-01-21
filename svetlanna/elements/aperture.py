import torch

from .element import Element
from ..simulation_parameters import SimulationParameters
from ..parameters import OptimizableTensor
from ..wavefront import Wavefront, mul


# TODO: check docstring
class Aperture(Element):
    """Aperture of the optical element with transmission function, which takes
    the value 0 or 1
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        mask: OptimizableTensor
    ):
        """Aperture of the optical element defined by mask tensor.

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Class exemplar that describes the optical system
        mask : torch.Tensor
            Two-dimensional tensor representing the aperture mask.
            Each element must be either 0 (blocks light) or 1 (allows light).
        """

        super().__init__(simulation_parameters)

        self.mask = self.process_parameter('mask', mask)

    def get_transmission_function(self) -> torch.Tensor:
        """Method which returns the transmission function of
        the aperture

        Returns
        -------
        torch.Tensor
            transmission function of the aperture
        """

        return self.mask

    def forward(self, input_field: Wavefront) -> Wavefront:
        """Method that calculates the field after propagating through the
        aperture

        Parameters
        ----------
        input_field : Wavefront
            Field incident on the aperture

        Returns
        -------
        Wavefront
            The field after propagating through the aperture
        """

        return mul(
            input_field,
            self.mask,
            ('H', 'W'),
            self.simulation_parameters
        )


# TODO" check docstring
class RectangularAperture(Aperture):
    """A rectangle-shaped aperture with a transmission function taking either
      a value of 0 or 1
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
        super().__init__(
            simulation_parameters=simulation_parameters,
            mask=torch.tensor(0)
        )

        self.height = self.process_parameter('height', height)
        self.width = self.process_parameter('width', width)
        self.mask = ((torch.abs(
            self._x_grid) <= self.width/2) * (torch.abs(
                self._y_grid) <= self.height/2)).to(
                    dtype=torch.get_default_dtype()
                )


# TODO: check docstrings
class RoundAperture(Aperture):
    """A round-shaped aperture with a transmission function taking either
      a value of 0 or 1
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
        super().__init__(
            simulation_parameters=simulation_parameters,
            mask=torch.tensor(0)
        )

        self.radius = self.process_parameter('radius', radius)
        self.mask = ((torch.pow(self._x_grid, 2) + torch.pow(
            self._y_grid, 2)) <= self.radius**2).to(
                dtype=torch.get_default_dtype()
            )
