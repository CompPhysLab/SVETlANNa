import torch

from .element import Element
from ..simulation_parameters import SimulationParameters
from ..parameters import OptimizableTensor
from ..wavefront import Wavefront, mul
from abc import ABC, abstractmethod
from typing import Iterable
from ..specs import ImageRepr, PrettyReprRepr, ParameterSpecs


class MulElement(Element, ABC):
    """Class that generalize all elements with E->T@E like forward function,
    where T is transmission function
    """

    @abstractmethod
    def get_transmission_function(self) -> torch.Tensor:
        """Method which returns the transmission function of
        the element

        Returns
        -------
        torch.Tensor
            transmission function
        """

    @property
    @abstractmethod
    def transmission_function_axes(self) -> tuple[str, ...]:
        """Axes of the transmission function.
        For example, `('H', 'W')`

        Returns
        -------
        tuple[str, ...]
            Axes
        """
        ...

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        """Calculate the field after propagating through the
        element

        Parameters
        ----------
        input_field : Wavefront
            Field incident on the aperture

        Returns
        -------
        Wavefront
            The field after propagating through the element
        """
        return mul(
            incident_wavefront,
            self.get_transmission_function(),
            self.transmission_function_axes,
            self.simulation_parameters
        )


# TODO: check docstring
class Aperture(MulElement):
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

        super().__init__(
            simulation_parameters=simulation_parameters
        )

        self.mask = self.process_parameter('mask', mask)
        self._calc_axes = ('H', 'W')

    @property
    def transmission_function_axes(self) -> tuple[str, ...]:
        return self._calc_axes

    def get_transmission_function(self) -> torch.Tensor:
        return self.mask

    def to_specs(self) -> Iterable[ParameterSpecs]:
        return [
            ParameterSpecs(
                'mask', [
                    PrettyReprRepr(self.mask),
                    ImageRepr(self.mask.numpy(force=True)),
                ]
            )
        ]


# TODO" check docstring
class RectangularAperture(MulElement):
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
            simulation_parameters=simulation_parameters
        )

        self.height = self.process_parameter('height', height)
        self.width = self.process_parameter('width', width)

        _x_grid, _y_grid = self.simulation_parameters.meshgrid(
            x_axis='W', y_axis='H'
        )

        self._calc_axes = ('H', 'W')
        self._mask = self.make_buffer(
            '_mask',
            (
                (
                    torch.abs(_x_grid) <= self.width/2
                ) * (
                    torch.abs(_y_grid) <= self.height/2
                )
            ).to(dtype=torch.get_default_dtype())
        )

    @property
    def transmission_function_axes(self) -> tuple[str, ...]:
        return self._calc_axes

    def get_transmission_function(self) -> torch.Tensor:
        return self._mask

    def to_specs(self) -> Iterable[ParameterSpecs]:
        return [
            ParameterSpecs(
                'height', [
                    PrettyReprRepr(self.height)
                ]
            ),
            ParameterSpecs(
                'width', [
                    PrettyReprRepr(self.width)
                ]
            )
        ]


# TODO: check docstrings
class RoundAperture(MulElement):
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
            simulation_parameters=simulation_parameters
        )

        self.radius = self.process_parameter('radius', radius)

        _x_grid, _y_grid = self.simulation_parameters.meshgrid(
            x_axis='W', y_axis='H'
        )

        self._calc_axes = ('H', 'W')
        self._mask = self.make_buffer(
            '_mask',
            (
                _x_grid**2 + _y_grid**2 <= self.radius**2
            ).to(dtype=torch.get_default_dtype())
        )

    @property
    def transmission_function_axes(self) -> tuple[str, ...]:
        return self._calc_axes

    def get_transmission_function(self) -> torch.Tensor:
        return self._mask

    def to_specs(self) -> Iterable[ParameterSpecs]:
        return [
            ParameterSpecs(
                'radius', [
                    PrettyReprRepr(self.radius)
                ]
            )
        ]
