import torch

from .element import Element
from ..simulation_parameters import SimulationParameters
from ..parameters import OptimizableTensor
from ..wavefront import Wavefront
from abc import ABC, abstractmethod
from typing import Iterable
from ..specs import ImageRepr, PrettyReprRepr, ParameterSpecs
from ..visualization import ElementHTML, jinja_env


class AbstractMulElement(Element, ABC):
    r"""
    Class that generalize all apertures with $E^\text{out} = \hat{T}E^\text{in}$ like forward function,
    where $\hat{T}$ is transmission function.
    """

    @property
    @abstractmethod
    def transmission_function(self) -> torch.Tensor:
        r"""
        The tensor representing transmission function of the element, $\hat{T}$.
        The shape of the transmission function should be broadcastable to the shape of the incident wavefront.
        To achive this, one can use `SimpulationParameters.cast` method to cast the transmission function to the shape of the incident wavefront:
        ```python linenums="0"
        @property
        def transmission_function(self) -> torch.Tensor:
            T = ...  # tensor with shape (Ny, Nx)
            return self.simulation_parameters.cast(T, "y", "x")
        ```
        """

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        return incident_wavefront * self.transmission_function

    @staticmethod
    def _widget_html_(
        index: int, name: str, element_type: str | None, subelements: list[ElementHTML]
    ) -> str:
        return jinja_env.get_template("widget_aperture.html.jinja").render(
            index=index, name=name, subelements=subelements
        )


class Aperture(AbstractMulElement):
    def __init__(
        self, simulation_parameters: SimulationParameters, mask: OptimizableTensor
    ):
        r"""
        Aperture defined by mask tensor.
        Commonly, the mask is a tensor with values of either 0 or 1,
        where 0 represents blocked light and 1 represents allowed light.

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Simulation parameters.
        mask : torch.Tensor
            Two-dimensional tensor representing the aperture mask of shape `(Ny, Nx)`.
            The mask works as following:

            $$E^\text{out}_{xyw...} = \text{mask}_{xy} E^\text{in}_{xyw...}$$

            In this case 0 blocks light and 1 allows light go through.
        """

        super().__init__(simulation_parameters=simulation_parameters)

        self.mask = self.process_parameter("mask", mask)

    @property
    def transmission_function(self) -> torch.Tensor:
        return self.simulation_parameters.cast(self.mask, "y", "x")

    def to_specs(self) -> Iterable[ParameterSpecs]:
        return [
            ParameterSpecs(
                "mask",
                [
                    PrettyReprRepr(self.mask),
                    ImageRepr(self.mask.numpy(force=True)),
                ],
            )
        ]


class RectangularAperture(AbstractMulElement):
    def __init__(
        self, simulation_parameters: SimulationParameters, height: float, width: float
    ):
        """Rectangular aperture.
        Through the rectangular area of defined height and width located in the
        center the light is allowed to pass, otherwise blocked.

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Simulation parameters.
        height : float
            Aperture height.
        width : float
            Aperture width.
        """
        super().__init__(simulation_parameters=simulation_parameters)

        self.height = self.process_parameter("height", height)
        self.width = self.process_parameter("width", width)

        _x_grid, _y_grid = self.simulation_parameters.meshgrid(x_axis="x", y_axis="y")

        _mask = self.simulation_parameters.cast(
            (torch.abs(_x_grid) <= self.width / 2)
            * (torch.abs(_y_grid) <= self.height / 2)
            + 0.0,  # to convert bool to float
            "y",
            "x",
        )
        self._mask = self.make_buffer("_mask", _mask)

    @property
    def transmission_function(self) -> torch.Tensor:
        return self._mask

    def to_specs(self) -> Iterable[ParameterSpecs]:
        return [
            ParameterSpecs("height", [PrettyReprRepr(self.height)]),
            ParameterSpecs("width", [PrettyReprRepr(self.width)]),
        ]


class RoundAperture(AbstractMulElement):
    def __init__(self, simulation_parameters: SimulationParameters, radius: float):
        """Round-shaped aperture.
        Through the round area of defined radius located in the center
        the light is allowed to pass, otherwise blocked.

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Simulation parameters.
        radius : float
            Radius of the round-shaped aperture.
        """
        super().__init__(simulation_parameters=simulation_parameters)

        self.radius = self.process_parameter("radius", radius)

        _x_grid, _y_grid = self.simulation_parameters.meshgrid(x_axis="x", y_axis="y")

        _mask = self.simulation_parameters.cast(
            (_x_grid**2 + _y_grid**2 <= self.radius**2)
            + 0.0,  # to convert bool to float
            "y",
            "x",
        )
        self._mask = self.make_buffer("_mask", _mask)

    @property
    def transmission_function(self) -> torch.Tensor:
        return self._mask

    def to_specs(self) -> Iterable[ParameterSpecs]:
        return [ParameterSpecs("radius", [PrettyReprRepr(self.radius)])]
