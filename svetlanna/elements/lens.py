import torch

from .element import Element
from ..simulation_parameters import SimulationParameters
from ..parameters import OptimizableFloat
from ..wavefront import Wavefront
from typing import Iterable
from ..specs import PrettyReprRepr, ParameterSpecs
from ..visualization import jinja_env, ElementHTML


class ThinLens(Element):
    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        focal_length: OptimizableFloat,
        radius: float = torch.inf,
    ):
        r"""Thin lens element.

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Simulation parameters.
        focal_length : OptimizableFloat
            The focal length of the lens.
            $\text{focal\_length} > 0$ for a converging lens.
        radius : float
            The radius of the thin lens.
            The field outside the radius ($x^2 + y^2 > \text{radius}^2$) will propagate with no change in phase.
            Default is infinity, meaning that the lens has no aperture and the field will propagate with a phase change everywhere.
        """

        super().__init__(simulation_parameters)

        self.focal_length = self.process_parameter("focal_length", focal_length)
        self.radius = self.process_parameter("radius", radius)

        wave_number = self.simulation_parameters.cast(
            2 * torch.pi / self.simulation_parameters.wavelength, "wavelength"
        )

        # Registering Buffer for _wave_number
        self._wave_number = self.make_buffer("_wave_number", wave_number)

        _x = self.simulation_parameters.cast(self.simulation_parameters.x, "x")
        _y = self.simulation_parameters.cast(self.simulation_parameters.y, "y")

        # Registering Buffer for _radius_squared
        self._radius_squared = self.make_buffer("_radius_squared", _x**2 + _y**2)

        # Create a mask that acts as an aperture:
        # Regions of the field where x^2 + y^2 > radius^2
        # will propagate with no change in phase.
        if self.radius == torch.inf:
            self._radius_mask: torch.Tensor | float = 1.0
        else:
            self._radius_mask = self.make_buffer(
                "_radius_mask",
                (self._radius_squared <= self.radius**2) + 0.0,  # cast bool to float
            )

    @property
    def transmission_function(self) -> torch.Tensor:
        r"""
        The tensor representing the transmission function of the element
        $\exp\left(-i \dfrac{k}{2f} (x^2 + y^2)\right)$,
        where $k$ is the wave number and $f$ is the focal length.
        The radius of the lens is taken into account.
        The shape of the tensor is broadcastable to the incident wavefront's shape.
        """
        return torch.exp(
            -1j
            * self._radius_mask
            * self._radius_squared
            * (self._wave_number / (2 * self.focal_length))
        )

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        return incident_wavefront * self.transmission_function

    def reverse(self, transmission_wavefront: Wavefront) -> Wavefront:
        return transmission_wavefront * torch.conj(self.transmission_function)

    def to_specs(self) -> Iterable[ParameterSpecs]:
        return [
            ParameterSpecs(
                "focal_length",
                [
                    PrettyReprRepr(self.focal_length),
                ],
            ),
            ParameterSpecs("radius", [PrettyReprRepr(self.radius)]),
        ]

    @staticmethod
    def _widget_html_(
        index: int, name: str, element_type: str | None, subelements: list[ElementHTML]
    ) -> str:
        return jinja_env.get_template("widget_lens.html.jinja").render(
            index=index, name=name, subelements=subelements
        )
