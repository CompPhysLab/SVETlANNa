import torch

from .element import Element
from ..simulation_parameters import SimulationParameters
from ..parameters import OptimizableFloat
from ..wavefront import Wavefront
from typing import Iterable
from ..specs import PrettyReprRepr, ParameterSpecs
from ..visualization import jinja_env, ElementHTML


class ThinLens(Element):
    """A class that described the field after propagating through the
    thin lens.
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        focal_length: OptimizableFloat,
        radius: float = torch.inf,
    ):
        """Thin lens element.

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            An instance describing the optical system's simulation parameters.
        focal_length : OptimizableFloat
            The focal length of the lens.
            Must be greater than 0 for a converging lens.
        radius : float
            The radius of the thin lens.
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
        return torch.exp(
            -1j
            * self._radius_mask
            * self._radius_squared
            * (self._wave_number / (2 * self.focal_length))
        )

    def get_transmission_function(self) -> torch.Tensor:
        """Returns the transmission function of the thin lens.

        Returns
        -------
        torch.Tensor
            The transmission function of the thin lens.
        """

        return self.transmission_function

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        """Calculates the field after propagation through the thin lens.

        Parameters
        ----------
        input_field : Wavefront
            The field incident on the thin lens.

        Returns
        -------
        Wavefront
            The field after propagation through the thin lens.
        """
        return incident_wavefront * self.transmission_function

    def reverse(self, transmission_wavefront: Wavefront) -> Wavefront:
        """Calculates the field after passing through the lens during
        back propagation.

        Parameters
        ----------
        transmission_field : Wavefront
            The field incident on the lens during back propagation.
            This corresponds to the transmitted field in forward propagation.

        Returns
        -------
        Wavefront
            The field transmitted through the lens during back propagation.
            This corresponds to the incident field in forward propagation.
        """
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
