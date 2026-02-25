import torch
from .element import Element
from ..simulation_parameters import SimulationParameters
from ..wavefront import Wavefront
from ..parameters import OptimizableTensor
from typing import Iterable
from ..specs import ImageRepr, PrettyReprRepr, ParameterSpecs
from ..visualization import jinja_env, ElementHTML


class DiffractiveLayer(Element):
    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        mask: OptimizableTensor,
        mask_norm: float = 2 * torch.pi,
    ):
        r"""
        Diffractive layer defined by a phase mask.
        The field after propagating through the layer is calculated as:

        $$
        E^\text{out}_{xyw...} = E^\text{in}_{xyw...} \cdot \exp\left(2\pi i \frac{\text{mask}_{xy}}{\text{mask\_norm}}\right)
        $$

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Simulation parameters.
        mask : OptimizableTensor
            Two-dimensional tensor representing the aperture mask of shape `(H, W)`.
        mask_norm : float, optional
            Mask normalization factor.
        """

        super().__init__(simulation_parameters)

        self.mask = self.process_parameter("mask", mask)
        self.mask_norm = self.process_parameter("mask_norm", mask_norm)

        # Precompute cast operations so that forward() uses only
        # tensor ops (no Python dict/cache lookups → torch.compile-safe)
        self._cast_idx, self._cast_swaps = \
            self.simulation_parameters.precompute_cast("y", "x")

    @property
    def transmission_function(self) -> torch.Tensor:
        r"""
        The tensor representing the transmission function of the element
        $\exp\left(2\pi i \dfrac{\text{mask}}{\text{mask\_norm}}\right)$.
        The shape of the tensor is broadcastable to the incident wavefront's shape.
        """
        t = torch.exp((2j * torch.pi / self.mask_norm) * self.mask)
        t = t[self._cast_idx]
        for i, j in self._cast_swaps:
            t = t.swapaxes(i, j)
        return t

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        return incident_wavefront * self.transmission_function

    def reverse(self, transmission_wavefront: Wavefront) -> Wavefront:
        return transmission_wavefront * torch.conj(self.transmission_function)

    def to_specs(self) -> Iterable[ParameterSpecs]:
        mask = self.mask.numpy(force=True)
        mask_min = mask.min()
        mask_max = mask.max()

        norm = mask_max - mask_min
        img = 255 * (mask - mask_min) / norm if norm > 0 else 0 * mask

        return [
            ParameterSpecs(
                "mask",
                [
                    PrettyReprRepr(self.mask),
                    ImageRepr(img.astype("uint8")),
                ],
            ),
            ParameterSpecs("mask_norm", [PrettyReprRepr(self.mask_norm)]),
        ]

    @staticmethod
    def _widget_html_(
        index: int, name: str, element_type: str | None, subelements: list[ElementHTML]
    ) -> str:
        return jinja_env.get_template("widget_diffractive_layer.html.jinja").render(
            index=index, name=name, subelements=subelements
        )
