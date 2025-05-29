import torch
from .element import Element
from ..simulation_parameters import SimulationParameters
from ..wavefront import Wavefront, mul
from ..parameters import OptimizableTensor
from typing import Iterable
from ..specs import ImageRepr, PrettyReprRepr, ParameterSpecs
from ..visualization import jinja_env, ElementHTML


class DiffractiveLayer(Element):
    """A class that described the field after propagating through the
    passive diffractive layer with a given phase mask
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        mask: OptimizableTensor,
        mask_norm: float = 2 * torch.pi,
    ):
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Simulation parameters
        mask : OptimizableTensor
            Phase mask
        mask_norm : float, optional
            This value will be used as following:
            the phase addition is equal to `2*torch.pi * mask / mask_norm`.
            By default, `2*torch.pi`
        """

        super().__init__(simulation_parameters)

        self.mask = self.process_parameter("mask", mask)
        self.mask_norm = self.process_parameter("mask_norm", mask_norm)

    @property
    def transmission_function(self) -> torch.Tensor:
        """
        Calculates the transmission function based on the mask and its norm.

                Args:
                    None

                Returns:
                    torch.Tensor: The calculated transmission function as a complex-valued tensor.
        """
        return torch.exp((2j * torch.pi / self.mask_norm) * self.mask)

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        """Method that calculates the field after propagating through the SLM

        Parameters
        ----------
        input_field : Wavefront
            Field incident on the SLM

        Returns
        -------
        Wavefront
            The field after propagating through the SLM
        """
        return mul(
            incident_wavefront,
            self.transmission_function,
            ("H", "W"),
            self.simulation_parameters,
        )

    def reverse(self, transmission_wavefront: Wavefront) -> Wavefront:
        """Method that calculates the field after passing the SLM in back
        propagation

        Parameters
        ----------
        transmitted_field : Wavefront
            Field incident on the SLM in back propagation
            (transmitted field in forward propagation)

        Returns
        -------
        Wavefront
            Field transmitted on the SLM in back propagation
            (incident field in forward propagation)
        """
        return mul(
            transmission_wavefront,
            torch.conj(self.transmission_function),
            ("H", "W"),
            self.simulation_parameters,
        )

    def to_specs(self) -> Iterable[ParameterSpecs]:
        """
        Returns parameter specifications for the mask and normalized mask.

            Args:
                None

            Returns:
                Iterable[ParameterSpecs]: A list of ParameterSpecs objects, one for the
                mask (with both a pretty representation and an image representation) and
                one for the normalized mask (with only a pretty representation).
        """
        mask = self.mask.numpy(force=True)
        mask_min = mask.min()
        mask_max = mask.max()

        return [
            ParameterSpecs(
                "mask",
                [
                    PrettyReprRepr(self.mask),
                    ImageRepr(
                        (255 * (mask - mask_min) / (mask_max - mask_min)).astype(
                            "uint8"
                        )
                    ),
                ],
            ),
            ParameterSpecs("mask_norm", [PrettyReprRepr(self.mask_norm)]),
        ]

    @staticmethod
    def _widget_html_(
        index: int, name: str, element_type: str | None, subelements: list[ElementHTML]
    ) -> str:
        """
        Renders the HTML for a widget using a Jinja2 template.

          Args:
              index: The index of the widget.
              name: The name of the widget.
              element_type: The type of element (optional).
              subelements: A list of sub-elements to include in the widget.

          Returns:
              str: The rendered HTML string for the widget.
        """
        return jinja_env.get_template("widget_diffractive_layer.html.jinja").render(
            index=index, name=name, subelements=subelements
        )
