import torch
from .elements import Element
from .specs import ParameterSpecs, SubelementSpecs
from torch import nn
from torch import Tensor
from warnings import warn
from .visualization import jinja_env, ElementHTML
from .wavefront import Wavefront
from typing import Iterable, TYPE_CHECKING


class LinearOpticalSetup(nn.Module):
    """Linear optical network composed of [`Element`][svetlanna.elements.Element] instances.
    It works the same way as a `torch.nn.Sequential` module, but with some additional features.
    """

    def __init__(self, elements: Iterable[Element]) -> None:
        """
        Parameters
        ----------
        elements : Iterable[Element]
            Optical elements that make up the setup. Elements are evaluated in
            the provided order.

        Examples
        --------
        ```python
        import svetlanna as sv

        setup = sv.LinearOpticalSetup(
            elements=[
                element1,
                element2,
                element3,
            ]
        )

        output_wavefront = setup(input_wavefront)
        ```
        """
        super().__init__()

        self.elements = list(elements)
        self.net = nn.Sequential(*self.elements)  # torch network

        if len(self.elements) > 0:
            first_sim_params = self.elements[0].simulation_parameters

            def check_sim_params_diveces(element: Element) -> bool:
                return first_sim_params.device == element.simulation_parameters.device

            def check_sim_params_equal(element: Element) -> bool:
                return first_sim_params.equal(element.simulation_parameters)

            if not all(map(check_sim_params_diveces, self.elements)):
                warn("Some elements have SimulationParameters on different devices.")
            else:
                # run the equality check only if all elements are on the same device, otherwise it will raise RuntimeError
                if not all(map(check_sim_params_equal, self.elements)):
                    warn("Some elements have different SimulationParameters.")

        if all((hasattr(el, "reverse") for el in self.elements)):

            class ReverseNet(nn.Module):
                def forward(_self, Ein: Tensor) -> Tensor:

                    for el in reversed(self.elements):
                        Ein = el.reverse(Ein)  # type: ignore
                    return Ein

            self._reverse_net: ReverseNet | None = ReverseNet()
        else:
            self._reverse_net = None

    def forward(self, input_wavefront: Wavefront) -> Wavefront:
        return self.net(input_wavefront)

    def stepwise_forward(self, input_wavefront: Wavefront) -> tuple[Wavefront, ...]:
        """Apply elements step-by-step and collect intermediate wavefronts.

        Parameters
        ----------
        input_wavefront : Wavefront
            A wavefront that enters the optical network.

        Returns
        -------
        tuple[Wavefront, ...]
            A tuple of wavefronts showing the propagation through the setup.
            The first wavefront is the input wavefront, and the last one is the output wavefront after propagation through all elements.
        """
        this_wavefront = input_wavefront
        # list of wavefronts while propagation of an initial wavefront through the system
        steps_wavefront = [this_wavefront]  # input wavefront is a zeroth step

        with torch.no_grad():

            for element in self.net:
                this_wavefront = element.forward(this_wavefront)
                steps_wavefront.append(
                    this_wavefront
                )  # add a wavefront to list of steps

        return tuple(steps_wavefront)

    def reverse(self, Ein: Tensor) -> Tensor:
        """Reverse propagation through the setup.
        All elements in the setup must have a `reverse` method. If any element
        lacks this method, a `TypeError` is raised.

        Parameters
        ----------
        Ein : Tensor
            Input wavefront to reverse propagate.

        Returns
        -------
        Tensor
            Output wavefront after reverse propagation.

        Raises
        ------
        TypeError
            If reverse propagation is not supported by all elements in the setup.
        """
        if self._reverse_net is not None:
            return self._reverse_net(Ein)
        raise TypeError(
            "Reverse propagation is impossible. "
            "All elements should have reverse method."
        )

    def to_specs(self) -> Iterable[ParameterSpecs | SubelementSpecs]:
        return (
            SubelementSpecs(str(i), element) for i, element in enumerate(self.elements)
        )

    @staticmethod
    def _widget_html_(
        index: int, name: str, element_type: str | None, subelements: list[ElementHTML]
    ) -> str:
        return jinja_env.get_template("widget_linear_setup.html.jinja").render(
            index=index, name=name, subelements=subelements
        )

    if TYPE_CHECKING:

        def __call__(self, input_wavefront: Wavefront) -> Wavefront: ...
