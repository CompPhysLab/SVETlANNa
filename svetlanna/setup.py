from typing import Iterable
from .elements import Element
from .specs import ParameterSpecs, SubelementSpecs
from torch import nn
from torch import Tensor
from warnings import warn
from .visualization import jinja_env, ElementHTML


class LinearOpticalSetup(nn.Module):
    """
    A linear optical network composed of Element's
    """

    def __init__(self, elements: Iterable[Element]) -> None:
        """
        Parameters
        ----------
        elements : Iterable[Element]
            A set of optical elements which make up a setup.
        """
        super().__init__()

        elements = list(elements)
        self.elements = elements
        self.net = nn.Sequential(*elements)  # torch network

        if len(elements) > 0:
            first_sim_params = elements[0].simulation_parameters

            def check_sim_params(element: Element) -> bool:
                return element.simulation_parameters is first_sim_params

            if not all(map(check_sim_params, self.elements)):
                warn(
                    "Some elements have different SimulationParameters "
                    "instance. It is more convenient to use "
                    "the same SimulationParameters instance."
                )

        if all((hasattr(el, "reverse") for el in self.elements)):

            class ReverseNet(nn.Module):
                def forward(self, Ein: Tensor) -> Tensor:
                    for el in reversed(elements):
                        Ein = el.reverse(Ein)
                    return Ein

            self._reverse_net = ReverseNet()
        else:
            self._reverse_net = None

    def forward(self, input_wavefront: Tensor) -> Tensor:
        """
        A forward function for a network assembled from elements.

        Parameters
        ----------
        input_wavefront : torch.Tensor
            A wavefront that enters the optical network.

        Returns
        -------
        torch.Tensor
            A wavefront after the last element of
            the network (output of the network).
        """
        return self.net(input_wavefront)

    def stepwise_forward(self, input_wavefront: Tensor):
        """
        Function that consistently applies forward method of each element
        to an input wavefront.

        Parameters
        ----------
        input_wavefront : torch.Tensor
            A wavefront that enters the optical network.

        Returns
        -------
        str
            A string that represents a scheme of a propagation through a setup.
        list(torch.Tensor)
            A list of an input wavefront evolution
            during a propagation through a setup.
        """
        this_wavefront = input_wavefront
        # list of wavefronts while propagation of an initial wavefront through the system
        steps_wavefront = [this_wavefront]  # input wavefront is a zeroth step

        optical_scheme = ""  # string that represents a linear optical setup (schematic)

        self.net.eval()
        for ind_element, element in enumerate(self.net):
            # for visualization in a console
            element_name = type(element).__name__
            optical_scheme += f"-({ind_element})-> [{ind_element + 1}. {element_name}] "
            # TODO: Replace len(...) with something for Iterable?
            if ind_element == len(self.net) - 1:
                optical_scheme += f"-({ind_element + 1})->"
            # element forward
            this_wavefront = element.forward(this_wavefront)
            steps_wavefront.append(this_wavefront)  # add a wavefront to list of steps

        return optical_scheme, steps_wavefront

    def reverse(self, Ein: Tensor) -> Tensor:
        """
        Reverses the input tensor using a pre-defined reverse network.

            Args:
                Ein: The input tensor to be reversed.

            Returns:
                Tensor: The reversed tensor if a reverse network is available.

            Raises:
                TypeError: If no reverse network is defined, indicating that
                    reverse propagation is not possible for this element.
        """
        if self._reverse_net is not None:
            return self._reverse_net(Ein)
        raise TypeError(
            "Reverse propagation is impossible. "
            "All elements should have reverse method."
        )

    def to_specs(self) -> Iterable[ParameterSpecs | SubelementSpecs]:
        """
        Converts the elements into a sequence of subelement specifications.

            Args:
                None

            Returns:
                Iterable[ParameterSpecs | SubelementSpecs]: An iterable yielding
                SubelementSpecs for each element in the 'elements' list, with an index.
        """
        return (
            SubelementSpecs(str(i), element) for i, element in enumerate(self.elements)
        )

    @staticmethod
    def _widget_html_(
        index: int, name: str, element_type: str | None, subelements: list[ElementHTML]
    ) -> str:
        """
        Renders the HTML for a widget using a Jinja template.

            Args:
                index: The index of the widget.
                name: The name of the widget.
                element_type: The type of element (optional).
                subelements: A list of sub-elements to include in the widget.

            Returns:
                str: The rendered HTML string for the widget.
        """
        return jinja_env.get_template("widget_linear_setup.html.jinja").render(
            index=index, name=name, subelements=subelements
        )
