from typing import Iterable
from .elements import Element
from torch import nn
from torch import Tensor


class LinearOpticalSetup:
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
        self.elements = elements
        self.net = nn.Sequential(*elements)  # torch network

        if all((hasattr(el, 'reverse') for el in self.elements)):

            class ReverseNet(nn.Module):
                def forward(self, Ein: Tensor) -> Tensor:
                    for el in reversed(list(elements)):
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
            A wavefront after the last element of the network (output of the network).
        """
        return self.net(input_wavefront)

    def stepwise_forward(self, input_wavefront: Tensor):
        """
        Function that consistently applies forward method of each element to an input wavefront.

        Parameters
        ----------
        input_wavefront : torch.Tensor
            A wavefront that enters the optical network.

        Returns
        -------
        str
            A string that represents a scheme of a propagation through a setup.
        list(torch.Tensor)
            A list of an input wavefront evolution during a propagation through a setup.
        """
        this_wavefront = input_wavefront
        # list of wavefronts while propagation of an initial wavefront through the system
        steps_wavefront = [this_wavefront]  # input wavefront is a zeroth step

        optical_scheme = ''  # string that represents a linear optical setup (schematic)

        self.net.eval()
        for ind_element, element in enumerate(self.net):
            # for visualization in a console
            element_name = type(element).__name__
            optical_scheme += f'-({ind_element})-> [{ind_element + 1}. {element_name}] '
            # TODO: Replace len(...) with something for Iterable?
            if ind_element == len(self.net) - 1:
                optical_scheme += f'-({ind_element + 1})->'
            # element forward
            this_wavefront = element.forward(this_wavefront)
            steps_wavefront.append(this_wavefront)  # add a wavefront to list of steps

        return optical_scheme, steps_wavefront

    def reverse(self, Ein: Tensor) -> Tensor:
        if self._reverse_net is not None:
            return self._reverse_net(Ein)
        raise TypeError('Reverse propagation is impossible. All elements should have reverse method.')
