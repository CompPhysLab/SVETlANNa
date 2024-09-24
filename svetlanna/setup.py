from typing import Iterable
from .elements import Element
from torch import nn
from torch import Tensor


class LinearOpticalSetup:
    def __init__(self, elements: Iterable[Element]) -> None:
        self.elements = elements
        self.net = nn.Sequential(*elements)

        if all((hasattr(el, 'reverse') for el in self.elements)):

            class ReverseNet(nn.Module):
                def forward(self, Ein: Tensor) -> Tensor:
                    for el in reversed(list(elements)):
                        Ein = el.reverse(Ein)
                    return Ein

            self._reverse_net = ReverseNet()
        else:
            self._reverse_net = None

    def forward(self, Ein: Tensor) -> Tensor:
        return self.net(Ein)

    def reverse(self, Ein: Tensor) -> Tensor:
        if self._reverse_net is not None:
            return self._reverse_net(Ein)
        raise TypeError('Reverse propagation is impossible. All elements should have reverse method.')
