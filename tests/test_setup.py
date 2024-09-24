from torch import Tensor
from typing import Any
import torch
from svetlanna.parameters import BoundedParameter, Parameter
from svetlanna import LinearOpticalSetup
from svetlanna import SimulationParameters
from svetlanna.elements import Element
from svetlanna.units import ureg
import pytest


class SimpleElement(Element):
    def __init__(
        self,
        a: Any,
        simulation_parameters: SimulationParameters
    ) -> None:
        super().__init__(simulation_parameters)

        self.a = a

    def forward(self, Ein: Tensor) -> Tensor:
        return Ein * self.a


def test_init():
    sim_params = SimulationParameters(
        x_size=10 * ureg.mm,
        y_size=10 * ureg.mm,
        x_nodes=10,
        y_nodes=10,
        wavelength=1
    )

    a = torch.tensor(2)
    el1 = SimpleElement(a=a, simulation_parameters=sim_params)
    el2 = SimpleElement(a=a, simulation_parameters=sim_params)
    el3 = SimpleElement(a=a, simulation_parameters=sim_params)

    setup = LinearOpticalSetup(elements=[
        el1, el2, el3
    ])

    assert isinstance(setup.net, torch.nn.Module)

    x = torch.tensor(123)
    assert setup.net(x) == x * a**3
    assert setup.forward(x) == x * a**3


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="cuda is not available"
)
def test_to_cuda_device():
    device = 'cuda'

    sim_params = SimulationParameters(
        x_size=10 * ureg.mm,
        y_size=10 * ureg.mm,
        x_nodes=10,
        y_nodes=10,
        wavelength=1
    )

    el1 = SimpleElement(
        a=Parameter(2.),
        simulation_parameters=sim_params
    )
    el2 = SimpleElement(
        a=BoundedParameter(
            data=0.5,
            min_value=0,
            max_value=1
        ),
        simulation_parameters=sim_params
    )

    setup = LinearOpticalSetup([el1, el2])

    setup.net.to(device)

    assert el1.a.device.type == device
    assert el1.a.inner_parameter.device.type == device

    assert el2.a.device.type == device
    assert el2.a.inner_parameter.device.type == device


def test_reverse_empty():
    setup = LinearOpticalSetup(elements=[])

    a = torch.Tensor([2., 3.])
    assert setup.reverse(a) is a
