from typing import Any
import torch
from svetlanna.parameters import ConstrainedParameter, Parameter
from svetlanna import LinearOpticalSetup, Wavefront
from svetlanna import SimulationParameters
from svetlanna.elements import Element
from svetlanna.units import ureg
import pytest


class SimpleElement(Element):
    def __init__(self, a: Any, simulation_parameters: SimulationParameters) -> None:
        super().__init__(simulation_parameters)

        self.a = a

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        return incident_wavefront * self.a


class ReversableSimpleElement(SimpleElement):
    def reverse(self, wavefront):
        return wavefront * self.a


def test_init():
    sim_params = SimulationParameters(
        {
            "x": torch.linspace(-5 * ureg.mm, 5 * ureg.mm, 10),
            "y": torch.linspace(-5 * ureg.mm, 5 * ureg.mm, 10),
            "wavelength": 1,
        }
    )

    a = torch.tensor(2)
    el1 = SimpleElement(a=a, simulation_parameters=sim_params)
    el2 = SimpleElement(a=a, simulation_parameters=sim_params)
    el3 = SimpleElement(a=a, simulation_parameters=sim_params)

    setup = LinearOpticalSetup(elements=[el1, el2, el3])

    assert isinstance(setup.net, torch.nn.Module)

    x = torch.tensor(123)
    assert setup.net(x) == x * a**3
    assert setup.forward(x) == x * a**3


def test_init_warning():
    sim_params1 = SimulationParameters(
        {
            "x": torch.linspace(-5 * ureg.mm, 5 * ureg.mm, 10),
            "y": torch.linspace(-5 * ureg.mm, 5 * ureg.mm, 10),
            "wavelength": 1,
        }
    )
    sim_params2 = SimulationParameters(
        {
            "x": torch.linspace(-5 * ureg.mm, 5 * ureg.mm, 10),
            "y": torch.linspace(-5 * ureg.mm, 5 * ureg.mm, 20),
            "wavelength": 1,
        }
    )

    a = torch.tensor(2)
    el1 = SimpleElement(a=a, simulation_parameters=sim_params1)
    el2 = SimpleElement(a=a, simulation_parameters=sim_params2)

    with pytest.warns(UserWarning):
        LinearOpticalSetup(elements=[el1, el2])


def test_to_device(device_simple: str):
    sim_params = SimulationParameters(
        {
            "x": torch.linspace(-5 * ureg.mm, 5 * ureg.mm, 10),
            "y": torch.linspace(-5 * ureg.mm, 5 * ureg.mm, 10),
            "wavelength": 1,
        }
    )

    el1 = SimpleElement(a=Parameter(2.0), simulation_parameters=sim_params)
    el2 = SimpleElement(
        a=ConstrainedParameter(data=0.5, min_value=0, max_value=1),
        simulation_parameters=sim_params,
    )

    setup = LinearOpticalSetup([el1, el2])

    setup.net.to(device_simple)

    assert el1.a.device.type == device_simple
    assert el1.a.inner_parameter.device.type == device_simple

    assert el2.a.device.type == device_simple
    assert el2.a.inner_parameter.device.type == device_simple

    # test warning when elements have different devices
    el3 = SimpleElement(
        a=torch.tensor(123), simulation_parameters=sim_params.to(device=device_simple)
    )
    if el1.a.device.type != el3.a.device.type:
        with pytest.warns(UserWarning):
            LinearOpticalSetup(elements=[el1, el3])


def test_reverse():
    # test empty setup
    setup = LinearOpticalSetup(elements=[])

    wf = torch.Tensor([2.0, 3.0])
    assert setup.reverse(wf) is wf

    # test setup
    sim_params = SimulationParameters(
        {
            "x": torch.linspace(-5 * ureg.mm, 5 * ureg.mm, 10),
            "y": torch.linspace(-5 * ureg.mm, 5 * ureg.mm, 10),
            "wavelength": 1,
        }
    )
    a = torch.tensor(2)

    # test unreversable element
    el = SimpleElement(a=a, simulation_parameters=sim_params)
    setup = LinearOpticalSetup(elements=[el])
    with pytest.raises(TypeError):
        setup.reverse(wf)

    # test reversable element
    el = ReversableSimpleElement(a=a, simulation_parameters=sim_params)
    setup = LinearOpticalSetup(elements=[el])
    torch.testing.assert_close(setup.reverse(wf), wf * a)


def test_to_specs():
    """Stupid test to increase code coverage."""
    sim_params = SimulationParameters(
        {
            "x": torch.linspace(-5 * ureg.mm, 5 * ureg.mm, 10),
            "y": torch.linspace(-5 * ureg.mm, 5 * ureg.mm, 10),
            "wavelength": 1,
        }
    )

    a = torch.tensor(2)
    el1 = SimpleElement(a=a, simulation_parameters=sim_params)
    el2 = SimpleElement(a=a, simulation_parameters=sim_params)
    el3 = SimpleElement(a=a, simulation_parameters=sim_params)

    setup = LinearOpticalSetup(elements=[el1, el2, el3])
    assert setup.to_specs()
    assert isinstance(setup._widget_html_(0, "", None, []), str)
