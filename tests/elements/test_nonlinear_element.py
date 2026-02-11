import pytest
import torch

from typing import Callable, Dict, Concatenate

from svetlanna import elements
from svetlanna import SimulationParameters
from svetlanna import Wavefront
from svetlanna import PartialWithParameters


def func(x, a, b):
    return a / 1 + torch.exp(-b * x)


@pytest.mark.parametrize(
    ["response_function", "response_parameters"],
    [
        (lambda x: x**2, None),
        (lambda x: torch.sin(x) + x**3, None),
        (lambda x: torch.sin(x) + x**3, None),
        (func, {"a": 1.0, "b": 9.0}),
    ],
)
def test_nonlinear_element(
    sim_params_simple: SimulationParameters,
    response_function: Callable[Concatenate[Wavefront, ...], Wavefront],
    response_parameters: Dict,
):
    incident_field = Wavefront(torch.rand(sim_params_simple.axes_size(("y", "x"))))

    nle = elements.NonlinearElement(
        simulation_parameters=sim_params_simple,
        response_function=PartialWithParameters(
            response_function, **(response_parameters or {})
        ),
    )

    if response_parameters is not None:
        output_field_expected = response_function(incident_field, **response_parameters)
    else:
        output_field_expected = response_function(incident_field)

    output_field = nle(incident_field)

    assert isinstance(output_field, Wavefront)
    assert torch.equal(output_field, output_field_expected)


def test_nonlinear_element_device(device_simple: str):
    """Test nonlinear element on different devices."""

    sim_params = SimulationParameters(
        x=torch.linspace(-10, 10, 20), y=torch.linspace(-10, 10, 20), wavelength=1.0
    )
    wavefront = Wavefront.plane_wave(sim_params).to(device=device_simple)

    assert sim_params.device == torch.get_default_device()
    nle = elements.NonlinearElement(
        simulation_parameters=sim_params, response_function=lambda x: x
    )
    nle.to(device=device_simple)

    assert nle(wavefront).device.type == device_simple

    # Simulation parameters on device
    sim_params.to(device=device_simple)

    assert sim_params.device.type == device_simple
    nle = elements.NonlinearElement(
        simulation_parameters=sim_params, response_function=lambda x: x
    )

    assert nle(wavefront).device.type == device_simple
