import pytest
import torch

from typing import Callable, Dict

from svetlanna import elements
from svetlanna import SimulationParameters
from svetlanna import Wavefront as w

nonlinear_element_parameters = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "wavelength_test",
    "response_function",
    "response_parameters",
]


def func(x, a, b):
    """
    Computes a value based on the given inputs.

        Args:
            x: The input value.
            a: A constant value.
            b: Another constant value.

        Returns:
            torch.Tensor: The computed result of the formula a / 1 + torch.exp(-b*x).
    """
    return a / 1 + torch.exp(-b * x)


@pytest.mark.parametrize(
    nonlinear_element_parameters,
    [
        (10, 10, 1000, 1200, 1064 * 1e-6, lambda x: x**2, None),
        (4, 4, 1300, 1000, 1064 * 1e-6, lambda x: torch.sin(x) + x**3, None),
        (
            15,
            8,
            1319,
            917,
            1e-6 * torch.tensor([330, 660, 1064]),
            lambda x: torch.sin(x) + x**3,
            None,
        ),  # noqa: E501
        (
            16,
            7,
            500,
            868,
            1e-6 * torch.tensor([330, 660, 1064]),
            func,
            {"a": 1.0, "b": 9.0},
        ),  # noqa: E501
    ],
)
def test_nonlinear_element(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    wavelength_test: float,
    response_function: Callable[[torch.Tensor], torch.Tensor],
    response_parameters: Dict,
):
    """
    Tests the NonlinearElement class with various parameters.

        Args:
            ox_size: The size of the simulation area in the x-direction.
            oy_size: The size of the simulation area in the y-direction.
            ox_nodes: The number of nodes in the x-direction.
            oy_nodes: The number of nodes in the y-direction.
            wavelength_test: The wavelength of the incident light.
            response_function: The nonlinear response function to use.
            response_parameters: A dictionary of parameters for the response function.

        Returns:
            None.  This method asserts that the output field from the NonlinearElement
            matches the analytically calculated output field.
    """
    params = SimulationParameters(
        {
            "W": torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes),
            "H": torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes),
            "wavelength": wavelength_test,
        }
    )

    incident_field = w(torch.rand(oy_nodes, ox_nodes))

    nle = elements.NonlinearElement(
        simulation_parameters=params,
        response_function=response_function,
        response_parameters=response_parameters,
    )

    incident_amplitude = torch.abs(incident_field)
    incident_phase = incident_field.phase

    if response_parameters is not None:
        keys = list(response_parameters.keys())

        output_amplitude = response_function(
            incident_amplitude,
            response_parameters[keys[0]],
            response_parameters[keys[1]],
        )

    else:
        output_amplitude = response_function(incident_amplitude)

    output_field_analytic = output_amplitude * torch.exp(1j * incident_phase)

    output_field = nle(incident_field)

    assert isinstance(output_field, w)
    assert torch.equal(output_field, output_field_analytic)
