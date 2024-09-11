from svetlanna import elements
from svetlanna import SimulationParameters

import pytest
import torch


rectangle_parameters = "ox_size, oy_size, ox_nodes, oy_nodes," + "wavelength_test, height_test, width_test, expected_std"


@pytest.mark.parametrize(
    rectangle_parameters,
    [(10, 10, 1000, 1200, 1064 * 1e-6, 4, 10, 1e-5),
     (4, 4, 1300, 1000, 1064 * 1e-6, 3, 1, 1e-5)]
)
def test_rectangle_aperture(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    wavelength_test: float,
    height_test: float,
    width_test: float,
    expected_std: float,
):
    params = SimulationParameters(
        x_size=ox_size,
        y_size=oy_size,
        x_nodes=ox_nodes,
        y_nodes=oy_nodes,
        wavelength=wavelength_test
    )

    transmission_function = elements.RectangularAperture(
        simulation_parameters=params,
        height=height_test,
        width=width_test
    ).get_transmission_function()

    x_linear = torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes)
    y_linear = torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes)
    x_grid, y_grid = torch.meshgrid(x_linear, y_linear, indexing='xy')

    transmission_function_analytic = 1 * (
        torch.abs(x_grid) <= width_test / 2
    ) * (torch.abs(y_grid) <= height_test / 2)

    standard_deviation = torch.std(
        transmission_function - transmission_function_analytic
    )

    assert standard_deviation <= expected_std


round_parameters = "ox_size, oy_size, ox_nodes, oy_nodes," + "wavelength_test, radius_test, expected_std"


@pytest.mark.parametrize(
    round_parameters,
    [(10, 15, 1200, 1000, 1064 * 1e-6, 4, 1e-5),
     (8, 4, 1000, 1300, 1064 * 1e-6, 2.5, 1e-5)]
)
def test_round_aperture(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    wavelength_test: float,
    radius_test: float,
    expected_std: float,
):
    params = SimulationParameters(
        x_size=ox_size,
        y_size=oy_size,
        x_nodes=ox_nodes,
        y_nodes=oy_nodes,
        wavelength=wavelength_test
    )

    transmission_function = elements.RoundAperture(
        simulation_parameters=params,
        radius=radius_test
    ).get_transmission_function()

    x_linear = torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes)
    y_linear = torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes)
    x_grid, y_grid = torch.meshgrid(x_linear, y_linear, indexing='xy')

    transmission_function_analytic = 1 * (
        torch.pow(x_grid, 2) + torch.pow(y_grid, 2) <= radius_test**2
    )

    standard_deviation = torch.std(
        transmission_function - transmission_function_analytic
    )

    assert standard_deviation <= expected_std


arbitrary_parameters = "ox_size, oy_size, ox_nodes, oy_nodes," + "wavelength_test, mask_test, expected_std"


@pytest.mark.parametrize(
    arbitrary_parameters,
    [(10, 15, 1200, 1000, 1064 * 1e-6, torch.rand(1000, 1200), 1e-5),
     (8, 4, 1100, 1000, 1064 * 1e-6, torch.rand(1100, 1000), 1e-5)]
)
def test_aperture(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    wavelength_test: float,
    mask_test: float,
    expected_std: float,
):
    params = SimulationParameters(
        x_size=ox_size,
        y_size=oy_size,
        x_nodes=ox_nodes,
        y_nodes=oy_nodes,
        wavelength=wavelength_test
    )

    transmission_function = elements.Aperture(
        simulation_parameters=params,
        mask=mask_test
    ).get_transmission_function()

    x_linear = torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes)
    y_linear = torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes)
    x_grid, y_grid = torch.meshgrid(x_linear, y_linear, indexing='xy')

    transmission_function_analytic = mask_test

    standard_deviation = torch.std(
        transmission_function - transmission_function_analytic
    )

    assert standard_deviation <= expected_std
