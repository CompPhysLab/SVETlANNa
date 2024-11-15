import pytest
import torch

from svetlanna import elements
from svetlanna import SimulationParameters

lens_parameters = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "wavelength_test",
    "focal_length_test",
    "radius_test",
    "expected_std"
]


@pytest.mark.parametrize(
    lens_parameters,
    [(8, 12, 1200, 1400, 1064 * 1e-6, 100, 10, 1e-5),
     (8, 4, 1100, 1000, 1064 * 1e-6, 200, 15, 1e-5)]
)
def test_lens(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    wavelength_test: float,
    focal_length_test: float,
    radius_test: float,
    expected_std: float,
):
    """Test for the transmission function for the thin collecting lens

    Parameters
    ----------
    ox_size : float
        System size along the axis ox
    oy_size : float
        System size along the axis oy
    ox_nodes : int
        Number of computational nodes along the axis ox
    oy_nodes : int
        Number of computational nodes along the axis oy
    wavelength_test : float
        Wavelength for the incident field
    focal_length_test : float
        Focal length for the thin lens
    radius_test : float
        Radius of the thin lens
    expected_std : float
        Criterion for accepting the test(standard deviation)
    """

    params = SimulationParameters(
        {
            'W': torch.linspace(-ox_size/2, ox_size/2, ox_nodes),
            'H': torch.linspace(-oy_size/2, oy_size/2, oy_nodes),
            'wavelength': wavelength_test
        }
    )

    # transmission function of the thin lens as a class method
    transmission_function = elements.ThinLens(
        simulation_parameters=params,
        focal_length=focal_length_test,
        radius=radius_test
    ).get_transmission_function()

    x_linear = torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes)
    y_linear = torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes)
    x_grid, y_grid = torch.meshgrid(x_linear, y_linear, indexing='xy')

    wave_number = 2 * torch.pi / wavelength_test
    radius_squared = torch.pow(x_grid, 2) + torch.pow(y_grid, 2)

    transmission_function_analytic = torch.exp(
        1j * (-wave_number / (2 * focal_length_test) * radius_squared * (
            radius_squared <= radius_test**2
        ))
    )

    standard_deviation = torch.std(
        torch.real((1 / 1j) * (
            torch.log(transmission_function) - torch.log(
                transmission_function_analytic
                )
            )
        )
    )

    assert standard_deviation <= expected_std
