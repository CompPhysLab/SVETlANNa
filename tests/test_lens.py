import pytest
import torch
import svetlanna
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
    [
        (
            8,      # ox_size, mm
            12,     # oy_size, mm
            1200,   # ox_nodes
            1400,   # oy_nodes
            torch.linspace(330 * 1e-6, 1064 * 1e-6, 20),    # wavelength_test, tensor   # noqa: E501
            100,    # focal_length_test, mm
            10,     # radius_test, mm
            1e-5    # expected_std
        ),
        (
            8,  # ox_size, mm
            4,  # oy_size, mm
            1100,   # ox_nodes
            1000,   # oy_nodes
            torch.linspace(660 * 1e-6, 1600 * 1e-6, 20),    # wavelength_test, tensor   # noqa: E501
            200,    # focal_length_test, mm
            15,     # radius_test, mm
            1e-5    # expected_std
        )
    ]
)
def test_lens(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    wavelength_test: torch.Tensor,
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

    # creating meshgrid
    x_grid = x_linear[None, :]
    y_grid = y_linear[:, None]

    x_grid = x_grid[None, ...]
    y_grid = y_grid[None, ...]

    wave_number = 2 * torch.pi / wavelength_test[..., None, None]
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


def test_reverse():
    params = SimulationParameters(
        {
            'W': torch.linspace(-10/2, 10/2, 10),
            'H': torch.linspace(-10/2, 10/2, 10),
            'wavelength': 1
        }
    )

    lens = elements.ThinLens(
        simulation_parameters=params,
        focal_length=1
    )

    # test is reverse(forward(x)) is x, where x is a wavefront
    wavefront = svetlanna.Wavefront.plane_wave(params)
    assert torch.allclose(
        lens.reverse(lens.forward(wavefront)), wavefront
    )
