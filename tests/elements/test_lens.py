import pytest
import torch
import svetlanna
from svetlanna import elements
from svetlanna import SimulationParameters


@pytest.mark.parametrize(
    ["focal_length", "radius"],
    [
        (1, 1),
        (2, 0.5),
        (1e-1, 3),
    ],
)
def test_lens(
    sim_params_simple: SimulationParameters,
    focal_length: float,
    radius: float,
):
    """Test thin lens transmission."""
    lens = elements.ThinLens(
        simulation_parameters=sim_params_simple,
        focal_length=focal_length,
        radius=radius,
    )

    x_linear = sim_params_simple.x
    y_linear = sim_params_simple.y

    # Create meshgrid.
    x_grid = x_linear[None, :]
    y_grid = y_linear[:, None]

    wave_number = 2 * torch.pi / sim_params_simple.wavelength[..., None, None]
    radius_squared = torch.pow(x_grid, 2) + torch.pow(y_grid, 2)

    transmission_function_expected = sim_params_simple.cast(
        torch.exp(
            1j
            * (
                -wave_number
                / (2 * focal_length)
                * radius_squared
                * (radius_squared <= radius**2)
            )
        ),
        "wavelength",
        "y",
        "x",
    )

    torch.testing.assert_close(
        lens.transmission_function, transmission_function_expected
    )

    # Validate forward calculation output.
    wavefront = svetlanna.Wavefront.plane_wave(sim_params_simple)
    torch.testing.assert_close(
        lens(wavefront), transmission_function_expected * wavefront
    )


def test_lens_device(device_simple: str):
    """Test thin lens on different devices."""

    sim_params = SimulationParameters(
        x=torch.linspace(-10, 10, 20), y=torch.linspace(-10, 10, 20), wavelength=1.0
    )
    wavefront = svetlanna.Wavefront.plane_wave(sim_params).to(device=device_simple)

    sim_params.to(torch.get_default_device())  # TODO: remove
    assert sim_params.device == torch.get_default_device()
    lens = elements.ThinLens(
        simulation_parameters=sim_params, focal_length=1.0, radius=5.0
    )
    lens.to(device=device_simple)

    assert lens(wavefront).device.type == device_simple


def test_reverse():
    params = SimulationParameters(
        x=torch.linspace(-10 / 2, 10 / 2, 10),
        y=torch.linspace(-10 / 2, 10 / 2, 10),
        wavelength=1,
    )

    lens = elements.ThinLens(simulation_parameters=params, focal_length=1)

    # Validate that reverse(forward(x)) returns the original wavefront.
    wavefront = svetlanna.Wavefront.plane_wave(params)
    assert torch.allclose(lens.reverse(lens.forward(wavefront)), wavefront)
