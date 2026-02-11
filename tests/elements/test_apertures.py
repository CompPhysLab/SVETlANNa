import pytest
import torch
import svetlanna
from svetlanna import elements
from svetlanna import SimulationParameters


def test_aperture(sim_params_simple: SimulationParameters):
    """Test aperture transmission with an arbitrary mask."""

    mask = torch.rand(sim_params_simple.axes_size(("y", "x")))
    aperture = elements.Aperture(simulation_parameters=sim_params_simple, mask=mask)

    transmission_function_expected = sim_params_simple.cast(mask, "y", "x")

    torch.testing.assert_close(
        aperture.transmission_function, transmission_function_expected
    )

    # Validate forward calculation output.
    wavefront = svetlanna.Wavefront.plane_wave(sim_params_simple)
    torch.testing.assert_close(
        aperture(wavefront), transmission_function_expected * wavefront
    )


@pytest.mark.parametrize("height", [0.5, 1])
@pytest.mark.parametrize("width", [0.5, 1])
def test_rectangle_aperture(
    sim_params_simple: SimulationParameters, height: float, width: float
):
    """Test rectangular aperture transmission."""

    aperture = elements.RectangularAperture(
        simulation_parameters=sim_params_simple, height=height, width=width
    )

    x_linear = sim_params_simple.x
    y_linear = sim_params_simple.y
    x_grid, y_grid = torch.meshgrid(x_linear, y_linear, indexing="xy")
    transmission_function_expected = sim_params_simple.cast(
        (1 * (torch.abs(x_grid) <= width / 2) * (torch.abs(y_grid) <= height / 2))
        + 0.0,
        "y",
        "x",
    )

    torch.testing.assert_close(
        aperture.transmission_function, transmission_function_expected
    )

    # Validate forward calculation output.
    wavefront = svetlanna.Wavefront.plane_wave(sim_params_simple)
    torch.testing.assert_close(
        aperture(wavefront), transmission_function_expected * wavefront
    )


@pytest.mark.parametrize("radius", [0.5, 1, 3])
def test_round_aperture(
    sim_params_simple: SimulationParameters,
    radius: float,
):
    """Test round aperture transmission."""
    aperture = elements.RoundAperture(
        simulation_parameters=sim_params_simple, radius=radius
    )

    x_linear = sim_params_simple.x
    y_linear = sim_params_simple.y
    x_grid, y_grid = torch.meshgrid(x_linear, y_linear, indexing="xy")

    transmission_function_expected = sim_params_simple.cast(
        (1 * (torch.pow(x_grid, 2) + torch.pow(y_grid, 2) <= radius**2) + 0.0), "y", "x"
    )

    torch.testing.assert_close(
        aperture.transmission_function, transmission_function_expected
    )

    # Validate forward calculation output.
    wavefront = svetlanna.Wavefront.plane_wave(sim_params_simple)
    torch.testing.assert_close(
        aperture(wavefront), transmission_function_expected * wavefront
    )


def test_aperture_device(device_simple: str):
    """Test apertures on different devices."""

    sim_params = SimulationParameters(
        x=torch.linspace(-10, 10, 20), y=torch.linspace(-10, 10, 20), wavelength=1.0
    )
    wavefront = svetlanna.Wavefront.plane_wave(sim_params).to(device=device_simple)

    # Aperture
    assert sim_params.device == torch.get_default_device()
    aperture = elements.Aperture(
        simulation_parameters=sim_params,
        mask=torch.zeros(sim_params.axes_size(("y", "x"))),
    )
    aperture.to(device=device_simple)

    assert aperture(wavefront).device.type == device_simple

    # Rectangular aperture
    assert sim_params.device == torch.get_default_device()
    rectangular_aperture = elements.RectangularAperture(
        simulation_parameters=sim_params, height=1, width=1
    )
    rectangular_aperture.to(device=device_simple)

    assert rectangular_aperture(wavefront).device.type == device_simple

    # Round aperture
    assert sim_params.device == torch.get_default_device()
    round_aperture = elements.RoundAperture(simulation_parameters=sim_params, radius=1)
    round_aperture.to(device=device_simple)

    assert round_aperture(wavefront).device.type == device_simple

    # Simulation parameters on device
    sim_params.to(device=device_simple)

    # Aperture
    assert sim_params.device.type == device_simple
    aperture = elements.Aperture(
        simulation_parameters=sim_params,
        mask=torch.zeros(sim_params.axes_size(("y", "x"))).to(device=device_simple),
    )

    assert aperture(wavefront).device.type == device_simple

    # Rectangular aperture
    assert sim_params.device.type == device_simple
    rectangular_aperture = elements.RectangularAperture(
        simulation_parameters=sim_params, height=1, width=1
    )

    assert rectangular_aperture(wavefront).device.type == device_simple

    # Round aperture
    assert sim_params.device.type == device_simple
    round_aperture = elements.RoundAperture(simulation_parameters=sim_params, radius=1)

    assert round_aperture(wavefront).device.type == device_simple


def test_to_specs():
    """Stupid test to increase code coverage."""
    sim_params = SimulationParameters(
        x=torch.linspace(-10, 10, 20), y=torch.linspace(-10, 10, 20), wavelength=1.0
    )

    aperture = elements.Aperture(
        simulation_parameters=sim_params,
        mask=torch.zeros(sim_params.axes_size(("y", "x"))),
    )

    assert aperture.to_specs()
    assert isinstance(aperture._widget_html_(0, "", None, []), str)

    rectangular_aperture = elements.RectangularAperture(
        simulation_parameters=sim_params, height=1, width=1
    )

    assert rectangular_aperture.to_specs()
    assert isinstance(rectangular_aperture._widget_html_(0, "", None, []), str)

    round_aperture = elements.RoundAperture(simulation_parameters=sim_params, radius=1)

    assert round_aperture.to_specs()
    assert isinstance(round_aperture._widget_html_(0, "", None, []), str)
