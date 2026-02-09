import torch
import svetlanna
from svetlanna import elements
from svetlanna import SimulationParameters


def test_diffractive_layer(
    sim_params_simple: SimulationParameters,
):
    """Test diffractive layer transmission."""
    mask = torch.rand(sim_params_simple.axes_size(("y", "x")))

    diffractive_layer = elements.DiffractiveLayer(
        simulation_parameters=sim_params_simple,
        mask=mask,
    )

    transmission_function_expected = sim_params_simple.cast(
        torch.exp(1j * mask), "y", "x"
    )

    torch.testing.assert_close(
        diffractive_layer.transmission_function, transmission_function_expected
    )

    # Validate forward calculation output.
    wavefront = svetlanna.Wavefront.plane_wave(sim_params_simple)
    torch.testing.assert_close(
        diffractive_layer(wavefront), transmission_function_expected * wavefront
    )


def test_diffractive_layer_device(device_simple: str):
    """Test diffractive layer on different devices."""

    sim_params = SimulationParameters(
        x=torch.linspace(-10, 10, 20), y=torch.linspace(-10, 10, 20), wavelength=1.0
    )
    wavefront = svetlanna.Wavefront.plane_wave(sim_params).to(device=device_simple)

    sim_params.to(torch.get_default_device())  # TODO: remove
    assert sim_params.device == torch.get_default_device()
    diffractive_layer = elements.DiffractiveLayer(
        simulation_parameters=sim_params,
        mask=torch.rand(sim_params.axes_size(("y", "x"))),
    )
    diffractive_layer.to(device=device_simple)

    assert diffractive_layer(wavefront).device.type == device_simple


def test_reverse():
    params = SimulationParameters(
        x=torch.linspace(-10 / 2, 10 / 2, 10),
        y=torch.linspace(-10 / 2, 10 / 2, 10),
        wavelength=1,
    )

    diffractive_layer = elements.DiffractiveLayer(
        simulation_parameters=params,
        mask=torch.rand(params.axes_size(("y", "x"))),
    )

    # Validate that reverse(forward(x)) returns the original wavefront.
    wavefront = svetlanna.Wavefront.plane_wave(params)
    assert torch.allclose(
        diffractive_layer.reverse(diffractive_layer.forward(wavefront)), wavefront
    )
