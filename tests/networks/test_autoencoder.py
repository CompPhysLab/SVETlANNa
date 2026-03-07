import torch
from svetlanna.wavefront import Wavefront, SimulationParameters
from svetlanna.networks import LinearAutoencoder
from svetlanna.elements import DiffractiveLayer


def test_autoencoder_forward(sim_params_simple: SimulationParameters):
    """Test forward function for a single Wavefront sequence."""
    random_tensor = sim_params_simple.cast(
        torch.rand(sim_params_simple.axis_sizes(("y", "x"))), "y", "x"
    )
    test_wavefront = Wavefront(random_tensor)

    encoder = DiffractiveLayer(
        sim_params_simple, mask=torch.rand(sim_params_simple.axis_sizes(("y", "x")))
    )
    decoder = DiffractiveLayer(
        sim_params_simple, mask=torch.rand(sim_params_simple.axis_sizes(("y", "x")))
    )
    autoencoder = LinearAutoencoder(
        encoder_element=encoder,
        decoder_element=decoder,
    )

    wf_encoded = autoencoder.encode(test_wavefront)
    wf_decoded = autoencoder.decode(wf_encoded)

    torch.testing.assert_close(wf_decoded, autoencoder(test_wavefront))


def test_device(device_simple: str):
    """Test reservoir on different devices."""

    sim_params = SimulationParameters(
        x=torch.tensor([0]), y=torch.tensor([0]), wavelength=1.0
    )
    wavefront = Wavefront.plane_wave(sim_params).to(device=device_simple)

    assert sim_params.device == torch.get_default_device()
    autoencoder = LinearAutoencoder(
        encoder_element=DiffractiveLayer(
            sim_params,
            mask=torch.rand(
                sim_params.axis_sizes(("y", "x")),
            ),
        ),
        decoder_element=DiffractiveLayer(
            sim_params,
            mask=torch.rand(
                sim_params.axis_sizes(("y", "x")),
            ),
        ),
    )
    autoencoder.to(device=device_simple)

    assert autoencoder(wavefront).device.type == device_simple

    # Simulation parameters on device
    sim_params.to(device=device_simple)

    assert sim_params.device.type == device_simple
    autoencoder = LinearAutoencoder(
        encoder_element=DiffractiveLayer(
            sim_params,
            mask=torch.rand(
                sim_params.axis_sizes(("y", "x")), device=sim_params.device
            ),
        ),
        decoder_element=DiffractiveLayer(
            sim_params,
            mask=torch.rand(
                sim_params.axis_sizes(("y", "x")), device=sim_params.device
            ),
        ),
    )

    assert autoencoder(wavefront).device.type == device_simple


def test_to_specs():
    """Stupid test to increase code coverage."""
    sim_params = SimulationParameters(
        x=torch.linspace(-10, 10, 20), y=torch.linspace(-10, 10, 20), wavelength=1.0
    )

    autoencoder = LinearAutoencoder(
        encoder_element=DiffractiveLayer(
            sim_params,
            mask=torch.rand(
                sim_params.axis_sizes(("y", "x")), device=sim_params.device
            ),
        ),
        decoder_element=DiffractiveLayer(
            sim_params,
            mask=torch.rand(
                sim_params.axis_sizes(("y", "x")), device=sim_params.device
            ),
        ),
    )

    assert autoencoder.to_specs()
