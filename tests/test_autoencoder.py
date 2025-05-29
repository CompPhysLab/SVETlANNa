import pytest
import torch

from svetlanna.wavefront import Wavefront
from svetlanna.networks.autoencoder import LinearAutoencoder

from tests.test_drnn import sim_params, zero_free_space


@pytest.fixture()
def empty_encoder_or_decoder(zero_free_space):
    """Returns list with a zero distance FreeSpace, i.e. empty encoder/decoder"""
    return [zero_free_space]


@pytest.mark.parametrize(
    "wf_real, wf_imag",
    [
        (1.00, 0.00),
        (0.00, 1.00),
        (2.50, 1.25),
    ],
)
def test_autoencoder_forward(
    sim_params, empty_encoder_or_decoder, wf_real, wf_imag  # fixtures
):
    """Test forward function for a single Wavefront sequence."""
    h, w = sim_params.axes_size(
        axs=("H", "W")
    )  # size of a wavefront according to SimulationParameters
    test_wavefront = Wavefront(
        torch.ones(size=(h, w), dtype=torch.float64) * wf_real
        + torch.ones(size=(h, w), dtype=torch.float64) * wf_imag * 1j
    )

    for to_return in ["wf", "amps"]:
        autoencoder = LinearAutoencoder(
            sim_params,
            encoder_elements_list=empty_encoder_or_decoder,
            decoder_elements_list=empty_encoder_or_decoder,
            to_return=to_return,
            device=torch.get_default_device(),
        )  # empty Autoencoder

        wf_encoded, wf_decoded = autoencoder(test_wavefront)  # forward for Autoencoder

        for wf in [wf_encoded, wf_decoded]:
            assert isinstance(wf, Wavefront)

            if to_return == "wf":
                assert torch.allclose(wf, test_wavefront)
            if to_return == "amps":
                assert torch.allclose(wf, test_wavefront.abs() + 0j)


def test_autoencoder_device(sim_params, empty_encoder_or_decoder):
    """Test .to(device) function for a D-RNN."""
    # empty D-RNN
    autoencoder = LinearAutoencoder(
        sim_params,
        encoder_elements_list=empty_encoder_or_decoder,
        decoder_elements_list=empty_encoder_or_decoder,
        device="cpu",
    )
    assert autoencoder.device == torch.device("cpu")

    new_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if new_device == torch.device(
        "cpu"
    ):  # if cuda is not available - check if `mps` is
        new_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    new_autoencoder = autoencoder.to(new_device)

    assert isinstance(new_autoencoder, LinearAutoencoder)
    assert new_autoencoder.device == new_device
