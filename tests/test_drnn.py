import pytest
import torch
from torch import nn

from svetlanna import SimulationParameters
from svetlanna.elements import FreeSpace
from svetlanna.detector import Detector
from svetlanna.wavefront import Wavefront
from svetlanna.networks.diffractive_rnn import DiffractiveRNN


@pytest.fixture()
def sim_params():
    """Returns SimulationParameters object."""
    return SimulationParameters(
        {
            "W": torch.linspace(-1e-2 / 2, 1e-2 / 2, 10),
            "H": torch.linspace(-1e-2 / 2, 1e-2 / 2, 10),
            "wavelength": 800e-6,
        }
    )


@pytest.fixture()
def zero_free_space(sim_params):
    """Returns FreeSpace with a zero distance!"""
    return FreeSpace(simulation_parameters=sim_params, distance=0.0, method="AS")


@pytest.fixture()
def empty_layer(zero_free_space):
    """Returns nn.Sequentional with a zero distance FreeSpace, i.e. empty layer for RNN"""
    return nn.Sequential(zero_free_space)


@pytest.fixture()
def detector(sim_params):
    """Returns nn.Sequentional with a Detector for RNN detector_layer"""
    return nn.Sequential(Detector(sim_params, func="intensity"))


@pytest.mark.parametrize(
    "sequence_len, fusing_coeff, sequence_amplitudes",
    [
        (1, 0.30, [1.00]),
        (2, 0.25, [0.77, 0.13]),
        (3, 0.50, [0.80, 1.50, 2.10]),
    ],
)
def test_drnn_forward(
    sim_params,
    empty_layer,
    detector,  # fixtures
    sequence_len,
    fusing_coeff,
    sequence_amplitudes,
):
    """Test forward function for a single Wavefront sequence."""
    h, w = sim_params.axes_size(
        axs=("H", "W")
    )  # size of a wavefront according to SimulationParameters
    wavefront_seq = Wavefront(
        torch.ones(size=(sequence_len, h, w), dtype=torch.float64)
    )

    for step_ind in range(sequence_len):  # set amplitudes for a wavefront sequence
        wavefront_seq[step_ind, :, :] *= sequence_amplitudes[step_ind]

    # calculate expected values for an empty D-RNN with specified fusing coefficient
    hidden = 0.0
    for step_ind in range(sequence_len):
        input = sequence_amplitudes[step_ind]
        hidden = fusing_coeff * hidden + (1 - fusing_coeff) * input
    out_expected_val = hidden**2  # intensity output

    # empty D-RNN
    drnn = DiffractiveRNN(
        sim_params,
        sequence_len=sequence_len,
        fusing_coeff=fusing_coeff,
        read_in_layer=empty_layer,
        memory_layer=empty_layer,
        hidden_forward_layer=empty_layer,
        read_out_layer=empty_layer,
        detector_layer=detector,
        device=torch.get_default_device(),
    )
    # forward for D-RNN
    out_drnn = drnn(wavefront_seq)

    assert torch.allclose(
        out_drnn, torch.ones(size=(h, w), dtype=torch.float64) * out_expected_val
    )


@pytest.mark.parametrize(
    "batch_size, sequence_len, fusing_coeff, sequence_amplitudes",
    [
        (
            3,
            1,
            0.30,
            [
                [1.00],
                [0.40],
                [0.55],
            ],
        ),
        (
            3,
            2,
            0.25,
            [
                [0.77, 0.13],
                [0.10, 1.10],
                [2.20, 5.00],
            ],
        ),
        (
            2,
            3,
            0.50,
            [
                [0.80, 1.50, 2.10],
                [1.00, 2.00, 3.00],
            ],
        ),
    ],
)
def test_drnn_batch_forward(
    sim_params,
    empty_layer,
    detector,  # fixtures
    batch_size,
    sequence_len,
    fusing_coeff,
    sequence_amplitudes,
):
    """Test forward function for a batch of Wavefront sequences."""
    h, w = sim_params.axes_size(
        axs=("H", "W")
    )  # size of a wavefront according to SimulationParameters
    wavefront_seq_batch = Wavefront(
        torch.ones(size=(batch_size, sequence_len, h, w), dtype=torch.float64)
    )

    for seq_ind in range(batch_size):
        for step_ind in range(sequence_len):  # set amplitudes for a wavefront sequence
            wavefront_seq_batch[seq_ind, step_ind, :, :] *= sequence_amplitudes[
                seq_ind
            ][step_ind]

    # calculate expected values for an empty D-RNN with specified fusing coefficient
    out_expected_values = []
    for seq_ind in range(batch_size):
        hidden = 0.0
        for step_ind in range(sequence_len):
            input = sequence_amplitudes[seq_ind][step_ind]
            hidden = fusing_coeff * hidden + (1 - fusing_coeff) * input
        out_expected_val = hidden**2  # intensity output
        out_expected_values.append(out_expected_val)

    # empty D-RNN
    drnn = DiffractiveRNN(
        sim_params,
        sequence_len=sequence_len,
        fusing_coeff=fusing_coeff,
        read_in_layer=empty_layer,
        memory_layer=empty_layer,
        hidden_forward_layer=empty_layer,
        read_out_layer=empty_layer,
        detector_layer=detector,
        device=torch.get_default_device(),
    )
    # forward for D-RNN
    out_drnn = drnn(wavefront_seq_batch)

    for ind_seq in range(batch_size):
        assert torch.allclose(
            out_drnn[ind_seq, :, :],
            torch.ones(size=(h, w), dtype=torch.float64) * out_expected_values[ind_seq],
        )


def test_drnn_device(sim_params, empty_layer, detector):
    """Test .to(device) function for a D-RNN."""
    # empty D-RNN
    drnn = DiffractiveRNN(
        sim_params,
        sequence_len=3,
        fusing_coeff=0.5,  # some values
        read_in_layer=empty_layer,
        memory_layer=empty_layer,
        hidden_forward_layer=empty_layer,
        read_out_layer=empty_layer,
        detector_layer=detector,
        device="cpu",
    )
    assert drnn.device == torch.device("cpu")

    new_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if new_device == torch.device(
        "cpu"
    ):  # if cuda is not available - check if `mps` is
        new_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    new_drnn = drnn.to(new_device)

    assert isinstance(new_drnn, DiffractiveRNN)
    assert new_drnn.device == new_device
