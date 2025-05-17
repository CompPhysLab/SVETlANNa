import pytest
import torch
from torch import nn

from svetlanna.wavefront import Wavefront
from svetlanna.networks.diffractive_conv import ConvDiffNetwork4F, ConvLayer4F
from svetlanna import elements

from tests.test_drnn import sim_params


@pytest.fixture()
def some_elements_list(sim_params):
    """Returns list with a zero distance FreeSpace, i.e. empty network"""
    h, w = sim_params.axes_size(axs=("H", "W"))

    elements_list = [
        elements.DiffractiveLayer(
            simulation_parameters=sim_params,
            mask=torch.rand(h, w)
            * 2
            * torch.pi,  # mask is not changing during the training!
        ),
        elements.FreeSpace(
            simulation_parameters=sim_params, distance=3.00 * 1e-2, method="AS"
        ),
    ]

    return elements_list


@pytest.mark.parametrize(
    "wf_real, wf_imag, focal_length",
    [
        (1.00, 0.00, 1.00 * 1e-2),
        (0.00, 1.00, 2.00 * 1e-2),
        (2.50, 1.25, 3.00 * 1e-2),
    ],
)
def test_conv4f_net_forward(
    sim_params, some_elements_list, wf_real, wf_imag, focal_length  # fixtures
):
    """Test forward function for a single Wavefront sequence."""
    h, w = sim_params.axes_size(
        axs=("H", "W")
    )  # size of a wavefront according to SimulationParameters
    test_wavefront = Wavefront(
        torch.ones(size=(h, w), dtype=torch.float64) * wf_real
        + torch.ones(size=(h, w), dtype=torch.float64) * wf_imag * 1j
    )

    random_diffractive_mask = (
        torch.rand(h, w) * 2 * torch.pi
    )  # random mask for a convolution

    # NETWORK
    conv4f_net = ConvDiffNetwork4F(
        sim_params=sim_params,
        network_elements_list=some_elements_list,
        focal_length=focal_length,
        conv_phase_mask=random_diffractive_mask,
        device=torch.get_default_device(),
    )

    # SEPARATE PARTS
    conv_layer = ConvLayer4F(
        sim_params=sim_params,
        focal_length=focal_length,
        conv_diffractive_mask=random_diffractive_mask,
    )
    net_after_conv = nn.Sequential(*some_elements_list)

    # COMPARE FORWARDS
    net_output_wf = conv4f_net(test_wavefront)
    sequential_output_wf = net_after_conv(conv_layer(test_wavefront))

    # ASSERTS
    assert isinstance(sequential_output_wf, Wavefront)
    assert isinstance(net_output_wf, Wavefront)
    assert torch.allclose(net_output_wf, sequential_output_wf)


def test_conv4f_net_device(sim_params, some_elements_list):
    """Test .to(device) function for a Convolutional Network."""
    h, w = sim_params.axes_size(
        axs=("H", "W")
    )  # size of a wavefront according to SimulationParameters
    random_diffractive_mask = torch.rand(h, w)  # random mask for a convolution

    # NETWORK
    conv4f_net = ConvDiffNetwork4F(
        sim_params=sim_params,
        network_elements_list=some_elements_list,
        focal_length=1.00 * 1e-2,
        conv_phase_mask=random_diffractive_mask,
        device="cpu",
    )

    assert conv4f_net.device == torch.device("cpu")

    new_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if new_device == torch.device(
        "cpu"
    ):  # if cuda is not available - check if `mps` is
        new_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    new_conv4f_net = conv4f_net.to(new_device)

    assert isinstance(new_conv4f_net, ConvDiffNetwork4F)
    assert new_conv4f_net.device == new_device
