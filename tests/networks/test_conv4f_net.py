import pytest
import torch
from torch import nn

from svetlanna import Wavefront, SimulationParameters
from svetlanna.networks.diffractive_conv import ConvDiffNetwork4F, ConvLayer4F
from svetlanna import elements


@pytest.mark.parametrize(
    "wf_real, wf_imag, focal_length",
    [
        (1.00, 0.00, 1.00 * 1e-2),
        (0.00, 1.00, 2.00 * 1e-2),
        (2.50, 1.25, 3.00 * 1e-2),
    ],
)
def test_conv4f_net_forward(
    sim_params_simple: SimulationParameters,
    wf_real,
    wf_imag,
    focal_length,  # fixtures
):
    """Test forward function for a single Wavefront sequence."""
    h, w = sim_params_simple.axis_sizes(
        axs=("y", "x")
    )  # size of a wavefront according to SimulationParameters
    test_wavefront = (
        torch.ones(size=(h, w), dtype=torch.float64) * wf_real
        + torch.ones(size=(h, w), dtype=torch.float64) * wf_imag * 1j
    )

    test_wavefront = Wavefront(sim_params_simple.cast(test_wavefront, "y", "x"))

    random_diffractive_mask = (
        torch.rand(h, w) * 2 * torch.pi
    )  # random mask for a convolution

    elements_list = [
        elements.DiffractiveLayer(
            simulation_parameters=sim_params_simple,
            mask=torch.rand(h, w)
            * 2
            * torch.pi,  # mask is not changing during the training!
        ),
        elements.FreeSpace(
            simulation_parameters=sim_params_simple, distance=3.00 * 1e-2, method="AS"
        ),
    ]

    # NETWORK
    conv4f_net = ConvDiffNetwork4F(
        simulation_parameters=sim_params_simple,
        network_elements=elements_list,
        focal_length=focal_length,
        conv_diffractive_mask=random_diffractive_mask,
    )

    # SEPARATE PARTS
    conv_layer = ConvLayer4F(
        simulation_parameters=sim_params_simple,
        focal_length=focal_length,
        conv_diffractive_mask=random_diffractive_mask,
    )
    net_after_conv = nn.Sequential(*elements_list)

    # COMPARE FORWARDS
    net_output_wf = conv4f_net(test_wavefront)
    sequential_output_wf = net_after_conv(conv_layer(test_wavefront))

    # ASSERTS
    assert isinstance(sequential_output_wf, Wavefront)
    assert isinstance(net_output_wf, Wavefront)
    assert torch.allclose(net_output_wf, sequential_output_wf)


def test_device(device_simple: str):
    """Test reservoir on different devices."""

    sim_params = SimulationParameters(
        x=torch.tensor([0]), y=torch.tensor([0]), wavelength=1.0
    )
    wavefront = Wavefront.plane_wave(sim_params).to(device=device_simple)

    assert sim_params.device == torch.get_default_device()
    conv4f_net = ConvDiffNetwork4F(
        simulation_parameters=sim_params,
        network_elements=[
            elements.DiffractiveLayer(
                simulation_parameters=sim_params,
                mask=torch.rand(
                    sim_params.axis_sizes(("y", "x")), device=sim_params.device
                ),
            )
        ],
        focal_length=1.0,
        conv_diffractive_mask=torch.rand(
            sim_params.axis_sizes(("y", "x")), device=sim_params.device
        ),
    )
    conv4f_net.to(device=device_simple)

    assert conv4f_net(wavefront).device.type == device_simple

    # Simulation parameters on device
    sim_params.to(device=device_simple)

    assert sim_params.device.type == device_simple
    conv4f_net = ConvDiffNetwork4F(
        simulation_parameters=sim_params,
        network_elements=[
            elements.DiffractiveLayer(
                simulation_parameters=sim_params,
                mask=torch.rand(
                    sim_params.axis_sizes(("y", "x")), device=sim_params.device
                ),
            )
        ],
        focal_length=1.0,
        conv_diffractive_mask=torch.rand(
            sim_params.axis_sizes(("y", "x")), device=sim_params.device
        ),
    )

    assert conv4f_net(wavefront).device.type == device_simple


def test_to_specs():
    """Stupid test to increase code coverage."""
    sim_params = SimulationParameters(
        x=torch.tensor([0]), y=torch.tensor([0]), wavelength=1.0
    )

    conv4f_net = ConvDiffNetwork4F(
        simulation_parameters=sim_params,
        network_elements=[
            elements.DiffractiveLayer(
                simulation_parameters=sim_params,
                mask=torch.rand(
                    sim_params.axis_sizes(("y", "x")), device=sim_params.device
                ),
            )
        ],
        focal_length=1.0,
        conv_diffractive_mask=torch.rand(
            sim_params.axis_sizes(("y", "x")), device=sim_params.device
        ),
    )

    assert conv4f_net.to_specs()
    assert conv4f_net.conv_layer.to_specs()
