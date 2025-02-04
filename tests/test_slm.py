import pytest
import torch

from svetlanna import elements
from svetlanna import SimulationParameters


parameters = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "number_of_levels",
    "expected_error"
]


@pytest.mark.skip
@pytest.mark.parametrize(
    parameters,
    [(10, 10, 1000, 1200, 256, 1e-5),
     (15, 8, 1920, 1080, 512, 1e-5)]
)
def test_slm(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    number_of_levels: int,
    expected_error: float
):

    x_length = torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes)
    y_length = torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes)

    params = SimulationParameters(
        axes={
            'W': x_length,
            'H': y_length,
            'wavelength': 1064 * 1e-6,
        }
    )

    mask = torch.randint(0, number_of_levels, (ox_nodes, oy_nodes))
    slm = elements.SpatialLightModulator(
        simulation_parameters=params,
        mask=mask,
        number_of_levels=number_of_levels
    )

    transmission_function = slm.get_transmission_function()

    transmission_function_analytic = torch.exp(
        1j * 2 * torch.pi / number_of_levels * mask
    )

    assert torch.allclose(
        transmission_function_analytic,
        transmission_function,
        atol=expected_error
    )
    assert slm.number_of_levels == number_of_levels


parameters_mask = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "height",
    "width",
    "mode",
    "mask",
    "resized_mask"
]


@pytest.mark.parametrize(
    parameters_mask,
    [(10,
      10,
      4,
      4,
      10,
      10,
      "nearest",
      torch.Tensor([[1., 2.], [3., 4.]]),
      torch.Tensor([[1., 1., 2., 2.,], [1., 1., 2., 2.,], [3., 3., 4., 4.,], [3., 3., 4., 4.,]])),  # noqa: E501
     (15,
      8,
      6,
      6,
      8,
      15,
      "nearest",
      torch.Tensor([[2., 3.], [4., 5.]]),
      torch.Tensor([
          [2., 2., 2., 3., 3., 3.],
          [2., 2., 2., 3., 3., 3.],
          [2., 2., 2., 3., 3., 3.],
          [4., 4., 4., 5., 5., 5.],
          [4., 4., 4., 5., 5., 5.],
          [4., 4., 4., 5., 5., 5.]])
        )]
)
def test_slm_mask(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    height: float,
    width: float,
    mode: str,
    mask: torch.Tensor,
    resized_mask: torch.Tensor
):
    x_length = torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes)
    y_length = torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes)

    params = SimulationParameters(
        axes={
            'W': x_length,
            'H': y_length,
            'wavelength': 1064 * 1e-6,
        }
    )

    slm = elements.SpatialLightModulator(
        simulation_parameters=params,
        mask=mask,
        height=height,
        width=width,
        mode=mode
    )
    slm.get_aperture
    resized_mask_slm = slm.resize_mask

    assert torch.allclose(resized_mask, resized_mask_slm)
