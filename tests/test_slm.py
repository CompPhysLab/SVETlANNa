import pytest
import torch

from svetlanna import elements
from svetlanna import SimulationParameters
from svetlanna import wavefront as w


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
    [
        (10, 10, 4, 4, 10, 10, "nearest", torch.Tensor([[1., 2.], [3., 4.]]),
            torch.Tensor([
                [1., 1., 2., 2.,],
                [1., 1., 2., 2.,],
                [3., 3., 4., 4.,],
                [3., 3., 4., 4.,]
            ])
         ),
        (15, 8, 6, 6, 8, 15, "nearest", torch.Tensor([[2., 3.], [4., 5.]]),
            torch.Tensor([
                [2., 2., 2., 3., 3., 3.],
                [2., 2., 2., 3., 3., 3.],
                [2., 2., 2., 3., 3., 3.],
                [4., 4., 4., 5., 5., 5.],
                [4., 4., 4., 5., 5., 5.],
                [4., 4., 4., 5., 5., 5.]
            ])
         )
    ]
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
    resized_mask_slm = slm.resized_mask

    assert torch.allclose(resized_mask, resized_mask_slm)


parameters_resize = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "mode",
    "mask"
]


@pytest.mark.parametrize(
    parameters_resize,
    [
        (10, 10, 1000, 1200, "nearest", torch.rand(100, 100)),
        (9.7, 11, 1100, 1200, "bilinear", torch.rand(100, 100)),
        (6, 5, 1570, 632, "bicubic", torch.rand(100, 100)),
        (15.8, 8.61, 109, 120, "area", torch.rand(100, 100)),
        (19, 7, 1089, 2007, "nearest-exact", torch.rand(100, 100)),
        (15, 8, 300, 400, "nearest-exact", torch.rand(1080, 1920))
    ]
)
def test_slm_resize(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    mode: str,
    mask: torch.Tensor
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
        height=oy_size,
        width=ox_size,
        mode=mode
    )
    aperture = slm.get_aperture
    resized_mask = slm.resized_mask

    assert resized_mask.size() == (oy_nodes, ox_nodes)
    assert aperture.size() == (oy_nodes, ox_nodes)


# test slm aperture

parameters_aperture = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "height",
    "width",
    "location",
    "aperture"
]


@pytest.mark.parametrize(
    parameters_aperture,
    [
        (6, 5, 6, 5, 3, 3, (-1.5, -1),
            torch.tensor([
                [1., 1., 1., 0., 0., 0.],
                [1., 1., 1., 0., 0., 0.],
                [1., 1., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.]
            ])
         ),
        (6, 5, 6, 5, 3, 3, (-1.5, 1),
            torch.tensor([
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 0., 0., 0.],
                [1., 1., 1., 0., 0., 0.],
                [1., 1., 1., 0., 0., 0.]
            ])
         ),
        (6, 5, 6, 5, 3, 3, (1.5, 1),
            torch.tensor([
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 1., 1.],
                [0., 0., 0., 1., 1., 1.],
                [0., 0., 0., 1., 1., 1.]
            ])
         ),
        (6, 5, 6, 5, 3, 3, (1.5, -1),
            torch.tensor([
                [0., 0., 0., 1., 1., 1.],
                [0., 0., 0., 1., 1., 1.],
                [0., 0., 0., 1., 1., 1.],
                [0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0.]
            ])
         ),
        (6, 5, 6, 5, 3, 3, (-100, 100), torch.zeros(5, 6))
    ]
)
def test_slm_aperture(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    height: float,
    width: float,
    location: tuple,
    aperture: torch.Tensor
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
        mask=torch.zeros(ox_nodes, oy_nodes),
        height=height,
        width=width,
        location=location
    )
    slm.get_aperture

    assert torch.allclose(aperture, slm.aperture)


parameters_propagation = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "height",
    "width",
    "location",
    "mask",
    "wavelength",
    "mode"
]


@pytest.mark.parametrize(
    parameters_propagation,
    [
        (10, 10, 1000, 1200, 3., 4., (0., 0.),
         torch.rand(100, 100), torch.linspace(330, 1064, 4) * 1e-6,
         "nearest"
         ),
        (9, 12, 1000, 1200, 3., 4., (-2., 3.),
         torch.rand(100, 100), torch.linspace(330, 1064, 4) * 1e-6,
         "bilinear"
         ),
        (15.8, 8.61, 1920, 1080, 2., 2., (2., 0.),
         torch.rand(100, 100), torch.linspace(330, 1064, 4) * 1e-6,
         "bicubic"
         ),
        (30, 15, 1920*2, 1080*2, 15.8, 8.61, (-1., 1.),
         torch.rand(1080, 1920), torch.linspace(330, 1064, 4) * 1e-6,
         "bicubic"
         ),
        # (5, 9, 100, 400, 3., 4., (-1., 8.),
        #  torch.rand(100, 100), torch.linspace(330, 1064, 4) * 1e-6,
        #  "area"
        #  ),
        # (8, 6, 1011, 1213, 3., 4., (-1., 2.),
        #  torch.rand(100, 100), torch.linspace(330, 1064, 4) * 1e-6,
        #  "nearest-exact"
        #  )
    ]
)
def test_slm_propagation(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    height: float,
    width: float,
    location: tuple,
    mask: torch.Tensor,
    wavelength: float,
    mode: str
):

    x_length = torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes)
    y_length = torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes)

    params = SimulationParameters(
        axes={
            'W': x_length,
            'H': y_length,
            'wavelength': wavelength,
        }
    )

    slm = elements.SpatialLightModulator(
        simulation_parameters=params,
        mask=mask,
        height=height,
        width=width,
        location=location,
        mode=mode
    )

    incident_field = w.Wavefront.gaussian_beam(
        simulation_parameters=params,
        waist_radius=2.,
        distance=100
    )
    transmitted_field = slm.forward(input_field=incident_field)

    assert transmitted_field.size() == (4, oy_nodes, ox_nodes)
