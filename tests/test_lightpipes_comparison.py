import pytest
import torch
import numpy as np

from LightPipes import *
from svetlanna import elements
from svetlanna import SimulationParameters
from svetlanna import wavefront as w
torch.set_default_dtype(torch.float64)

parameters = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "wavelength",
    "radius",
    "distance",
    "focal_length",
    "expected_std",
    "error_energy"
]


# TODO: fix docstrings
@pytest.mark.parametrize(
    parameters,
    [
        (
            25,  # ox_size
            25,  # oy_size
            3000,   # ox_nodes
            3000,   # oy_nodes
            1064 * 1e-6,  # wavelength, mm
            4.,     # radius, mm
            800,    # distance, mm
            1000,    # focal_length, mm
            0.02,   # expected_std
            0.01    # error_energy
        )
    ]
)
def test_circular_aperture(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    wavelength: torch.Tensor,
    radius: float,
    distance: float,
    focal_length: float,
    expected_std: float,
    error_energy: float
):
    F = Begin(ox_size*mm, wavelength*mm, ox_nodes)
    F = CircAperture(radius*mm, 0, 0, F)
    F = Fresnel(F,distance*mm)
    intensity_before_lens_lightpipes = torch.tensor(Intensity(F))
    F = Lens(F, focal_length*mm)
    F = Forvard(focal_length*mm, F)
    output_intensity_lightpipes = torch.tensor(Intensity(0, F))

    x_length = torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes)
    y_length = torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes)

    dx = ox_size / ox_nodes
    dy = oy_size / oy_nodes

    params = SimulationParameters(
        axes={
            'W': x_length,
            'H': y_length,
            'wavelength': wavelength
            }
    )

    # create plane  before the aperture
    incident_field = w.Wavefront.plane_wave(
        simulation_parameters=params,
        distance=0.,
        wave_direction=[0., 0., 1.]
    )

    aperture = elements.RoundAperture(
        simulation_parameters=params,
        radius=radius
    )

    field_after_aperture = aperture.forward(input_field=incident_field)

    fs1 = elements.FreeSpace(
        simulation_parameters=params,
        distance=distance,
        method='AS'
    )
    field_before_lens = fs1.forward(input_field=field_after_aperture)

    intensity_before_lens = field_before_lens.intensity
    intensity_before_lens = intensity_before_lens[0, :, :]

    lens = elements.ThinLens(
        simulation_parameters=params,
        focal_length=focal_length,
        radius=radius
    )

    field_after_lens = lens.forward(input_field=field_before_lens)

    fs2 = elements.FreeSpace(
        simulation_parameters=params,
        distance=focal_length,
        method='AS'
    )

    output_field = fs2.forward(input_field=field_after_lens)

    output_intensity = output_field.intensity[0, :, :]

    energy_before_lens = torch.sum(intensity_before_lens) * dx * dy
    energy_before_lens_lightpipes = torch.sum(
        intensity_before_lens_lightpipes
    ) * dx * dy

    error = torch.abs(
        energy_before_lens - energy_before_lens_lightpipes
    ) / energy_before_lens

    assert error <= error_energy
    assert torch.std(
        output_intensity - output_intensity_lightpipes
    ) <= expected_std
    assert torch.std(
        intensity_before_lens - intensity_before_lens_lightpipes
    ) <= expected_std

# TODO: сравнить пиковую мощность и положение максимумов