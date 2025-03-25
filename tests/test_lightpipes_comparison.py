import pytest
import torch

import LightPipes as lp
import svetlanna as sv

from svetlanna import elements


parameters = [
    "ox_size",
    "ox_nodes",
    "wavelength",
    "radius",
    "distance",
    "focal_length"
]


# TODO: fix docstrings
@pytest.mark.parametrize(
    parameters,
    [
        (
            25 * lp.mm,  # ox_size
            3000,   # ox_nodes
            1064 * lp.nm,  # wavelength, mm
            2 * lp.mm,     # radius, mm
            2000 * lp.mm,    # distance, mm
            2000 * lp.mm,    # focal_length, mm
        ),
        (
            25 * lp.mm,  # ox_size
            3000,   # ox_nodes
            1064 * lp.nm,  # wavelength, mm
            2 * lp.mm,     # radius, mm
            100 * lp.mm,    # distance, mm
            20 * lp.mm,    # focal_length, mm
        ),
        (
            25 * lp.mm,  # ox_size
            100,   # ox_nodes
            123 * lp.nm,  # wavelength, mm
            2 * lp.mm,     # radius, mm
            200 * lp.mm,    # distance, mm
            2100 * lp.mm,    # focal_length, mm
        )
    ]
)
def test_circular_aperture(
    ox_size: float,
    ox_nodes: int,
    wavelength: float,
    radius: float,
    distance: float,
    focal_length: float
):
    # ----------------------------------
    #   LightPipes fields calculations
    # ----------------------------------
    F = lp.Begin(ox_size, wavelength, ox_nodes)
    F = lp.CircAperture(F, radius)
    F = lp.Forvard(F, distance)
    field_before_lens_lp = torch.tensor(F.field)
    F = lp.Lens(F, focal_length)
    F = lp.Forvard(F, focal_length)
    field_output_lp = torch.tensor(F.field)

    # ----------------------------------
    #   SVETlANNa fields calculations
    # ----------------------------------
    oy_size = ox_size
    oy_nodes = ox_nodes
    x_length = torch.linspace(
        -ox_size / 2, ox_size / 2, ox_nodes, dtype=torch.float64
    )
    y_length = torch.linspace(
        -oy_size / 2, oy_size / 2, oy_nodes, dtype=torch.float64
    )

    simulation_parameters = sv.SimulationParameters(
        axes={
            'W': x_length,
            'H': y_length,
            'wavelength': torch.tensor(wavelength, dtype=torch.float64)
        }
    )
    # elements' definitions
    aperture = elements.RoundAperture(
        simulation_parameters,
        radius
    )
    fs1 = elements.FreeSpace(
        simulation_parameters,
        distance,
        method='fresnel'
    )
    lens = elements.ThinLens(
        simulation_parameters,
        focal_length
    )
    fs2 = elements.FreeSpace(
        simulation_parameters,
        focal_length,
        method='fresnel'
    )

    # field calculations
    G = sv.Wavefront.plane_wave(simulation_parameters)
    G = aperture(G)
    G = fs1(G)
    field_before_lens_sv = G
    G = lens(G)
    G = fs2(G)
    field_output_sv = G

    # it is better to compare normalized fields
    before_lens_norm = torch.max(torch.abs(field_before_lens_lp))
    output_norm = torch.max(torch.abs(field_output_lp))

    # ----------------------------------
    #          results testing
    # ----------------------------------
    assert torch.mean(
        torch.abs(field_before_lens_lp - field_before_lens_sv)
    ) / before_lens_norm < 0.01

    assert torch.mean(
        torch.abs(field_output_lp - field_output_sv)
    ) / output_norm < 0.01

# TODO: сравнить пиковую мощность и положение максимумов
