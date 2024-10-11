# TODO: rename to test_analytic.py
# TODO: docstrings

from svetlanna import elements
from svetlanna import SimulationParameters
from svetlanna import Wavefront

from examples import analytical_solutions as anso

import pytest
import torch
import numpy as np

square_parameters = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "wavelength_test",
    "distance_test",
    "square_size_test",
    "expected_std",
    "error_energy"
]


@pytest.mark.parametrize(
    square_parameters,
    [(3, 3, 1000, 1000, 1064 * 1e-6, 150, 1.5, 0.065, 0.05),
     (4, 4, 1000, 1000, 660 * 1e-6, 600, 1, 0.05, 0.05)]
)
def test_rectangle_fresnel(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    wavelength_test: float,
    distance_test: float,
    square_size_test: float,
    expected_std: float,
    error_energy: float
):

    params = SimulationParameters(
        x_size=ox_size,
        y_size=oy_size,
        x_nodes=ox_nodes,
        y_nodes=oy_nodes,
        wavelength=wavelength_test
    )

    dx = ox_size / ox_nodes
    dy = oy_size / oy_nodes

    incident_field = Wavefront.plane_wave(
        simulation_parameters=params,
        distance=distance_test,
        wave_direction=[0, 0, 1]
    )

    # field after the square aperture
    transmission_field = elements.RectangularAperture(
        simulation_parameters=params,
        height=square_size_test,
        width=square_size_test
    ).forward(input_field=incident_field)

    # field on the screen by using Fresnel propagation method
    output_field_fresnel = elements.FreeSpace(
        simulation_parameters=params,
        distance=distance_test,
        method='fresnel'
        ).forward(input_field=transmission_field)
    # field on the screen by using Angular Spectrum method
    output_field_as = elements.FreeSpace(
        simulation_parameters=params,
        distance=distance_test,
        method='AS'
        ).forward(input_field=transmission_field)

    # intensity distribution on the screen by using Fresnel propagation method
    intensity_output_fresnel = (
        torch.pow(torch.abs(output_field_fresnel), 2)
    ).detach().numpy()
    # intensity distribution on the screen by using Angular Spectrum method
    intensity_output_as = (
        torch.pow(torch.abs(output_field_as), 2)
    ).detach().numpy()

    # analytical intensity distribution on the screen
    intensity_analytic = anso.SquareFresnel(
        distance=distance_test,
        x_size=ox_size,
        y_size=oy_size,
        x_nodes=ox_nodes,
        y_nodes=oy_nodes,
        square_size=square_size_test,
        wavelength=wavelength_test
    ).intensity()

    energy_analytic = np.sum(intensity_analytic) * dx * dy
    energy_numeric_fresnel = np.sum(intensity_output_fresnel) * dx * dy
    energy_numeric_as = np.sum(intensity_output_as) * dx * dy

    standard_deviation_fresnel = np.std(
        intensity_analytic - intensity_output_fresnel
    )
    standard_deviation_as = np.std(
        intensity_analytic - intensity_output_as
    )

    energy_error_fresnel = np.abs(
        (energy_analytic - energy_numeric_fresnel) / energy_analytic
    )
    energy_error_as = np.abs(
        (energy_analytic - energy_numeric_as) / energy_analytic
    )

    assert standard_deviation_fresnel <= expected_std
    assert standard_deviation_as <= expected_std
    assert energy_error_fresnel <= error_energy
    assert energy_error_as <= error_energy
