import pytest
import torch

from svetlanna import elements
from svetlanna import SimulationParameters
from svetlanna import Wavefront


parameters = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "wavelength_test",
    "waist_radius_test",
    "distance_total",
    "distance_end",
    "expected_std",
    "error_energy"
]


# TODO: fix docstrings
@pytest.mark.parametrize(
    parameters,
    [(2, 2, 1500, 1600, 1064 * 1e-6, 1., 300, 200, 0.02, 0.01)]
)
def test_gaussian_beam_propagation(
    ox_size,
    oy_size,
    ox_nodes,
    oy_nodes,
    wavelength_test,
    waist_radius_test,
    distance_total,
    distance_end,
    expected_std,
    error_energy
):
    """Test for the free field propagation problem: free propagation of the
    Gaussian beam at the arbitrary distance

    Parameters
    ----------
    ox_size : float
        System size along the axis ox
    oy_size : float
        System size along the axis oy
    ox_nodes : int
        Number of computational nodes along the axis ox
    oy_nodes : int
        Number of computational nodes along the axis oy
    wavelength_test : float
        Wavelength for the incident field
    waist_radius_test : _type_
        Waist radius of the Gaussian beam
    distance_total : _type_
        Total propagation distance of the Gaussian beam
    distance_end : _type_
        Propagation distance of the Gaussian beam which calculates by using
        Fresnel propagation method or angular spectrum method
    expected_std : _type_
        Criterion for accepting the test(standard deviation)
    error_energy : _type_
        Criterion for accepting the test(energy loss by propagation)
    """

    x_linear = torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes)
    y_linear = torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes)
    x_grid, y_grid = torch.meshgrid(x_linear, y_linear, indexing='xy')

    wave_number = 2 * torch.pi / wavelength_test

    amplitude = 1.

    dx = ox_size / ox_nodes
    dy = oy_size / oy_nodes

    rayleigh_range = torch.pi * (waist_radius_test**2) / wavelength_test

    radial_distance_squared = torch.pow(x_grid, 2) + torch.pow(y_grid, 2)

    hyperbolic_relation = waist_radius_test * (1 + (
        distance_total / rayleigh_range)**2)**(1/2)

    radius_of_curvature = distance_total * (
        1 + (rayleigh_range / distance_total)**2
    )

    # Gouy phase
    gouy_phase = torch.arctan(torch.tensor(distance_total / rayleigh_range))

    # analytical equation for the propagation of the Gaussian beam
    field = amplitude * (waist_radius_test / hyperbolic_relation) * (
        torch.exp(-radial_distance_squared / (hyperbolic_relation)**2) * (
            torch.exp(-1j * (wave_number * distance_total + wave_number * (
                radial_distance_squared) / (2 * radius_of_curvature) - (
                    gouy_phase)))))

    intensity_analytic = torch.pow(torch.abs(field), 2)

    params = SimulationParameters(
        {
            'W': torch.linspace(-ox_size/2, ox_size/2, ox_nodes),
            'H': torch.linspace(-oy_size/2, oy_size/2, oy_nodes),
            'wavelength': wavelength_test
        }
    )

    distance_start = distance_total - distance_end

    field_gb_start = Wavefront.gaussian_beam(
        simulation_parameters=params,
        distance=distance_start,
        waist_radius=waist_radius_test
    )

    # field on the screen by using Fresnel propagation method
    field_end_fresnel = elements.FreeSpace(
        simulation_parameters=params, distance=distance_end, method='fresnel'
    ).forward(input_field=field_gb_start)
    # field on the screen by using angular spectrum method
    field_end_as = elements.FreeSpace(
        simulation_parameters=params, distance=distance_end, method='fresnel'
    ).forward(input_field=field_gb_start)

    intensity_output_fresnel = torch.pow(torch.abs(field_end_fresnel), 2)
    intensity_output_as = torch.pow(torch.abs(field_end_as), 2)

    energy_analytic = torch.sum(intensity_analytic) * dx * dy
    energy_numeric_fresnel = torch.sum(intensity_output_fresnel) * dx * dy
    energy_numeric_as = torch.sum(intensity_output_as) * dx * dy

    standard_deviation_fresnel = torch.std(
        intensity_output_fresnel - intensity_analytic
    )
    standard_deviation_as = torch.std(intensity_output_as - intensity_analytic)

    energy_error_fresnel = torch.abs(
        (energy_analytic - energy_numeric_fresnel) / energy_analytic
    )
    energy_error_as = torch.abs(
        (energy_analytic - energy_numeric_as) / energy_analytic
    )

    assert standard_deviation_fresnel <= expected_std
    assert standard_deviation_as <= expected_std
    assert energy_error_fresnel <= error_energy
    assert energy_error_as <= error_energy
