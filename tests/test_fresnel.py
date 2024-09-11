from svetlanna import elements
from svetlanna import SimulationParameters
from svetlanna import beam_generator
from examples import analytical_solutions as anso
import pytest
import torch
import numpy as np

square_parameters = "ox_size, oy_size, ox_nodes, oy_nodes," + "wavelength_test, distance_test, square_size_test," + "expected_std,error_energy"


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

    wave_number = 2 * torch.pi / wavelength_test

    params = SimulationParameters(
        x_size=ox_size,
        y_size=oy_size,
        x_nodes=ox_nodes,
        y_nodes=oy_nodes,
        wavelength=wavelength_test
    )

    dx = ox_size / ox_nodes
    dy = oy_size / oy_nodes

    incident_field = beam_generator.PlaneWave(
        simulation_parameters=params
    ).forward(
        distance=distance_test, wave_vector=torch.tensor([0., 0., wave_number])
    )

    # field after the square aperture
    transmission_field = elements.RectangularAperture(
        simulation_parameters=params,
        height=square_size_test,
        width=square_size_test
    ).forward(input_field=incident_field)

    # field on the screen
    output_field = elements.FreeSpace(
        simulation_parameters=params,
        distance=distance_test,
        method='fresnel'
        ).forward(input_field=transmission_field)

    # intensity distribution on the screen
    intensity_output = (torch.pow(torch.abs(output_field), 2)).detach().numpy()

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
    energy_numeric = np.sum(intensity_output) * dx * dy

    standard_deviation = np.std(intensity_analytic - intensity_output)
    error = np.abs((energy_analytic - energy_numeric) / energy_analytic)

    assert standard_deviation <= expected_std
    assert error <= error_energy


parameters = "ox_size, oy_size, ox_nodes, oy_nodes," + "wavelength_test, waist_radius_test, distance_total, distance_end," + "expected_std, error_energy"


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

    x_linear = torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes)
    y_linear = torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes)
    x_grid, y_grid = torch.meshgrid(x_linear, y_linear, indexing='xy')

    wave_number = 2 * torch.pi / wavelength_test

    amplitude = 1.

    dx = ox_size / ox_nodes
    dy = oy_size / oy_nodes

    rayleigh_range = torch.pi * (waist_radius_test**2)  / wavelength_test

    radial_distance_squared = torch.pow(x_grid, 2) + torch.pow(y_grid, 2)

    hyperbolic_relation = waist_radius_test * (1 + (
        distance_total / rayleigh_range)**2)**(1/2)

    radius_of_curvature = distance_total * (
        1 + (rayleigh_range / distance_total)**2
    )

    # Gouy phase
    gouy_phase = torch.arctan(torch.tensor(distance_total / rayleigh_range))

    field = amplitude * (waist_radius_test / hyperbolic_relation) * (
        torch.exp(-radial_distance_squared / (hyperbolic_relation)**2) * (
            torch.exp(-1j * (wave_number * distance_total + wave_number * (
                radial_distance_squared) / (2 * radius_of_curvature) - (
                    gouy_phase)))))

    intensity_analytic = torch.pow(torch.abs(field), 2)

    params = SimulationParameters(
        x_size=ox_size,
        y_size=oy_size,
        x_nodes=ox_nodes,
        y_nodes=oy_nodes,
        wavelength=wavelength_test
    )

    distance_start = distance_total - distance_end

    field_gb_start = beam_generator.GaussianBeam(
        simulation_parameters=params
    ).forward(distance=distance_start, waist_radius=waist_radius_test)
    field_gb_end = elements.FreeSpace(
        simulation_parameters=params, distance=distance_end, method='fresnel'
    ).forward(input_field=field_gb_start)

    intensity_output = torch.pow(torch.abs(field_gb_end), 2)

    energy_analytic = torch.sum(intensity_analytic) * dx * dy
    energy_numeric = torch.sum(intensity_output) * dx * dy

    standard_deviation = torch.std(intensity_output - intensity_analytic)
    error = torch.abs((energy_analytic - energy_numeric) / energy_analytic)

    assert standard_deviation <= expected_std
    assert error <= error_energy
