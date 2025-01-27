import pytest
import torch
import numpy as np

from svetlanna import elements
from svetlanna import SimulationParameters
from svetlanna import Wavefront
from svetlanna.phase_retrieval_problem import phase_retrieval
from svetlanna import LinearOpticalSetup


parameters = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "wavelength_test",
    "waist_radius_test",
    "distance_test",
    "radius_test",
    "error_energy"
]


@pytest.mark.skip
@pytest.mark.parametrize(
    parameters,
    [
        (10, 10, 1000, 1000, 660 * 1e-6, 0.5, 100., 10., 1e-4),
        (7, 8, 1000, 1000, 1064 * 1e-6, 0.7, 150., 10., 1e-4),
        (15, 8, 1500, 1000, 550 * 1e-6, 0.5, 120., 10., 1e-4)
    ]
)
def test_phase_retrieval(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    wavelength_test: float,
    waist_radius_test: float,
    distance_test: float,
    radius_test: float,
    error_energy: float
):
    """Test for phase reconstruction problem and generate target intensity
    problem using HIO and Gerchberg-Saxton algorithms on the example of a
    single lens optical setup

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
    distance_test : float
        Distance between the lens and the screen
    radius_test : float
        Radius of the thin lens
    error_energy : float
        Criterion for accepting the test(energy loss)
    """
    x_linear = torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes)
    y_linear = torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes)
    x_grid, y_grid = torch.meshgrid(x_linear, y_linear, indexing='xy')

    dx = ox_size / ox_nodes
    dy = oy_size / oy_nodes

    params = SimulationParameters(
        {
            'W': torch.linspace(-ox_size/2, ox_size/2, ox_nodes),
            'H': torch.linspace(-oy_size/2, oy_size/2, oy_nodes),
            'wavelength': wavelength_test
        }
    )

    field_before_lens1 = Wavefront.gaussian_beam(
        simulation_parameters=params,
        distance=5 * distance_test,
        waist_radius=waist_radius_test
    )

    intensity_source = field_before_lens1.intensity.detach().numpy()

    lens1 = elements.ThinLens(
        simulation_parameters=params,
        focal_length=distance_test,
        radius=radius_test
    )

    field_after_lens1 = lens1.forward(input_field=field_before_lens1)

    free_space1 = elements.FreeSpace(
        simulation_parameters=params,
        distance=torch.tensor(3 * distance_test),
        method='AS'
    )
    output_field = free_space1.forward(input_field=field_after_lens1)

    # target phase profile for phase reconstruction problem
    phase_target = (
        torch.angle(output_field) + 2 * torch.pi * (
            torch.angle(output_field) < 0.
        ).float()
    ).detach().numpy()

    # target intensity profile on the screen
    intensity_target = output_field.intensity.detach().numpy()

    optical_setup = LinearOpticalSetup([free_space1])

    goal = (x_grid**2 + y_grid ** 2 <= 5).float()

    result_hio = phase_retrieval.retrieve_phase(
        source_intensity=torch.tensor(intensity_source),
        optical_setup=optical_setup,
        target_intensity=torch.tensor(intensity_target),
        target_phase=torch.tensor(phase_target),
        target_region=goal,
        initial_phase=None,
        method='HIO'
    )

    phase_reconstruction_hio = result_hio.solution

    result_gs = phase_retrieval.retrieve_phase(
        source_intensity=torch.tensor(intensity_source),
        optical_setup=optical_setup,
        target_intensity=torch.tensor(intensity_target),
        target_phase=None,
        target_region=None,
        initial_phase=None,
        method='GS'
    )

    phase_get_intensity = result_gs.solution

    step = 2 * torch.pi / 256
    mask_reconstruction_hio = phase_reconstruction_hio // step
    mask_get_intensity = phase_get_intensity // step

    field_after_slm = elements.SpatialLightModulator(
        simulation_parameters=params,
        mask=mask_reconstruction_hio
    ).forward(field_before_lens1)

    output_field = optical_setup.forward(field_after_slm)
    intensity_target_opt = output_field.intensity.detach().numpy()

    energy_reconstruction_hio = np.sum(intensity_target_opt) * dx * dy
    energy_true = np.sum(intensity_target) * dx * dy

    field_after_slm = elements.SpatialLightModulator(
        simulation_parameters=params,
        mask=mask_get_intensity
    ).forward(field_before_lens1)

    output_field = optical_setup.forward(field_after_slm)
    intensity_target_opt = output_field.intensity.detach().numpy()

    energy_get_intensity = np.sum(intensity_target_opt) * dx * dy

    assert np.abs(energy_true - energy_reconstruction_hio) <= error_energy
    assert np.abs(energy_true - energy_get_intensity) <= error_energy


parameters_4f = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "wavelength_test",
    "waist_radius_test",
    "distance_test",
    "radius_test",
    "error_energy"
]


@pytest.mark.skip
@pytest.mark.parametrize(
    parameters_4f,
    [
        (10, 10, 1000, 1000, 660 * 1e-6, 0.5, 100., 10., 1e-4),
        (7, 8, 1000, 1000, 1064 * 1e-6, 0.7, 150., 10., 1e-4),
        (15, 8, 1500, 1000, 550 * 1e-6, 0.5, 120., 10., 1e-4)
    ]
)
def test_4f_system(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    wavelength_test: float,
    waist_radius_test: float,
    distance_test: float,
    radius_test: float,
    error_energy: float
):
    """Test for phase reconstruction problem using HIO algorithm on the
    example of a 4f optical setup

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
    distance_test : float
        Distance between the lens and the screen
    radius_test : float
        Radius of the thin lens
    error_energy : float
        Criterion for accepting the test(energy loss)
    """
    x_linear = torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes)
    y_linear = torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes)
    x_grid, y_grid = torch.meshgrid(x_linear, y_linear, indexing='xy')

    dx = ox_size / ox_nodes
    dy = oy_size / oy_nodes

    params = SimulationParameters(
        {
            'W': torch.linspace(-ox_size/2, ox_size/2, ox_nodes),
            'H': torch.linspace(-oy_size/2, oy_size/2, oy_nodes),
            'wavelength': wavelength_test
        }
    )

    field_before_lens1 = Wavefront.gaussian_beam(
        simulation_parameters=params,
        distance=distance_test,
        waist_radius=waist_radius_test
    )

    intensity_source = field_before_lens1.intensity.detach().numpy()

    lens1 = elements.ThinLens(
        simulation_parameters=params,
        focal_length=distance_test,
        radius=radius_test
    )

    field_after_lens1 = lens1.forward(input_field=field_before_lens1)

    free_space1 = elements.FreeSpace(
        simulation_parameters=params,
        distance=torch.tensor(2 * distance_test),
        method='AS'
    )

    field_before_lens2 = free_space1.forward(input_field=field_after_lens1)

    lens2 = elements.ThinLens(
        simulation_parameters=params,
        focal_length=distance_test,
        radius=radius_test
    )

    field_after_lens2 = lens2.forward(input_field=field_before_lens2)

    free_space2 = elements.FreeSpace(
        simulation_parameters=params,
        distance=torch.tensor(distance_test),
        method='AS'
    )

    output_field = free_space2.forward(input_field=field_after_lens2)

    phase_target = (
        torch.angle(output_field) + 2 * torch.pi * (
            torch.angle(output_field) < 0.
        ).float()
    ).detach().numpy()

    intensity_target = output_field.intensity.detach().numpy()

    optical_setup = LinearOpticalSetup([free_space1, lens2, free_space2])

    goal = (x_grid**2 + y_grid ** 2 <= 2).float()

    result_hio = phase_retrieval.retrieve_phase(
        source_intensity=torch.tensor(intensity_source),
        optical_setup=optical_setup,
        target_intensity=torch.tensor(intensity_target),
        target_phase=torch.tensor(phase_target),
        target_region=goal,
        initial_phase=None,
        method='HIO',
    )

    phase_reconstruction_hio = result_hio.solution

    step = 2 * torch.pi / 256
    mask_reconstruction_hio = phase_reconstruction_hio // step

    field_after_slm = elements.SpatialLightModulator(
        simulation_parameters=params,
        mask=mask_reconstruction_hio
    ).forward(field_before_lens1)

    output_field = optical_setup.forward(field_after_slm)
    intensity_target_opt = torch.pow(
        torch.abs(output_field), 2
    ).detach().numpy()

    energy_reconstruction_hio = np.sum(intensity_target_opt) * dx * dy
    energy_true = np.sum(intensity_target) * dx * dy

    assert np.abs(energy_true - energy_reconstruction_hio) <= error_energy
