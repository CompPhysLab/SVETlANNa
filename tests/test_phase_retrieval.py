import pytest
import torch
import numpy as np

from svetlanna import elements
from svetlanna import SimulationParameters
from svetlanna import Wavefront
from svetlanna.phase_retrieval_problem import phase_retrieval
from svetlanna import LinearOpticalSetup


def test_retrieve_phase_api(capsys):
    """
    Tests the retrieve_phase API with different scenarios."""
    params = SimulationParameters(
        {
            "W": torch.linspace(-1, 1, 10),
            "H": torch.linspace(-1, 1, 10),
            "wavelength": 1,
        }
    )
    # no initial_phase, no additional options
    phase_retrieval.retrieve_phase(
        Wavefront.plane_wave(params).abs(),
        LinearOpticalSetup([]),
        Wavefront.plane_wave(params).abs(),
    )

    # Test wrong method name
    with pytest.raises(ValueError):
        phase_retrieval.retrieve_phase(
            Wavefront.plane_wave(params).abs(),
            LinearOpticalSetup([]),
            Wavefront.plane_wave(params).abs(),
            method="abs",  # type: ignore
        )

    # Test disp option for the intensity profile problem type
    phase_retrieval.retrieve_phase(
        Wavefront.plane_wave(params).abs(),
        LinearOpticalSetup([]),
        Wavefront.plane_wave(params).abs(),
        options={"disp": True},
    )
    captured = capsys.readouterr().out.split("\n")[0]
    assert captured == "Type of problem: generate intensity profile"

    # Test disp option for the phase reconstruction problem type
    phase_retrieval.retrieve_phase(
        Wavefront.plane_wave(params).abs(),
        LinearOpticalSetup([]),
        Wavefront.plane_wave(params).abs(),
        target_phase=torch.zeros((10, 10)),
        target_region=torch.zeros((10, 10)),
        options={"disp": True},
    )
    captured = capsys.readouterr().out.split("\n")[0]
    assert captured == "Type of problem: phase reconstruction"


parameters = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "wavelength_test",
    "waist_radius_test",
    "distance_test",
]


@pytest.mark.parametrize(
    parameters,
    [
        (10, 10, 200, 200, 0.025, 0.7, 100.0),
        (7, 8, 200, 200, 0.02, 0.7, 150.0),
        (15, 8, 300, 200, 0.02, 0.5, 120.0),
    ],
)
@pytest.mark.parametrize("use_phase_target", [True, False])
@pytest.mark.parametrize("method", ["HIO", "GS"])
def test_phase_retrieval(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    wavelength_test: float,
    waist_radius_test: float,
    distance_test: float,
    use_phase_target: bool,
    method: phase_retrieval.Method,
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
    waist_radius_test : float
        Waist radius of the Gaussian beam
    distance_test : float
        Distance between the lens and the screen
    radius_test : float
        Radius of the thin lens
    error_energy : float
        Criterion for accepting the test(energy loss)
    """

    torch.set_default_dtype(torch.float32)

    params = SimulationParameters(
        {
            "W": torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes),
            "H": torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes),
            "wavelength": wavelength_test,
        }
    )
    x_grid, y_grid = params.meshgrid("W", "H")

    field_before_lens1 = Wavefront.gaussian_beam(
        simulation_parameters=params,
        distance=0.05 * distance_test,
        waist_radius=waist_radius_test,
    )

    intensity_source = field_before_lens1.intensity

    lens1 = elements.ThinLens(simulation_parameters=params, focal_length=distance_test)

    field_after_lens1 = lens1(field_before_lens1)

    free_space1 = elements.FreeSpace(
        simulation_parameters=params, distance=0.05 * distance_test, method="AS"
    )
    output_field = free_space1(field_after_lens1)

    # target intensity profile on the screen
    intensity_target = output_field.intensity

    optical_setup = LinearOpticalSetup([free_space1])

    # target phase profile for phase reconstruction problem
    if use_phase_target:
        phase_target = torch.angle(output_field)
        target_region = (x_grid**2 + y_grid**2 <= 0.12).float()

        result_hio = phase_retrieval.retrieve_phase(
            source_intensity=intensity_source,
            optical_setup=optical_setup,
            target_intensity=intensity_target,
            target_phase=phase_target,
            target_region=target_region,
            initial_phase=torch.full_like(intensity_target, 0),
            method=method,
            options={"maxiter": 100, "constant_factor": 0.5},
        )
    else:
        result_hio = phase_retrieval.retrieve_phase(
            source_intensity=intensity_source,
            optical_setup=optical_setup,
            target_intensity=intensity_target,
            initial_phase=torch.full_like(intensity_target, 0),
            method=method,
            options={"maxiter": 100, "constant_factor": 0.5},
        )

    errors = result_hio.cost_func_evolution

    # test if the error decreases
    assert np.sum(np.diff(errors) < 0) > 0.7 * (len(errors) - 1)
    assert (errors[0] - errors[-1]) / errors[0] > 0.6


# TODO: Rewrite the 4f problem test
parameters_4f = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "wavelength_test",
    "waist_radius_test",
    "distance_test",
    "radius_test",
    "error_energy",
]


@pytest.mark.skip
@pytest.mark.parametrize(
    parameters_4f,
    [
        (10, 10, 1000, 1000, 660 * 1e-6, 0.5, 100.0, 10.0, 1e-4),
        (7, 8, 1000, 1000, 1064 * 1e-6, 0.7, 150.0, 10.0, 1e-4),
        (15, 8, 1500, 1000, 550 * 1e-6, 0.5, 120.0, 10.0, 1e-4),
    ],
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
    error_energy: float,
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
    x_grid, y_grid = torch.meshgrid(x_linear, y_linear, indexing="xy")

    dx = ox_size / ox_nodes
    dy = oy_size / oy_nodes

    params = SimulationParameters(
        {
            "W": torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes),
            "H": torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes),
            "wavelength": wavelength_test,
        }
    )

    field_before_lens1 = Wavefront.gaussian_beam(
        simulation_parameters=params,
        distance=distance_test,
        waist_radius=waist_radius_test,
    )

    intensity_source = field_before_lens1.intensity.detach().numpy()

    lens1 = elements.ThinLens(
        simulation_parameters=params, focal_length=distance_test, radius=radius_test
    )

    field_after_lens1 = lens1.forward(input_field=field_before_lens1)

    free_space1 = elements.FreeSpace(
        simulation_parameters=params,
        distance=torch.tensor(2 * distance_test),
        method="AS",
    )

    field_before_lens2 = free_space1.forward(input_field=field_after_lens1)

    lens2 = elements.ThinLens(
        simulation_parameters=params, focal_length=distance_test, radius=radius_test
    )

    field_after_lens2 = lens2.forward(input_field=field_before_lens2)

    free_space2 = elements.FreeSpace(
        simulation_parameters=params, distance=torch.tensor(distance_test), method="AS"
    )

    output_field = free_space2.forward(input_field=field_after_lens2)

    phase_target = (
        (
            torch.angle(output_field)
            + 2 * torch.pi * (torch.angle(output_field) < 0.0).float()
        )
        .detach()
        .numpy()
    )

    intensity_target = output_field.intensity.detach().numpy()

    optical_setup = LinearOpticalSetup([free_space1, lens2, free_space2])

    goal = (x_grid**2 + y_grid**2 <= 2).float()

    result_hio = phase_retrieval.retrieve_phase(
        source_intensity=torch.tensor(intensity_source),
        optical_setup=optical_setup,
        target_intensity=torch.tensor(intensity_target),
        target_phase=torch.tensor(phase_target),
        target_region=goal,
        initial_phase=None,
        method="HIO",
    )

    phase_reconstruction_hio = result_hio.solution

    step = 2 * torch.pi / 256
    mask_reconstruction_hio = phase_reconstruction_hio // step

    field_after_slm = elements.SpatialLightModulator(
        simulation_parameters=params, mask=mask_reconstruction_hio
    ).forward(field_before_lens1)

    output_field = optical_setup.forward(field_after_slm)
    intensity_target_opt = torch.pow(torch.abs(output_field), 2).detach().numpy()

    energy_reconstruction_hio = np.sum(intensity_target_opt) * dx * dy
    energy_true = np.sum(intensity_target) * dx * dy

    assert np.abs(energy_true - energy_reconstruction_hio) <= error_energy
