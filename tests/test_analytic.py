import pytest
import torch
import numpy as np

from svetlanna import elements
from svetlanna import SimulationParameters
from svetlanna import Wavefront
from svetlanna.units import ureg

import analytical_solutions as anso


square_parameters = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "wavelength_test",
    "distance_test",
    "width_test",
    "height_test",
    "methods",
    "expected_error",
    "error_energy",
]


@pytest.mark.parametrize(
    square_parameters,
    [
        (
            8 * ureg.mm,  # ox_size, mm
            8 * ureg.mm,  # oy_size, mm
            1200,  # ox_nodes
            1300,  # oy_nodes
            540 * ureg.nm,  # wavelength_test, mm
            60 * ureg.mm,  # distance_test, mm
            3 * ureg.mm,  # width_test, mm
            2 * ureg.mm,  # height_test, mm
            ["ASM", "zpASM", "zpRSC", "RSC"],  # methods to test
            0.075,  # expected intensity error
            0.05,  # error energy
        ),
        (
            10 * ureg.mm,  # ox_size, mm
            10 * ureg.mm,  # oy_size, mm
            1900,  # ox_nodes
            1500,  # oy_nodes
            torch.linspace(490, 660, 5)
            * ureg.nm,  # wavelength_test tensor, mm    # noqa: E501
            100 * ureg.mm,  # distance_test, mm
            3 * ureg.mm,  # width_test, mm
            3 * ureg.mm,  # height_test, mm
            ["ASM", "zpASM", "zpRSC", "RSC"],
            0.065,  # expected intensity error
            0.08,  # error energy
        ),
        (
            8 * ureg.mm,  # ox_size, mm
            8 * ureg.mm,  # oy_size, mm
            1200,  # ox_nodes
            1300,  # oy_nodes
            torch.linspace(440, 660, 5, dtype=torch.float64)
            * ureg.nm,  # wavelength_test tensor, mm
            600 * ureg.mm,  # distance_test, mm
            2 * ureg.mm,  # width_test, mm
            2 * ureg.mm,  # height_test, mm
            ["ASM"],  # methods to test
            0.075,  # expected intensity error
            0.05,  # error energy
        ),
    ],
)
def test_rectangle_fresnel(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    wavelength_test: torch.Tensor | float,
    distance_test: float,
    width_test: float,
    height_test: float,
    methods: list[str],
    expected_error: float,
    error_energy: float,
):
    """Test for the free propagation problem on the example of diffraction of
    the plane wave on the rectangular aperture

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
    wavelength_test : torch.Tensor | float
        Wavelength for the incident field
    distance_test : float
        The distance between square aperture and the screen
    width_test : float
        The width of the square aperture
    height_test : float
        The height of the square aperture
    methods : list[str]
        Propagation methods to test
    expected_error : float
        Criterion for accepting the test
    error_energy : float
        Criterion for accepting the test(energy loss)
    """

    params = SimulationParameters(
        {
            "x": torch.linspace(
                -ox_size / 2, ox_size / 2, ox_nodes, dtype=torch.float64
            ),
            "y": torch.linspace(
                -oy_size / 2, oy_size / 2, oy_nodes, dtype=torch.float64
            ),
            "wavelength": wavelength_test,
        }
    )

    dx = ox_size / ox_nodes
    dy = oy_size / oy_nodes

    # analytical intensity distribution on the screen
    intensity_analytic = anso.RectangleFresnel(
        distance=distance_test,
        x_size=ox_size,
        y_size=oy_size,
        x_nodes=ox_nodes,
        y_nodes=oy_nodes,
        width=width_test,
        height=height_test,
        wavelength=wavelength_test,
    ).intensity()

    if isinstance(intensity_analytic, np.ndarray):
        intensity_analytic = torch.from_numpy(intensity_analytic)

    energy_analytic = torch.sum(intensity_analytic, dim=(-2, -1)) * dx * dy

    incident_field = Wavefront.plane_wave(
        simulation_parameters=params, distance=distance_test, wave_direction=[0, 0, 1]
    )

    # field after the square aperture
    transmission_field = elements.RectangularAperture(
        simulation_parameters=params, height=height_test, width=width_test
    )(incident_field)

    errors_energy = []
    errors = []

    for method in methods:
        output_wf = elements.FreeSpace(
            simulation_parameters=params, distance=distance_test, method=method
        )(transmission_field)

        output_intensity = output_wf.intensity

        energy_numeric = torch.sum(output_intensity, dim=(-2, -1)) * dx * dy

        intensity_difference = torch.abs(intensity_analytic - output_intensity) / (
            ox_nodes * oy_nodes
        )

        error, _ = intensity_difference.view(intensity_difference.size(0), -1).max(
            dim=1
        )

        energy_error = torch.abs((energy_analytic - energy_numeric) / energy_analytic)

        errors.append(error)
        errors_energy.append(energy_error)

    errors = torch.stack(errors)
    errors_energy = torch.stack(errors_energy)

    assert torch.all(errors < expected_error), f"Errors: {errors}"
    assert torch.all(errors_energy < error_energy), f"Energy errors: {errors_energy}"
