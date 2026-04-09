import pytest
import torch

from svetlanna import elements
from svetlanna import SimulationParameters
from svetlanna import Wavefront
from svetlanna.units import ureg
from typing import List


parameters = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "wavelength_test",
    "waist_radius_test",
    "distance_total",
    "distance_end",
    "methods",
    "expected_error",
    "error_energy",
]


# TODO: fix docstrings
@pytest.mark.parametrize(
    parameters,
    [
        (
            6 * ureg.mm,  # ox_size
            6 * ureg.mm,  # oy_size
            1500,  # ox_nodes
            1600,  # oy_nodes
            torch.linspace(
                330 * ureg.nm, 660 * ureg.nm, 5
            ),  # wavelength_test tensor, mm    # noqa: E501
            2.0 * ureg.mm,  # waist_radius_test, mm
            300 * ureg.mm,  # distance_total, mm
            200 * ureg.mm,  # distance_end, mm
            ["ASM", "zpASM"],
            0.02,  # expected intensity error
            1.0,  # error energy, %
        ),
        (
            6 * ureg.mm,  # ox_size
            6 * ureg.mm,  # oy_size
            1500,  # ox_nodes
            1600,  # oy_nodes
            660 * ureg.nm,  # wavelength_test, mm
            2.0 * ureg.mm,  # waist_radius_test, mm
            300 * ureg.mm,  # distance_total, mm
            200 * ureg.mm,  # distance_end, mm
            ["ASM", "zpASM"],
            0.02,  # expected intensity error
            1.0,  # error energy, %
        ),
    ],
)
def test_gaussian_beam_propagation(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    wavelength_test: torch.Tensor | float,
    waist_radius_test: float,
    distance_total: float,
    distance_end: float,
    methods: List,
    expected_error: float,
    error_energy: float,
):
    """Test for the free field propagation problem: free propagation of the
    Gaussian beam at the arbitrary distance(distance_total). We calculate the
    field at the distance_total by using analytical expression and calculate
    the field at the distance_total by splitting on two FreeSpace exemplars(
    distance_total - distance_end + distance_end)

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
    wavelength_test : torch.Tensor
        Wavelength for the incident field
    waist_radius_test : float
        Waist radius of the Gaussian beam
    distance_total : float
        Total propagation distance of the Gaussian beam
    distance_end : float
        Propagation distance of the Gaussian beam which calculates by using
        Fresnel propagation method or angular spectrum method
    expected_error : float
        Criterion for accepting the test
    error_energy : float
        Criterion for accepting the test(energy loss by propagation)
    """

    x_linear = torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes)
    y_linear = torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes)
    x_grid, y_grid = torch.meshgrid(x_linear, y_linear, indexing="xy")

    # creating meshgrid
    x_grid = x_grid[None, :]
    y_grid = y_grid[None, :]

    amplitude = 1.0

    dx = ox_size / ox_nodes
    dy = oy_size / oy_nodes

    if not isinstance(wavelength_test, torch.Tensor):
        wave_number = 2 * torch.pi / wavelength_test
        rayleigh_range = torch.pi * (waist_radius_test**2) / wavelength_test
    else:
        rayleigh_range = (
            torch.pi * (waist_radius_test**2) / wavelength_test[..., None, None]
        )  # noqa: E501
        wave_number = 2 * torch.pi / wavelength_test[..., None, None]

    radial_distance_squared = torch.pow(x_grid, 2) + torch.pow(y_grid, 2)

    hyperbolic_relation = waist_radius_test * (
        1 + (distance_total / rayleigh_range) ** 2
    ) ** (1 / 2)

    radius_of_curvature = distance_total * (1 + (rayleigh_range / distance_total) ** 2)

    # Gouy phase
    gouy_phase = torch.arctan(torch.tensor(distance_total) / rayleigh_range)

    # analytical equation for the propagation of the Gaussian beam
    field = (
        amplitude
        * (waist_radius_test / hyperbolic_relation)
        * (
            torch.exp(-radial_distance_squared / (hyperbolic_relation) ** 2)
            * (
                torch.exp(
                    -1j
                    * (
                        wave_number * distance_total
                        + wave_number
                        * (radial_distance_squared)
                        / (2 * radius_of_curvature)
                        - (gouy_phase)
                    )
                )
            )
        )
    )

    intensity_analytic = torch.pow(torch.abs(field), 2)

    energy_analytic = torch.sum(intensity_analytic, dim=(-2, -1)) * dx * dy

    params = SimulationParameters(
        {
            "x": torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes),
            "y": torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes),
            "wavelength": wavelength_test,
        }
    )

    distance_start = distance_total - distance_end

    field_gb_start = Wavefront.gaussian_beam(
        simulation_parameters=params,
        distance=distance_start,
        waist_radius=waist_radius_test,
    )

    errors_energy = []
    errors = []

    for method in methods:

        wavefront_end = elements.FreeSpace(
            simulation_parameters=params, distance=distance_end, method=method
        )(field_gb_start)

        intensity_end = wavefront_end.intensity

        energy_numeric = torch.sum(intensity_end, dim=(-2, -1)) * dx * dy

        intensity_difference = torch.abs(intensity_analytic - intensity_end) / (
            ox_nodes * oy_nodes
        )

        error, _ = intensity_difference.view(intensity_difference.size(0), -1).max(
            dim=1
        )

        energy_error = (
            torch.abs((energy_analytic - energy_numeric) / energy_analytic) * 100
        )

        errors.append(error)
        errors_energy.append(energy_error)

    for i in range(len(errors)):

        current_errors_energy = errors_energy[i]
        current_errors = errors[i]

        assert (current_errors <= expected_error).all()
        assert (current_errors_energy <= error_energy).all()


parameters = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "wavelength_test",
    "waist_radius_test",
    "distance",
    "methods",
    "expected_error",
]


@pytest.mark.parametrize(
    parameters,
    [
        (
            6 * ureg.mm,  # ox_size
            6 * ureg.mm,  # oy_size
            1569,  # ox_nodes
            1698,  # oy_nodes
            660 * ureg.nm,  # wavelength_test tensor, mm    # noqa: E501
            1.5 * ureg.mm,  # waist_radius_test, mm
            200 * ureg.mm,  # distance, mm
            ["ASM", "zpASM", "zpRSC", "RSC"],  # methods to test
            2.0,  # expected relative error, %
        ),
        (
            15 * ureg.mm,  # ox_size
            8 * ureg.mm,  # oy_size
            1111,  # ox_nodes
            14070,  # oy_nodes
            330 * ureg.nm,  # wavelength_test tensor, mm    # noqa: E501
            1.0 * ureg.mm,  # waist_radius_test, mm
            50 * ureg.mm,  # distance, mm
            ["ASM", "zpASM"],  # methods to test
            1.7,  # expected relative error, %
        ),
        (
            20 * ureg.mm,  # ox_size
            23 * ureg.mm,  # oy_size
            1800,  # ox_nodes
            1032,  # oy_nodes
            540 * ureg.nm,  # wavelength_test tensor, mm    # noqa: E501
            3.0 * ureg.mm,  # waist_radius_test, mm
            500 * ureg.mm,  # distance, mm
            ["ASM", "zpASM", "zpRSC"],  # methods to test
            5.0,  # expected relative error, %
        ),
    ],
)
def test_gaussian_beam_fwhm(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    wavelength_test: torch.Tensor | float,
    waist_radius_test: float,
    distance: float,
    methods: List,
    expected_error: float,
):

    params = SimulationParameters(
        {
            "x": torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes),
            "y": torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes),
            "wavelength": wavelength_test,
        }
    )

    field_gb_start = Wavefront.gaussian_beam(
        simulation_parameters=params, distance=0.0, waist_radius=waist_radius_test
    )

    fwhm_analytical = (
        torch.sqrt(2.0 * torch.log(torch.tensor([2.0])))
        * waist_radius_test
        * torch.sqrt(
            torch.tensor([1.0])
            + (distance / (torch.pi * waist_radius_test**2 / wavelength_test)) ** 2
        )
    )

    errors_x = []
    errors_y = []

    for method in methods:

        output_wavefront = elements.FreeSpace(
            simulation_parameters=params, distance=distance, method=method
        )(field_gb_start)

        fwhm_x, fwhm_y = output_wavefront.fwhm(simulation_parameters=params)

        relative_error_x = torch.abs(fwhm_x - fwhm_analytical) / fwhm_analytical * 100
        relative_error_y = torch.abs(fwhm_y - fwhm_analytical) / fwhm_analytical * 100

        errors_x.append(relative_error_x)
        errors_y.append(relative_error_y)

    for i in range(len(methods)):
        current_error_x = errors_x[i]
        current_error_y = errors_y[i]

        assert (current_error_x <= expected_error).all()
        assert (current_error_y <= expected_error).all()
