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
    "expected_error",
    "error_energy"
]


# TODO: fix docstrings
@pytest.mark.parametrize(
    parameters,
    [
        (
            6,  # ox_size
            6,  # oy_size
            1500,   # ox_nodes
            1600,   # oy_nodes
            torch.linspace(330*1e-6, 660*1e-6, 5),  # wavelength_test tensor, mm    # noqa: E501
            2.,     # waist_radius_test, mm
            300,    # distance_total, mm
            200,    # distance_end, mm
            0.02,   # expected_std
            0.01    # error_energy
        ),
        (
            6,  # ox_size
            6,  # oy_size
            1500,   # ox_nodes
            1600,   # oy_nodes
            660 * 1e-6,  # wavelength_test, mm
            2.,     # waist_radius_test, mm
            300,    # distance_total, mm
            200,    # distance_end, mm
            0.02,   # expected_std
            0.01    # error_energy
        )
    ]
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
    expected_error: float,
    error_energy: float
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
    x_grid, y_grid = torch.meshgrid(x_linear, y_linear, indexing='xy')

    # creating meshgrid
    x_grid = x_grid[None, :]
    y_grid = y_grid[None, :]

    # wave_number = 2 * torch.pi / wavelength_test[..., None, None]

    amplitude = 1.

    dx = ox_size / ox_nodes
    dy = oy_size / oy_nodes

    if not isinstance(wavelength_test, torch.Tensor):
        wave_number = 2 * torch.pi / wavelength_test
        rayleigh_range = torch.pi * (waist_radius_test**2) / wavelength_test
    else:
        rayleigh_range = torch.pi * (waist_radius_test**2) / wavelength_test[..., None, None]   # noqa: E501
        wave_number = 2 * torch.pi / wavelength_test[..., None, None]

    radial_distance_squared = torch.pow(x_grid, 2) + torch.pow(y_grid, 2)

    hyperbolic_relation = waist_radius_test * (1 + (
        distance_total / rayleigh_range)**2)**(1/2)

    radius_of_curvature = distance_total * (
        1 + (rayleigh_range / distance_total)**2
    )

    # Gouy phase
    gouy_phase = torch.arctan(torch.tensor(distance_total) / rayleigh_range)

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
    )(field_gb_start)
    # field on the screen by using angular spectrum method
    field_end_as = elements.FreeSpace(
        simulation_parameters=params, distance=distance_end, method='AS'
    )(field_gb_start)

    intensity_output_fresnel = field_end_fresnel.intensity
    intensity_output_as = field_end_as.intensity

    energy_analytic = torch.sum(
        intensity_analytic, dim=(-2, -1)
    ) * dx * dy
    energy_numeric_fresnel = torch.sum(
        intensity_output_fresnel, dim=(-2, -1)
    ) * dx * dy
    energy_numeric_as = torch.sum(
        intensity_output_as, dim=(-2, -1)
    ) * dx * dy

    intensity_difference_fresnel = torch.abs(
        intensity_analytic - intensity_output_fresnel
    ) / (ox_nodes * oy_nodes)

    intensity_difference_as = torch.abs(
        intensity_analytic - intensity_output_as
    ) / (ox_nodes * oy_nodes)

    error_fresnel, _ = intensity_difference_fresnel.view(
        intensity_difference_fresnel.size(0), -1
    ).max(dim=1)

    error_as, _ = intensity_difference_as.view(
        intensity_difference_as.size(0), -1
    ).max(dim=1)

    energy_error_fresnel = torch.abs(
        (energy_analytic - energy_numeric_fresnel) / energy_analytic
    )
    energy_error_as = torch.abs(
        (energy_analytic - energy_numeric_as) / energy_analytic
    )

    assert (error_fresnel <= expected_error).all()
    assert (error_as <= expected_error).all()
    assert (energy_error_fresnel <= error_energy).all()
    assert (energy_error_as <= error_energy).all()


parameters = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "wavelength_test",
    "waist_radius_test",
    "distance",
    "expected_error"
]


@pytest.mark.parametrize(
    parameters,
    [
        (
            6,  # ox_size
            6,  # oy_size
            1569,   # ox_nodes
            1698,   # oy_nodes
            660 * 1e-6,  # wavelength_test tensor, mm    # noqa: E501
            2.,     # waist_radius_test, mm
            300,    # distance, mm
            0.5  # expected relative error
        ),

        (
            15,  # ox_size
            8,  # oy_size
            1111,   # ox_nodes
            14070,   # oy_nodes
            330 * 1e-6,  # wavelength_test tensor, mm    # noqa: E501
            1.,     # waist_radius_test, mm
            50,    # distance, mm
            1.7  # expected relative error
        ),

        (
            20,  # ox_size
            23,  # oy_size
            1800,   # ox_nodes
            1032,   # oy_nodes
            540 * 1e-6,  # wavelength_test tensor, mm    # noqa: E501
            4.,     # waist_radius_test, mm
            500,    # distance, mm
            0.5  # expected relative error
        ),
    ]
)
def test_gaussian_beam_fwhm(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    wavelength_test: torch.Tensor | float,
    waist_radius_test: float,
    distance: float,
    expected_error: float,
):

    params = SimulationParameters(
        {
            'W': torch.linspace(-ox_size/2, ox_size/2, ox_nodes),
            'H': torch.linspace(-oy_size/2, oy_size/2, oy_nodes),
            'wavelength': wavelength_test
        }
    )

    field_gb_start = Wavefront.gaussian_beam(
        simulation_parameters=params,
        distance=0.,
        waist_radius=waist_radius_test
    )

    # field on the screen by using Fresnel propagation method
    field_end_fresnel = elements.FreeSpace(
        simulation_parameters=params, distance=distance, method='fresnel'
    )(field_gb_start)
    # field on the screen by using angular spectrum method
    field_end_as = elements.FreeSpace(
        simulation_parameters=params, distance=distance, method='AS'
    )(field_gb_start)

    fwhm_x_as, fwhm_y_as = field_end_as.fwhm(simulation_parameters=params)
    fwhm_x_fresnel, fwhm_y_fresnel = field_end_fresnel.fwhm(
        simulation_parameters=params
    )

    fwhm_analytical = torch.sqrt(
        2. * torch.log(torch.tensor([2.]))
    ) * waist_radius_test * torch.sqrt(
        torch.tensor([1.]) + (
            distance / (torch.pi * waist_radius_test**2 / wavelength_test)
        )**2
    )

    relative_error_x_as = torch.abs(
        fwhm_x_as - fwhm_analytical
    ) / fwhm_analytical * 100
    relative_error_y_as = torch.abs(
        fwhm_y_as - fwhm_analytical
    ) / fwhm_analytical * 100

    relative_error_x_fresnel = torch.abs(
        fwhm_x_fresnel - fwhm_analytical
    ) / fwhm_analytical * 100
    relative_error_y_fresnel = torch.abs(
        fwhm_y_fresnel - fwhm_analytical
    ) / fwhm_analytical * 100

    assert (relative_error_x_as <= expected_error).all()
    assert (relative_error_y_as <= expected_error).all()
    assert (relative_error_x_fresnel <= expected_error).all()
    assert (relative_error_y_fresnel <= expected_error).all()


parameters = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "wavelength_test",
    "waist_radius_test",
    "distance",
    "expected_error"
]


@pytest.mark.parametrize(
    parameters,
    [
        (
            6,  # ox_size
            6,  # oy_size
            1569,   # ox_nodes
            1698,   # oy_nodes
            660 * 1e-6,  # wavelength_test tensor, mm    # noqa: E501
            2.,     # waist_radius_test, mm
            300,    # distance, mm
            0.5  # expected relative error
        ),

        (
            15,  # ox_size
            8,  # oy_size
            1111,   # ox_nodes
            14070,   # oy_nodes
            330 * 1e-6,  # wavelength_test tensor, mm    # noqa: E501
            1.,     # waist_radius_test, mm
            50,    # distance, mm
            1.7  # expected relative error
        ),

        (
            20,  # ox_size
            23,  # oy_size
            1800,   # ox_nodes
            1032,   # oy_nodes
            540 * 1e-6,  # wavelength_test tensor, mm    # noqa: E501
            4.,     # waist_radius_test, mm
            500,    # distance, mm
            0.5  # expected relative error
        ),
    ]
)
def test_gaussian_beam_phase_profile(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    wavelength_test: torch.Tensor | float,
    waist_radius_test: float,
    distance: float,
    expected_error: float,
):

    params = SimulationParameters(
        {
            'W': torch.linspace(-ox_size/2, ox_size/2, ox_nodes),
            'H': torch.linspace(-oy_size/2, oy_size/2, oy_nodes),
            'wavelength': wavelength_test
        }
    )

    field_gb_start = Wavefront.gaussian_beam(
        simulation_parameters=params,
        distance=0.,
        waist_radius=waist_radius_test
    )

    # field on the screen by using Fresnel propagation method
    field_end_fresnel = elements.FreeSpace(
        simulation_parameters=params, distance=distance, method='fresnel'
    )(field_gb_start)
    # field on the screen by using angular spectrum method
    field_end_as = elements.FreeSpace(
        simulation_parameters=params, distance=distance, method='AS'
    )(field_gb_start)

    total_field = Wavefront.gaussian_beam(
        simulation_parameters=params,
        waist_radius=waist_radius_test,
        distance=distance
    )

    intensity_analytic = total_field.intensity

    # highlight target region
    target_region = intensity_analytic >= torch.max(intensity_analytic) / 5

    output_phase_as = field_end_as.phase * target_region
    output_phase_fresnel = field_end_fresnel.phase * target_region
    output_phase_analytical = total_field.phase * target_region

    assert torch.std(
        output_phase_as - output_phase_analytical
    ) <= expected_error
    assert torch.std(
        output_phase_fresnel - output_phase_analytical
    ) <= expected_error
