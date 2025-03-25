import pytest
import torch
import numpy as np

from svetlanna import elements
from svetlanna import SimulationParameters
from svetlanna import Wavefront

from examples import analytical_solutions as anso


square_parameters = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "wavelength_test",
    "distance_test",
    "width_test",
    "height_test",
    "expected_error",
    "error_energy"
]


@pytest.mark.parametrize(
    square_parameters,
    [
        (
            8,  # ox_size, mm
            8,  # oy_size, mm
            1200,   # ox_nodes
            1300,   # oy_nodes
            540 * 1e-6,  # wavelength_test, mm
            600,    # distance_test, mm
            4,  # width_test, mm
            2,  # height_test, mm
            0.075,  # expected error
            0.05    # error_energy
        ),
        (
            10,  # ox_size, mm
            10,  # oy_size, mm
            1400,   # ox_nodes
            1300,   # oy_nodes
            torch.linspace(330 * 1e-6, 660 * 1e-6, 5),  # wavelength_test tensor, mm    # noqa: E501
            150,    # distance_test, mm
            3,    # width_test, mm
            3,    # height_test, mm
            0.065,  # expected error
            0.05    # error_energy
        ),
        (
            8,  # ox_size, mm
            8,  # oy_size, mm
            1200,   # ox_nodes
            1300,   # oy_nodes
            torch.linspace(330 * 1e-6, 660 * 1e-6, 5, dtype=torch.float64),  # wavelength_test tensor, mm    # noqa: E501
            600,    # distance_test, mm
            2,  # width_test, mm
            2,  # height_test, mm
            0.075,  # expected error
            0.05    # error_energy
        ),
        (
            8,  # ox_size, mm
            8,  # oy_size, mm
            1200,   # ox_nodes
            1300,   # oy_nodes
            torch.linspace(330 * 1e-6, 660 * 1e-6, 5, dtype=torch.float64),  # wavelength_test tensor, mm    # noqa: E501
            600,    # distance_test, mm
            4,  # width_test, mm
            2,  # height_test, mm
            0.075,  # expected std
            0.05    # error_energy
        )
    ]
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
    expected_error: float,
    error_energy: float
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
    expected_error : float
        Criterion for accepting the test
    error_energy : float
        Criterion for accepting the test(energy loss)
    """

    params = SimulationParameters(
        {
            'W': torch.linspace(
                -ox_size/2, ox_size/2, ox_nodes, dtype=torch.float64
            ),
            'H': torch.linspace(
                -oy_size/2, oy_size/2, oy_nodes, dtype=torch.float64
            ),
            'wavelength': wavelength_test
        }
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
        height=height_test,
        width=width_test
    )(incident_field)

    # field on the screen by using Fresnel propagation method
    output_field_fresnel = elements.FreeSpace(
        simulation_parameters=params,
        distance=distance_test,
        method='fresnel'
        )(transmission_field)
    # field on the screen by using Angular Spectrum method
    output_field_as = elements.FreeSpace(
        simulation_parameters=params,
        distance=distance_test,
        method='AS'
        )(transmission_field)

    # intensity distribution on the screen by using Fresnel propagation method
    intensity_output_fresnel = output_field_fresnel.intensity
    # intensity distribution on the screen by using Angular Spectrum method
    intensity_output_as = output_field_as.intensity

    # analytical intensity distribution on the screen
    intensity_analytic = anso.RectangleFresnel(
        distance=distance_test,
        x_size=ox_size,
        y_size=oy_size,
        x_nodes=ox_nodes,
        y_nodes=oy_nodes,
        width=width_test,
        height=height_test,
        wavelength=wavelength_test
    ).intensity()

    if isinstance(intensity_analytic, np.ndarray):
        intensity_analytic = torch.from_numpy(intensity_analytic)

    energy_analytic = torch.sum(intensity_analytic, dim=(-2, -1)) * dx * dy
    energy_numeric_fresnel = torch.sum(
        intensity_output_fresnel, dim=(-2, -1)
    ) * dx * dy
    energy_numeric_as = torch.sum(intensity_output_as, dim=(-2, -1)) * dx * dy

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

    assert (error_fresnel < expected_error).all()
    assert (error_as < expected_error).all()
    assert (energy_error_fresnel < error_energy).all()
    assert (energy_error_as < error_energy).all()
