import pytest
import torch

from svetlanna import elements
from svetlanna import SimulationParameters
from svetlanna import Wavefront

from examples import analytical_solutions as anso

torch.set_default_dtype(torch.float64)

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
    [
        (
            3,  # ox_size, mm
            3,  # oy_size, mm
            1000,   # ox_nodes
            1000,   # oy_nodes
            torch.linspace(330 * 1e-6, 660 * 1e-6, 5),  # wavelength_test tensor, mm    # noqa: E501
            150,    # distance_test, mm
            1.5,    # waist radius, mm
            0.065,  # expected std
            0.05    # error_energy
        ),
        (
            4,  # ox_size, mm
            4,  # oy_size, mm
            1200,   # ox_nodes
            1300,   # oy_nodes
            torch.linspace(330 * 1e-6, 660 * 1e-6, 5),  # wavelength_test tensor, mm    # noqa: E501
            600,    # distance_test, mm
            1,  # waist radius, mm
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
    wavelength_test: torch.Tensor,
    distance_test: float,
    square_size_test: float,
    expected_std: float,
    error_energy: float
):
    """Test for the free propagation problem on the example of diffraction of
    the plane wave on the square aperture

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
    distance_test : float
        The distance between square aperture and the screen
    square_size_test : float
        The size of the square aperture
    expected_std : float
        Criterion for accepting the test(standard deviation)
    error_energy : float
        Criterion for accepting the test(energy loss)
    """

    params = SimulationParameters(
        {
            'W': torch.linspace(-ox_size/2, ox_size/2, ox_nodes),
            'H': torch.linspace(-oy_size/2, oy_size/2, oy_nodes),
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
    intensity_output_fresnel = output_field_fresnel.intensity
    # intensity distribution on the screen by using Angular Spectrum method
    intensity_output_as = output_field_as.intensity

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

    intensity_analytic = intensity_analytic.clone().detach()

    energy_analytic = torch.sum(intensity_analytic, dim=(-2, -1)) * dx * dy
    energy_numeric_fresnel = torch.sum(
        intensity_output_fresnel, dim=(-2, -1)
    ) * dx * dy
    energy_numeric_as = torch.sum(intensity_output_as, dim=(-2, -1)) * dx * dy

    standard_deviation_fresnel = torch.std(
        intensity_analytic - intensity_output_fresnel, dim=(-2, -1)
    )
    standard_deviation_as = torch.std(
        intensity_analytic - intensity_output_as, dim=(-2, -1)
    )

    energy_error_fresnel = torch.abs(
        (energy_analytic - energy_numeric_fresnel) / energy_analytic
    )
    energy_error_as = torch.abs(
        (energy_analytic - energy_numeric_as) / energy_analytic
    )

    assert (standard_deviation_fresnel <= expected_std).all()
    assert (standard_deviation_as <= expected_std).all()
    assert (energy_error_fresnel <= error_energy).all()
    assert (energy_error_as <= error_energy).all()