import pytest
import torch

from svetlanna import elements
from svetlanna import SimulationParameters
from svetlanna import Wavefront as w

parameters = "default_type"


@pytest.mark.parametrize(parameters, [
    torch.float64,
    torch.float32
])
def test_types(default_type: torch.dtype):
    """A test that checks that all elements belong to the same data type

    Parameters
    ----------
    default_type : torch.dtype
        dtype for objects
    """

    torch.set_default_dtype(default_type)

    ox_size = 15.
    oy_size = 8.
    ox_nodes = 1200
    oy_nodes = 1100
    wavelength = torch.linspace(330*1e-6, 660*1e-6, 5)
    waist_radius = 2.
    distance = 100.
    focal_length = 100.
    radius = 10.
    height = 4.
    width = 3.

    if torch.get_default_dtype() == torch.float64:
        default_complex_dtype = torch.complex128
    else:
        default_complex_dtype = torch.complex64

    params = SimulationParameters(
        axes={
            'W': torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes),
            'H': torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes),
            'wavelength': wavelength
            }
    )

    x_linear = params.axes.W
    y_linear = params.axes.H
    wavelength = params.axes.wavelength

    x_grid, y_grid = params.meshgrid(x_axis='W', y_axis='H')

    gaussian_beam = w.gaussian_beam(
        simulation_parameters=params,
        waist_radius=waist_radius,
        distance=distance
    )

    plane_wave = w.plane_wave(
        simulation_parameters=params,
        distance=distance
    )

    spherical_wave = w.spherical_wave(
        simulation_parameters=params,
        distance=distance
    )

    lens = elements.ThinLens(
        simulation_parameters=params,
        focal_length=focal_length,
        radius=radius
    ).get_transmission_function()

    aperture = elements.Aperture(
        simulation_parameters=params,
        mask=torch.zeros(x_grid.shape)
    ).get_transmission_function()

    rectangular_aperture = elements.RectangularAperture(
        simulation_parameters=params,
        height=height,
        width=width
    ).get_transmission_function()

    round_aperture = elements.RoundAperture(
        simulation_parameters=params,
        radius=radius
    ).get_transmission_function()

    slm = elements.SpatialLightModulator(
        simulation_parameters=params,
        mask=torch.ones_like(x_grid)
    ).get_transmission_function()

    layer = elements.DiffractiveLayer(
        simulation_parameters=params,
        mask=torch.zeros(x_grid.shape)
    ).transmission_function

    free_space_as = elements.FreeSpace(
        simulation_parameters=params,
        distance=distance, method='AS'
    ).forward(input_field=gaussian_beam)

    free_space_fresnel = elements.FreeSpace(
        simulation_parameters=params,
        distance=distance, method='fresnel'
    ).forward(input_field=gaussian_beam)

    free_space_reverse = elements.FreeSpace(
        simulation_parameters=params,
        distance=distance, method='fresnel'
    ).reverse(transmission_field=gaussian_beam)

    default_type = torch.get_default_dtype()

    assert x_linear.dtype == torch.get_default_dtype()
    assert y_linear.dtype == torch.get_default_dtype()
    assert x_grid.dtype == torch.get_default_dtype()
    assert y_grid.dtype == torch.get_default_dtype()
    assert wavelength.dtype == torch.get_default_dtype()
    assert gaussian_beam.dtype == default_complex_dtype
    assert plane_wave.dtype == default_complex_dtype
    assert spherical_wave.dtype == default_complex_dtype
    assert lens.dtype == default_complex_dtype
    assert aperture.dtype == torch.get_default_dtype()
    assert rectangular_aperture.dtype == torch.get_default_dtype()
    assert round_aperture.dtype == torch.get_default_dtype()
    assert slm.dtype == default_complex_dtype
    assert layer.dtype == default_complex_dtype
    assert free_space_as.dtype == default_complex_dtype
    assert free_space_fresnel.dtype == default_complex_dtype
    assert free_space_reverse.dtype == default_complex_dtype
