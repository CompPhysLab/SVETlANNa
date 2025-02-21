import pytest
import torch

from svetlanna import elements
from svetlanna import SimulationParameters
from svetlanna import Wavefront as w
from svetlanna import LinearOpticalSetup
from svetlanna import detector


parameters = "device_type"


@pytest.mark.parametrize(parameters, [
    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    torch.device("cpu")
])
def test_devices(device_type: torch.device):
    """A test that checks that all elements belong to the same device

    Parameters
    ----------
    device_type : torch.device
        device for objects
    """

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

    tensors = []

    params = SimulationParameters(
        axes={
            'W': torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes).to(device_type),   # noqa: E501
            'H': torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes).to(device_type),   # noqa: E501
            'wavelength': wavelength.to(device_type)
            }
    ).to(device=device_type)

    x_linear = params.axes.W
    y_linear = params.axes.H
    tensors.append(x_linear)
    tensors.append(y_linear)
    wavelength = params.axes.wavelength
    tensors.append(wavelength)

    x_grid, y_grid = params.meshgrid(x_axis='W', y_axis='H')
    tensors.extend([x_grid, y_grid])

    gaussian_beam = w.gaussian_beam(
        simulation_parameters=params,
        waist_radius=waist_radius,
        distance=distance
    )
    tensors.append(gaussian_beam)

    plane_wave = w.plane_wave(
        simulation_parameters=params,
        distance=distance
    )

    tensors.append(plane_wave)

    spherical_wave = w.spherical_wave(
        simulation_parameters=params,
        distance=distance
    )

    tensors.append(spherical_wave)

    lens = elements.ThinLens(
        simulation_parameters=params,
        focal_length=focal_length,
        radius=radius
    )

    tensors.append(lens.get_transmission_function())
    tensors.append(lens.forward(gaussian_beam))
    tensors.append(lens.reverse(gaussian_beam))

    aperture = elements.Aperture(
        simulation_parameters=params,
        mask=torch.zeros(x_grid.shape).to(device_type)
    )

    tensors.append(aperture.get_transmission_function())
    tensors.append(aperture.forward(gaussian_beam))

    rectangular_aperture = elements.RectangularAperture(
        simulation_parameters=params,
        height=height,
        width=width
    )

    tensors.append(rectangular_aperture.get_transmission_function())
    tensors.append(rectangular_aperture.forward(gaussian_beam))

    round_aperture = elements.RoundAperture(
        simulation_parameters=params,
        radius=radius
    )

    tensors.append(round_aperture.get_transmission_function())
    tensors.append(round_aperture.forward(gaussian_beam))

    slm = elements.SpatialLightModulator(
        simulation_parameters=params,
        height=height,
        width=width,
        mask=torch.ones_like(x_grid)
    )

    tensors.append(slm.transmission_function)
    tensors.append(slm.forward(gaussian_beam))
    tensors.append(slm.reverse(gaussian_beam))

    layer = elements.DiffractiveLayer(
        simulation_parameters=params,
        mask=torch.zeros_like(x_grid)
    )

    tensors.append(layer.transmission_function)
    tensors.append(layer.forward(gaussian_beam))
    tensors.append(layer.reverse(gaussian_beam))

    free_space_as = elements.FreeSpace(
        simulation_parameters=params,
        distance=distance, method='AS'
    )

    tensors.append(free_space_as.forward(gaussian_beam))

    free_space_fresnel = elements.FreeSpace(
        simulation_parameters=params,
        distance=distance, method='fresnel'
    )

    tensors.append(free_space_fresnel.forward(gaussian_beam))

    free_space_reverse = elements.FreeSpace(
        simulation_parameters=params,
        distance=distance, method='fresnel'
    )

    tensors.append(free_space_reverse.reverse(gaussian_beam))

    nl = elements.NonlinearElement(
        simulation_parameters=params,
        response_function=lambda x: x**2
    )

    tensors.append(nl.forward(gaussian_beam))

    assert all(tensor.device.type == device_type.type for tensor in tensors)


@pytest.mark.parametrize(parameters, [
    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    torch.device("cpu")
])
def test_device_setup(device_type: torch.device):

    ox_size = 15.
    oy_size = 8.
    ox_nodes = 1200
    oy_nodes = 1100
    wavelength = torch.linspace(330*1e-6, 660*1e-6, 5)
    # waist_radius = 2.
    distance = 50.
    focal_length = 100.
    radius = 10.
    height = 4.
    width = 3.

    params = SimulationParameters(
        axes={
            'W': torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes),
            'H': torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes),
            'wavelength': wavelength
            }
    )

    x_grid, _ = params.meshgrid(x_axis='W', y_axis='H')

    # gaussian_beam = w.gaussian_beam(
    #     simulation_parameters=params,
    #     waist_radius=waist_radius,
    #     distance=distance
    # )

    free_space = elements.FreeSpace(
        simulation_parameters=params,
        distance=distance,
        method="AS"
    )

    circle = elements.RoundAperture(
        simulation_parameters=params,
        radius=radius
    )

    rectangle = elements.RectangularAperture(
        simulation_parameters=params,
        height=height,
        width=width
    )

    aperture = elements.Aperture(
        simulation_parameters=params,
        mask=torch.ones_like(x_grid)
    )

    lens = elements.ThinLens(
        simulation_parameters=params,
        focal_length=distance,
        radius=focal_length
    )

    slm = elements.SpatialLightModulator(
        simulation_parameters=params,
        mask=torch.tensor([[1., 1.], [1., 1.]]),
        height=height,
        width=width
    )

    nl = elements.NonlinearElement(
        simulation_parameters=params,
        response_function=lambda x: x**2
    )

    dl = elements.DiffractiveLayer(
        simulation_parameters=params,
        mask=torch.zeros_like(x_grid)
    )

    det = detector.Detector(simulation_parameters=params)

    optical_setup = LinearOpticalSetup(
        [
            circle,
            free_space,
            rectangle,
            free_space,
            aperture,
            free_space,
            lens,
            free_space,
            slm,
            free_space,
            nl,
            free_space,
            dl,
            free_space,
            det

        ]
    )

    optical_setup.net.to(device_type)
    params.to(device_type)
    # output_field = optical_setup.net.forward(
    #     input_wavefront=gaussian_beam.to(device_type)
    # )

    for param in optical_setup.net.parameters():
        assert param.device == device_type

    # assert optical_setup.net.device.type == device_type.type
