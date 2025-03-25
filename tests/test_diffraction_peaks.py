import pytest
import torch
import numpy as np

from svetlanna import elements
from svetlanna import SimulationParameters
from svetlanna import Wavefront

from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar


parameters = [
    "ox_size",
    "oy_size",
    "ox_nodes",
    "oy_nodes",
    "wavelength_test",
    "distance",
    "width",
    "diffraction_order",
    "expected_error",
]


@pytest.mark.parametrize(
    parameters,
    [
        (
            500,  # ox_size
            500,  # oy_size
            1000000,   # ox_nodes
            10,   # oy_nodes
            1064 * 1e-6,  # wavelength_test tensor, mm    # noqa: E501
            1500,    # distance, mm
            0.1,    # width, mm
            5,  # max diffraction order to check
            0.02  # expected_error
        ),
        (
            500,  # ox_size
            500,  # oy_size
            1000000,   # ox_nodes
            10,   # oy_nodes
            660 * 1e-6,  # wavelength_test tensor, mm    # noqa: E501
            1500,    # distance, mm
            0.1,    # width, mm
            6,  # max diffraction order to check
            0.02   # expected_error
        ),
        (
            500,  # ox_size
            500,  # oy_size
            1000000,   # ox_nodes
            10,   # oy_nodes
            540 * 1e-6,  # wavelength_test tensor, mm    # noqa: E501
            1500,    # distance, mm
            0.1,    # width, mm
            4,  # max diffraction order to check
            0.02   # expected_error
        ),
        (
            500,  # ox_size
            500,  # oy_size
            1000000,   # ox_nodes
            10,   # oy_nodes
            990 * 1e-6,  # wavelength_test tensor, mm    # noqa: E501
            1500,    # distance, mm
            0.1,    # width, mm
            8,  # max diffraction order to check
            0.02   # expected_error
        ),
    ]
)
def test_diffraction_peaks(
    ox_size: float,
    oy_size: float,
    ox_nodes: int,
    oy_nodes: int,
    wavelength_test: float,
    distance: float,
    width: float,
    diffraction_order: int,
    expected_error: float
):
    """Test checking the coincidence of diffraction maxima at diffraction on a
    thin slit

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
    distance : float
        Distance between the aperture and the slit
    width : float
        Width of the slit
    diffraction_order : int
        Number of diffraction maxima to test
    expected_error : float
        Criterion for accepting the test(relative error for the position of
        maxima)
    """
    height = oy_size

    x_length = torch.linspace(-ox_size / 2, ox_size / 2, ox_nodes)
    y_length = torch.linspace(-oy_size / 2, oy_size / 2, oy_nodes)

    params = SimulationParameters(
        axes={
            'W': x_length,
            'H': y_length,
            'wavelength': wavelength_test
            })

    beam = Wavefront.gaussian_beam(
        simulation_parameters=params,
        waist_radius=2.,
        distance=distance
    )

    # create rectangular aperture
    rectangular_aperture = elements.RectangularAperture(
        simulation_parameters=params,
        height=height,
        width=width
    )

    field_after_aperture = rectangular_aperture(beam)

    fs = elements.FreeSpace(
        simulation_parameters=params, distance=distance, method='AS'
    )

    output_field = fs.forward(field_after_aperture)
    intensity_output = output_field.intensity

    amplitude_1d = np.sqrt(intensity_output.detach().numpy())[int(oy_nodes/2)]

    def intensity_analytic(coordinates: torch.Tensor) -> np.ndarray:
        phi = np.arctan(coordinates / distance)
        u = np.pi / wavelength_test * width * np.sin(phi)
        return (np.sin(u) / u)**2 * intensity_output[int(oy_nodes/2), int(ox_nodes/2)]   # noqa: E501

    def find_maximum(start, end):
        result = minimize_scalar(
            lambda x: -intensity_analytic(x),
            bounds=(start, end),
            method='bounded'
        )
        return result.x

    x_max = []
    for m in range(1, diffraction_order):  # Находим первые 5 максимумов
        start = distance * np.tan(np.arcsin(m * wavelength_test / width))
        end = distance * np.tan(np.arcsin((m + 1) * wavelength_test / width))
        max_x = find_maximum(start, end)
        x_max.append(max_x)

    x_max = torch.tensor(x_max)

    # define Gaussian function
    def gaussian(x, amp, cen, wid):
        return amp * np.exp(-(x-cen)**2 / (2*wid**2))

    x_max_averaged = np.array([])

    peaks = torch.cat((x_max, -x_max), dim=0)
    gaussian_params = []

    for peak in peaks:

        peak_index = np.argmin(np.abs(x_length - peak))
        left = max(0, peak_index - 10000)
        right = min(len(x_length), peak_index + 10000)
        x_peak = x_length[left:right]
        y_peak = amplitude_1d[left:right]

        # Начальные параметры для аппроксимации
        initial_guess = [amplitude_1d[peak_index], x_length[peak_index], 1.0]

        # Аппроксимация
        popt, _ = curve_fit(gaussian, x_peak, y_peak, p0=initial_guess)
        gaussian_params.append(popt)

        x_max_averaged = np.append(x_max_averaged, popt[1])

    error = torch.abs((torch.tensor(x_max_averaged) - peaks) / peaks)

    assert (error <= expected_error).all()
