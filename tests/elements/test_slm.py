from typing import Literal
import pytest
import torch

from svetlanna import elements
from svetlanna import SimulationParameters
from svetlanna import Wavefront
from svetlanna.elements.slm import one_step_tanh, one_step_cos
from svetlanna.elements.slm import QuantizerFromStepFunction


def test_slm(sim_params_simple: SimulationParameters):
    """Test SLM transmission with a mask aligned to the simulation grid."""

    mask = torch.rand(sim_params_simple.axes_size(("y", "x")))

    slm = elements.SpatialLightModulator(
        simulation_parameters=sim_params_simple,
        mask=mask,
        width=float(sim_params_simple.x[-1] - sim_params_simple.x[0]),
        height=float(sim_params_simple.y[-1] - sim_params_simple.y[0]),
    )

    transmission_function_expected = sim_params_simple.cast(
        torch.exp(1j * mask), "y", "x"
    )

    torch.testing.assert_close(
        slm.transmission_function, transmission_function_expected
    )

    # Validate forward calculation output.
    wavefront = Wavefront.plane_wave(sim_params_simple)
    torch.testing.assert_close(
        slm(wavefront), transmission_function_expected * wavefront
    )


@pytest.mark.parametrize(
    ["mode", "width", "height", "center", "mask", "phase_expected"],
    [
        (
            "nearest",
            2,
            2,
            (0.0, 0.0),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor(
                [
                    [1.0, 1.0, 2.0, 2.0],
                    [1.0, 1.0, 2.0, 2.0],
                    [3.0, 3.0, 4.0, 4.0],
                    [3.0, 3.0, 4.0, 4.0],
                ]
            ),
        ),
        (
            "nearest",
            2,
            2,
            (0.0, 0.0),
            torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
            torch.tensor(
                [
                    [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                    [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                    [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0, 5.0, 5.0, 5.0],
                    [4.0, 4.0, 4.0, 5.0, 5.0, 5.0],
                    [4.0, 4.0, 4.0, 5.0, 5.0, 5.0],
                ]
            ),
        ),
        (
            "nearest",
            1,
            2,
            (-0.5, 0.0),
            torch.tensor([[1.0]]),
            torch.tensor(
                [
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                ]
            ),
        ),
        (
            "nearest",
            1,
            1,
            (-0.5, 0.0),
            torch.tensor([[1.0]]),
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
        (
            "nearest",
            1,
            1,
            (-0.5, 1.0),
            torch.tensor([[1.0]]),
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                ]
            ),
        ),
        (
            "nearest",
            1,
            1,
            (0.5, -0.5),
            torch.tensor([[1.0]]),
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
        (
            "nearest",
            1,
            1,
            (2, -0.5),
            torch.tensor([[1.0]]),
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
        (
            "nearest",
            2,
            1,
            (0.5, -0.5),
            torch.tensor([[1.0]]),
            torch.tensor(
                [
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
        (
            "nearest",
            1,
            1,
            (0.0, 0.0),
            torch.tensor([[1.0]]),
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
    ],
)
def test_slm_mask_var(
    mode: Literal["nearest", "bilinear", "bicubic", "area", "nearest-exact"],
    width: float,
    height: float,
    center: tuple,
    mask: torch.Tensor,
    phase_expected: torch.Tensor,
):
    Ny, Nx = phase_expected.size()
    sim_params = SimulationParameters(
        {
            "x": torch.linspace(-1, 1, Nx),
            "y": torch.linspace(-1, 1, Ny),
            "wavelength": 1.0,
        }
    )

    slm = elements.SpatialLightModulator(
        simulation_parameters=sim_params,
        mask=mask,
        mode=mode,
        center=center,
        width=width,
        height=height,
    )

    aperture = phase_expected.abs() != 0.0
    transmission_function_expected = aperture * torch.exp(1j * phase_expected)
    transmission_function = slm.transmission_function
    torch.testing.assert_close(transmission_function, transmission_function_expected)


def test_slm_device(device_simple: str):
    """Test SLM on different devices."""

    sim_params = SimulationParameters(
        x=torch.linspace(-10, 10, 20), y=torch.linspace(-10, 10, 20), wavelength=1.0
    )
    wavefront = Wavefront.plane_wave(sim_params).to(device=device_simple)

    sim_params.to(torch.get_default_device())  # TODO: remove
    assert sim_params.device == torch.get_default_device()
    slm = elements.SpatialLightModulator(
        simulation_parameters=sim_params,
        mask=torch.ones(1, 1),
        mode="nearest",
        center=(0, 0),
        width=1.0,
        height=1.0,
    )
    slm.to(device=device_simple)

    assert slm(wavefront).device.type == device_simple


@pytest.mark.parametrize(
    "func",
    [
        one_step_tanh,
        one_step_cos,
    ],
)
@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_one_step_funcs(func, alpha):
    alpha = torch.tensor(alpha)

    # Validate values at boundaries.
    x0 = torch.tensor(0.0)
    x1 = torch.tensor(1.0)
    assert func(x0, alpha=alpha) == 0
    assert func(x1, alpha=alpha) == 1

    # Validate gradients at boundaries.
    x0 = torch.nn.Parameter(x0)
    x1 = torch.nn.Parameter(x1)
    func(x0, alpha=alpha).backward()
    func(x1, alpha=alpha).backward()
    torch.testing.assert_close(x0.grad, x1.grad)


def test_quantizer_from_step_function():
    max_value = 3.0
    quantizer = QuantizerFromStepFunction(
        N=3,
        max_value=max_value,
        one_step_function=lambda x: torch.where(
            x < 0.5, torch.tensor(0.0), torch.tensor(1.0)
        ),
    )

    # Validate values at boundaries.
    x = torch.linspace(0, max_value, 8)
    x[-1] -= 1e-5
    assert torch.sum(quantizer(x) == 0.0) == 2
    assert torch.sum(quantizer(x) == 1.0) == 2
    assert torch.sum(quantizer(x) == 2.0) == 2
    assert torch.sum(quantizer(x) == 3.0) == 2

    # Validate gradients for a non-trivial step function.
    quantizer = QuantizerFromStepFunction(
        N=3,
        max_value=max_value,
        one_step_function=lambda x: one_step_tanh(x, alpha=torch.tensor(10.0)),
    )

    x = torch.nn.Parameter(x)
    torch.sum(quantizer(x)).backward()
    grads = x.grad
    assert grads is not None

    # Gradients should match at the boundaries.
    torch.testing.assert_close(grads[0], grads[-1])
    # Gradients should be non-negative.
    assert torch.all(grads >= 0)
