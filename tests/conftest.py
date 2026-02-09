import random
import pytest
import torch
from itertools import product
from typing import Generator
from svetlanna import SimulationParameters


def _shuffle_dict(d: dict) -> dict:
    items = list(d.items())
    random.shuffle(items)
    return dict(items)


def sim_params_simple_cases() -> Generator[SimulationParameters, None, None]:
    N_cases = [
        99,
        100,
    ]
    width_cases = [
        1,
        2,
    ]
    wavelengths_cases = [
        torch.tensor(1.0),
        torch.tensor([1.0, 2, 3]),
    ]
    t_cases = [
        torch.tensor(1.0),
        torch.tensor([1.0, 2, 3]),
    ]

    for Nx, Ny in product(N_cases, repeat=2):
        for width_x, width_y in product(width_cases, repeat=2):
            case: dict[str, torch.Tensor | float] = {
                "x": torch.linspace(-width_x / 2, width_x / 2, Nx),
                "y": torch.linspace(-width_y / 2, width_y / 2, Ny),
            }

            for wavelength in wavelengths_cases:
                case["wavelength"] = wavelength

                # no t axes
                yield SimulationParameters(axes=_shuffle_dict(case))

                # with t axes
                for t in t_cases:
                    case["t"] = t
                    yield SimulationParameters(axes=_shuffle_dict(case))
                    del case["t"]


def device_cases() -> list:
    return [
        pytest.param("cpu"),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="cuda is not available",
            ),
        ),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(),
                reason="mps is not available",
            ),
        ),
    ]


def pytest_generate_tests(metafunc):
    if "sim_params_simple" in metafunc.fixturenames:
        metafunc.parametrize("sim_params_simple", list(sim_params_simple_cases()))

    if "device_simple" in metafunc.fixturenames:
        metafunc.parametrize("device_simple", device_cases())
