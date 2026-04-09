from svetlanna import Wavefront, SimulationParameters
import torch
import pytest

import sympy as sp  # type: ignore


def _gaussian_beam_symbolic():
    x, y, w0, wavelength, z, dx, dy = sp.symbols("x y w0 wavelength z dx dy", real=True)

    k = 2 * sp.pi / wavelength
    zR = sp.pi * w0**2 / wavelength

    w = w0 * sp.sqrt(1 + (z / zR) ** 2)
    R = z * (1 + (zR / z) ** 2)
    zeta = sp.atan(z / zR)

    r2 = (x - dx) ** 2 + (y - dy) ** 2

    E = (w0 / w) * sp.exp(-r2 / w**2) * sp.exp(sp.I * (k * z + k * r2 / (2 * R) - zeta))
    E = sp.simplify(E)

    return sp.lambdify((x, y, w0, wavelength, z, dx, dy), E, "numpy")


gaussian_beam_analytical = _gaussian_beam_symbolic()


@pytest.mark.parametrize("distance", (1, 10, 100))
@pytest.mark.parametrize("waist_radius", (1, 1.23, 1e-4, 1e4))
@pytest.mark.parametrize("dx", (1.0, 123, 2e-4))
@pytest.mark.parametrize("dy", (1.0, 123, 2e-4))
@pytest.mark.parametrize("wavelength", (1.0, torch.tensor([1.23, 10.0, 40.0])))
def test_gaussian_beam(distance, waist_radius, dx, dy, wavelength):
    sim_params = SimulationParameters(
        {
            "x": torch.linspace(-0.1, 2, 10),
            "y": torch.linspace(-1, 5, 20),
            "wavelength": wavelength,
        }
    )

    x = sim_params.cast(sim_params.x, "x")
    y = sim_params.cast(sim_params.y, "y")
    wl = sim_params.cast(sim_params.wavelength, "wavelength")

    torch.testing.assert_close(
        Wavefront.gaussian_beam(
            sim_params, waist_radius=waist_radius, distance=distance, dx=dx, dy=dy
        ),
        torch.from_numpy(
            gaussian_beam_analytical(
                x=x.numpy(),
                y=y.numpy(),
                w0=waist_radius,
                wavelength=wl.numpy(),
                z=distance,
                dx=dx,
                dy=dy,
            )
        ),
    )
