from svetlanna import Wavefront, SimulationParameters
import torch
import pytest

import sympy as sp  # type: ignore


def _spherical_wave_symbolic():
    x, y, wavelength, z, dx, dy, phi0 = sp.symbols(
        "x y wavelength z dx dy phi0", real=True
    )

    k = 2 * sp.pi / wavelength
    r = sp.sqrt((x - dx) ** 2 + (y - dy) ** 2 + z**2)

    E = (sp.S.One / r) * sp.exp(sp.I * (k * r + phi0))

    return sp.lambdify((x, y, wavelength, z, dx, dy, phi0), E, "numpy")


spherical_wave_analytical = _spherical_wave_symbolic()


@pytest.mark.parametrize("distance", (1, 10, 100))
@pytest.mark.parametrize("initial_phase", (1, 1.23, 1e-4, 20))
@pytest.mark.parametrize("dx", (1.0, 123, 2e-4))
@pytest.mark.parametrize("dy", (1.0, 123, 2e-4))
@pytest.mark.parametrize("wavelength", (1.0, torch.tensor([1.23, 10.0, 40.0])))
def test_spherical_wave(distance, initial_phase, dx, dy, wavelength):
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
        Wavefront.spherical_wave(
            sim_params, distance, initial_phase=initial_phase, dx=dx, dy=dy
        ),
        torch.from_numpy(
            spherical_wave_analytical(
                x=x.numpy(),
                y=y.numpy(),
                wavelength=wl.numpy(),
                z=distance,
                dx=dx,
                dy=dy,
                phi0=initial_phase,
            )
        ),
    )
