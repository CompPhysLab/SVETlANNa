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


def _spherical_wave_symbolic():
    x, y, wavelength, z, dx, dy, phi0 = sp.symbols(
        "x y wavelength z dx dy phi0", real=True
    )

    k = 2 * sp.pi / wavelength
    r = sp.sqrt((x - dx) ** 2 + (y - dy) ** 2 + z**2)

    E = (sp.S.One / r) * sp.exp(sp.I * (k * r + phi0))

    return sp.lambdify((x, y, wavelength, z, dx, dy, phi0), E, "numpy")


gaussian_beam_analytical = _gaussian_beam_symbolic()
spherical_wave_analytical = _spherical_wave_symbolic()


def test_creation():
    wf = Wavefront(1.0)
    assert isinstance(wf, torch.Tensor)

    wf = Wavefront(1.0 + 1.0j)
    assert isinstance(wf, torch.Tensor)

    wf = Wavefront([1 + 2.0j])
    assert isinstance(wf, torch.Tensor)

    data = torch.tensor([1, 2, 3])
    wf = Wavefront(data)
    assert isinstance(wf, torch.Tensor)
    assert isinstance(wf, Wavefront)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (1.0, 2.0),
        (
            1.0,
            1.0,
        ),
        (-1.0, 1.3),
    ],
)
def test_intensity(a: float, b: float):
    """Test intensity calculations"""
    wf = Wavefront([a + 1j * b])
    real_intensity = torch.tensor([a**2 + b**2])

    torch.testing.assert_close(wf.intensity, real_intensity)

    # Test maximum
    real_maximum = torch.max(wf.intensity).item()
    torch.testing.assert_close(wf.max_intensity, real_maximum)

    # Test type
    assert not isinstance(wf.intensity, Wavefront)
    assert isinstance(wf.intensity, torch.Tensor)


@pytest.mark.parametrize(
    ("r", "phi"), [(1.0, 0.0), (1.0, [1.0]), (10.0, [1.0, 2.0, 3.0])]
)
def test_phase(r, phi):
    wf = Wavefront(r * torch.exp(1j * torch.tensor(phi)))

    torch.testing.assert_close(wf.phase, torch.tensor(phi))


@pytest.mark.parametrize("waist_radius", (1, 0.5, 0.2))
def test_fwhm(waist_radius):
    sim_params = SimulationParameters(
        {
            "x": torch.linspace(-1, 1, 1000),
            "y": torch.linspace(-1, 1, 1000),
            "wavelength": 1,
        }
    )

    wf = Wavefront.gaussian_beam(
        sim_params, waist_radius=waist_radius, distance=0, dx=0, dy=0
    )

    # Test symmetric Gaussian beam FWHM
    assert wf.fwhm(sim_params)[0] == wf.fwhm(sim_params)[1]
    torch.testing.assert_close(
        torch.tensor(wf.fwhm(sim_params)[0]),
        torch.sqrt(2 * torch.log(torch.tensor(2.0))) * waist_radius,
        rtol=0.001,
        atol=0.01,
    )


@pytest.mark.parametrize("distance", (1, 1.23, 1e-4, 1e4))
@pytest.mark.parametrize("wavelength", (1.0, torch.tensor([1.23, 20])))
@pytest.mark.parametrize("initial_phase", (1.0, 123, 2e-4))
def test_plane_wave(distance, wavelength, initial_phase):
    sim_params = SimulationParameters(
        {
            "x": torch.linspace(-0.1, 2, 10),
            "y": torch.linspace(-1, 5, 20),
            "wavelength": wavelength,
        }
    )
    k = 2 * torch.pi / sim_params.wavelength

    # z propagation
    wf = Wavefront.plane_wave(
        sim_params, distance=distance, initial_phase=initial_phase
    )
    assert isinstance(wf, Wavefront)
    torch.allclose(
        wf.angle(),
        torch.exp(1j * (k * distance + initial_phase)[..., None, None]).angle(),
    )
    torch.allclose(wf.abs(), torch.tensor(1.0))

    # x,y propagation
    dir_x = 0.1312234
    dir_y = 0.5231432
    kx = k * dir_x / torch.linalg.norm(torch.tensor([dir_x, dir_y]))
    ky = k * dir_x / torch.linalg.norm(torch.tensor([dir_x, dir_y]))
    x = sim_params.x[None, :]
    y = sim_params.y[:, None]
    wf = Wavefront.plane_wave(
        sim_params,
        distance=distance,
        wave_direction=[dir_x, dir_y, 0],
        initial_phase=initial_phase,
    )
    torch.allclose(
        wf.angle(),
        torch.exp(
            1j * (kx[..., None, None] * x + ky[..., None, None] * y + initial_phase)
        ).angle(),
    )
    torch.allclose(wf.abs(), torch.tensor(1.0))

    # Test wrong wave direction
    with pytest.raises(ValueError):
        Wavefront.plane_wave(
            sim_params, distance=distance, wave_direction=[dir_x, dir_y]
        )


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


def test_wavefront_as_a_tensor():
    tensor = torch.rand((2, 10, 20))
    wf = Wavefront(tensor)

    # Test arithmetical operations results type
    assert isinstance(wf, Wavefront)
    assert isinstance(wf + tensor, Wavefront)
    assert isinstance(tensor + wf, Wavefront)
    assert isinstance(tensor * wf, Wavefront)
    assert isinstance(wf * tensor, Wavefront)
    assert isinstance(tensor / wf, Wavefront)
    assert isinstance(wf / tensor, Wavefront)
