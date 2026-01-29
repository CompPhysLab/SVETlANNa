from svetlanna import Wavefront, SimulationParameters
import torch
import pytest


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
    k = 2 * torch.pi / sim_params.axes.wavelength

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
    x = sim_params.axes.x[None, :]
    y = sim_params.axes.y[:, None]
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


# TODO: Test Gaussian beam against precomputed values
@pytest.mark.parametrize("distance", (1, 1.23, 1e-4, 1e4))
@pytest.mark.parametrize("waist_radius", (1, 1.23, 1e-4, 1e4))
@pytest.mark.parametrize("dx", (1.0, 123, 2e-4))
@pytest.mark.parametrize("dy", (1.0, 123, 2e-4))
@pytest.mark.parametrize("wavelength", (1.0, torch.tensor([1.23, 20])))
def test_gaussian_beam(distance, waist_radius, dx, dy, wavelength):
    sim_params = SimulationParameters(
        {
            "x": torch.linspace(-0.1, 2, 10),
            "y": torch.linspace(-1, 5, 20),
            "wavelength": wavelength,
        }
    )
    # Stupid test
    Wavefront.gaussian_beam(
        sim_params, waist_radius=waist_radius, distance=distance, dx=dx, dy=dy
    )


# TODO: Test spherical wave against precomputed values
@pytest.mark.parametrize("distance", (1, 1.23, 1e-4, 1e4))
@pytest.mark.parametrize("initial_phase", (1, 1.23, 1e-4, 1e4))
@pytest.mark.parametrize("dx", (1.0, 123, 2e-4))
@pytest.mark.parametrize("dy", (1.0, 123, 2e-4))
@pytest.mark.parametrize("wavelength", (1.0, torch.tensor([1.23, 20])))
def test_spherical_wave(distance, initial_phase, dx, dy, wavelength):
    sim_params = SimulationParameters(
        {
            "x": torch.linspace(-0.1, 2, 10),
            "y": torch.linspace(-1, 5, 20),
            "wavelength": wavelength,
        }
    )
    # Stupid test
    Wavefront.spherical_wave(
        sim_params, distance, initial_phase=initial_phase, dx=dx, dy=dy
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
