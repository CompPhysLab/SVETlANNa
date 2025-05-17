from svetlanna import Wavefront, SimulationParameters
import torch
import pytest


def test_creation():
    """
    Tests the creation of Wavefront objects with various inputs.

        This method checks that a Wavefront object can be successfully initialized
        with different types of input data (float, complex number, list of complex numbers, and torch tensor)
        and verifies that the resulting object is a PyTorch tensor and an instance of the Wavefront class.

        Returns:
            None
    """
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
    """
    Tests that the wavefront phase is correctly initialized.

        Args:
            r: The radius of the wavefront.
            phi: The initial phase values.

        Returns:
            None: This function asserts a condition and does not return a value.
    """
    wf = Wavefront(r * torch.exp(1j * torch.tensor(phi)))

    torch.testing.assert_close(wf.phase, torch.tensor(phi))


@pytest.mark.parametrize("waist_radius", (1, 0.5, 0.2))
def test_fwhm(waist_radius):
    """
    Tests the full width at half maximum (FWHM) calculation for a Gaussian beam.

        Args:
            waist_radius: The waist radius of the Gaussian beam.

        Returns:
            None: This function asserts properties of the FWHM and does not return a value.
    """
    sim_params = SimulationParameters(
        {
            "W": torch.linspace(-1, 1, 1000),
            "H": torch.linspace(-1, 1, 1000),
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
    """
    Tests the plane_wave method of the Wavefront class.

        Args:
            distance: The distance to propagate the plane wave.
            wavelength: The wavelength of the plane wave.
            initial_phase: The initial phase of the plane wave.

        Returns:
            None.  This function asserts properties of the generated Wavefront object.
    """
    sim_params = SimulationParameters(
        {
            "W": torch.linspace(-0.1, 2, 10),
            "H": torch.linspace(-1, 5, 20),
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
    x = sim_params.axes.W[None, :]
    y = sim_params.axes.H[:, None]
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
    """
    Tests the gaussian_beam method with various parameters.

        Args:
            distance: The distance to propagate the beam.
            waist_radius: The radius of the Gaussian beam at its waist.
            dx:  Offset in x direction.
            dy: Offset in y direction.
            wavelength: The wavelength of the light. Can be a float or a torch tensor.

        Returns:
            None: This test does not return any value; it asserts that the
                  gaussian_beam method runs without errors for given parameters.
    """
    sim_params = SimulationParameters(
        {
            "W": torch.linspace(-0.1, 2, 10),
            "H": torch.linspace(-1, 5, 20),
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
    """
    Tests the spherical wave function with various parameters.

        Args:
            distance: The distance from the source of the spherical wave.
            initial_phase: The initial phase of the wave.
            dx:  The x-coordinate offset.
            dy: The y-coordinate offset.
            wavelength: The wavelength of the wave.

        Returns:
            None: This function does not return a value; it asserts that the
                spherical_wave function runs without errors for given parameters.
    """
    sim_params = SimulationParameters(
        {
            "W": torch.linspace(-0.1, 2, 10),
            "H": torch.linspace(-1, 5, 20),
            "wavelength": wavelength,
        }
    )
    # Stupid test
    Wavefront.spherical_wave(
        sim_params, distance, initial_phase=initial_phase, dx=dx, dy=dy
    )


def test_wavefront_as_a_tensor():
    """
    Tests that Wavefront operations with tensors return a Wavefront object.

        This method creates a Wavefront object from a random tensor and then performs
        various arithmetic operations (addition, multiplication, division) between the
        Wavefront object and the original tensor. It asserts that the result of each
        operation is also a Wavefront object.

        Args:
            None

        Returns:
            None
    """
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
