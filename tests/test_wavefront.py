from svetlanna import Wavefront
import torch
import pytest


def test_creation():
    wf = Wavefront(1.)
    assert isinstance(wf, torch.Tensor)

    wf = Wavefront(1. + 1.j)
    assert isinstance(wf, torch.Tensor)

    wf = Wavefront([1 + 2.j])
    assert isinstance(wf, torch.Tensor)

    data = torch.tensor([1, 2, 3])
    wf = Wavefront(data)
    assert isinstance(wf, torch.Tensor)
    assert isinstance(wf, Wavefront)


@pytest.mark.parametrize(
    ('a', 'b'), [
        (1., 2.),
        (1., 1.,),
        (-1., 1.3)
    ]
)
def test_intensity(a: float, b: float):
    """Test intensity calculations"""
    wf = Wavefront([a + 1j*b])
    real_intensity = torch.tensor([a**2 + b**2])

    torch.testing.assert_close(wf.intensity, real_intensity)


@pytest.mark.parametrize(
    ('r', 'phi'), [
        (1., 0.),
        (1., 1.,),
        (10., 3.)
    ]
)
def test_phase(r, phi):
    wf = Wavefront(r * torch.exp(torch.tensor(1j * phi)))

    torch.testing.assert_close(wf.phase, torch.tensor(phi))
