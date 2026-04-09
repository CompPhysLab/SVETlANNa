from svetlanna.units import ureg
import torch
import numpy as np
import pytest


@pytest.mark.parametrize(
    "other",
    (
        123,
        1.234,
        torch.tensor(123),
        torch.tensor(1.234),
        torch.tensor([[1.23, 4.56]]),
        np.array(123),
        np.array(1.234),
        np.array([[1.23, 4.56]]),
    ),
)
def test_arithmetics(other):
    torch.testing.assert_close(other * ureg.mm, other * ureg.mm.value)
    torch.testing.assert_close(ureg.mm * other, other * ureg.mm.value)
    torch.testing.assert_close(other / ureg.mm, other / ureg.mm.value)
    torch.testing.assert_close(ureg.mm / other, ureg.mm.value / other)
    torch.testing.assert_close(ureg.mm**other, ureg.mm.value**other)


def test_array_api():
    assert isinstance(ureg.m + np.array([0.0]), np.ndarray)

    with pytest.raises(ValueError):
        ureg.mm.__array__(copy=False)
