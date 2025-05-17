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
    """
    Tests arithmetic operations with the unit 'mm'.

        This function checks if basic arithmetic operations (multiplication, division, and exponentiation)
        between a given value and the 'mm' unit from astropy.units produce the expected results when compared to
        the underlying numerical value of the unit. It tests both left-hand and right-hand side operations.

        Args:
            other: The value to perform arithmetic with. Can be an integer, float, torch tensor or numpy array.

        Returns:
            None: This function only performs assertions and does not return a value.
    """
    torch.testing.assert_close(other * ureg.mm, other * ureg.mm.value)
    torch.testing.assert_close(ureg.mm * other, other * ureg.mm.value)
    torch.testing.assert_close(other / ureg.mm, other / ureg.mm.value)
    torch.testing.assert_close(ureg.mm / other, ureg.mm.value / other)
    torch.testing.assert_close(ureg.mm**other, ureg.mm.value**other)


def test_array_api():
    """
    Tests array API compatibility with pint and numpy.

        This tests that adding a pint Quantity to a NumPy array results in a NumPy array,
        and that attempting to use __array__ with copy=False on a pint unit raises a ValueError.
    """
    assert isinstance(ureg.m + np.array([0.0]), np.ndarray)

    with pytest.raises(ValueError):
        ureg.mm.__array__(copy=False)
