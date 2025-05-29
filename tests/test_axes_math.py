from svetlanna.axes_math import _append_slice, _axes_indices_to_sort, _swaps
from svetlanna.axes_math import cast_tensor, _new_axes, _axis_to_tuple
from svetlanna.axes_math import is_scalar, _check_axis, tensor_dot
from svetlanna.wavefront import Wavefront, mul
from svetlanna import SimulationParameters
from itertools import permutations
import torch
import pytest


def test_append_slice():
    """Test that append slice"""
    axes = ("a",)
    new_axes = ("a", "b")
    full_slice = slice(None, None, None)

    # no additional axes
    assert _append_slice(axes, axes) == (..., full_slice)
    assert _append_slice(new_axes, new_axes) == (..., full_slice, full_slice)
    # only one additional axis should be at the end
    assert _append_slice(axes, new_axes) == (..., full_slice, None)

    # two additional axis should be at the end
    for new_axes in permutations(("a", "b", "c")):
        assert _append_slice(axes, new_axes) == (..., full_slice, None, None)


def test_axes_indices_to_sort():
    """Test for `_axes_indices_to_sort` function"""
    axes = ("a", "b")
    new_axes = ("b", "d", "a", "c")
    # axes of the tensor expanded with _append_slice
    appended_tensor_axes = ("a", "b", "d", "c")

    assert _axes_indices_to_sort(axes, new_axes) == tuple(
        new_axes.index(axis) for axis in appended_tensor_axes
    )


def test_swaps():
    """Test if `_swaps` function works properly"""
    axes = [1, 2, 3, 4]
    for new_axes in permutations(axes):
        new_axes_list = list(new_axes)

        # elements swap
        for i, j in _swaps(new_axes):
            new_axes_list[i], new_axes_list[j] = new_axes_list[j], new_axes_list[i]

        # test if new_axes_list is sorted after swapping
        assert sorted(axes) == new_axes_list


def test_cast_tensor():
    """
    Tests the cast_tensor function with various axis configurations.

        This tests checks that the `cast_tensor` function correctly adds, maintains, and swaps axes
        of a given tensor while raising ValueErrors when invalid configurations are provided.

        Args:
            None

        Returns:
            None
    """
    a = torch.tensor([[1, 2], [3, 4]])

    # additional axes
    b = cast_tensor(a=a, axes=("a",), new_axes=("a", "b", "c"))
    assert len(b.shape) == 4
    assert b.shape[-1] == b.shape[-2] == 1

    b = cast_tensor(a=a, axes=("a", "b"), new_axes=("a", "b", "c"))
    assert len(b.shape) == 3
    assert b.shape[-1] == 1

    # same axes test
    b = cast_tensor(a=a, axes=("a", "b"), new_axes=("a", "b"))
    assert len(b.shape) == 2

    # swap axes test
    b = cast_tensor(a=a, axes=("a", "b"), new_axes=("b", "a"))
    assert torch.allclose(a, b.T)

    with pytest.raises(ValueError):
        b = cast_tensor(a=a, axes=("a", "b"), new_axes=("a", "c"))


def test_axis_to_tuple():
    """
    Tests the _axis_to_tuple function for various inputs and caching behavior.

        This test verifies that _axis_to_tuple correctly converts different input types
        (empty tuple, string, tuple of strings) into tuples and also confirms that it
        caches results for identical inputs.

        Parameters:
            None

        Returns:
            None
    """
    a = _axis_to_tuple(())
    b = _axis_to_tuple("a")
    c = _axis_to_tuple(("a", "b"))

    # test for values
    assert a == ()
    assert b == ("a",)
    assert c == ("a", "b")

    # check for cache
    assert a is _axis_to_tuple(())
    assert b is _axis_to_tuple("a")
    assert c is _axis_to_tuple(("a", "b"))


def test_new_axes():
    """Axis algebra test
    ```
    (a, b), (a) -> (a, b)  # existing axis
    (a, b), (c) -> (a, b, c)  # non-existing axis
    (a, b), (c, b) -> (a, b, c)  # both cases
    ```
    """

    assert _new_axes(("a", "b"), ("a",)) == ("a", "b")

    assert _new_axes(("a", "b"), ("c",)) == ("a", "b", "c")
    assert _new_axes(("a", "b"), ("c", "d")) == ("a", "b", "c", "d")

    assert _new_axes(("a", "b"), ("a", "c")) == ("a", "b", "c")
    assert _new_axes(("a", "b"), ("c", "a")) == ("a", "b", "c")
    assert _new_axes(("a", "b"), ("b", "c")) == ("a", "b", "c")
    assert _new_axes(("a", "b"), ("c", "b")) == ("a", "b", "c")
    assert _new_axes(("a", "b"), ("c", "d", "b", "e")) == ("a", "b", "c", "d", "e")


def test_is_scalar():
    """
    Tests the is_scalar function with various inputs.

        This tests that single number values and tensors containing a single value are correctly identified as scalar,
        while tensors containing multiple values are not.

        Returns:
            None
    """
    assert is_scalar(123.0)
    assert is_scalar(torch.tensor(123.0))
    assert not is_scalar(torch.tensor([123.0]))
    assert not is_scalar(torch.tensor([123.0, 123]))
    assert not is_scalar(torch.tensor([[123.0, 123]]))


def test_check_axis():
    """
    Tests the _check_axis function for various error conditions."""
    # test for unique
    with pytest.raises(ValueError):
        _check_axis(torch.tensor([[[123]]]), ("a", "a", "b"))

    # test for number of axes in tensor
    with pytest.raises(ValueError):
        _check_axis(torch.tensor([123]), ("a", "b"))

    # and for number of axes in float
    assert _check_axis(123, ("a", "b")) is None


def test_tensor_dot():
    """Test `tensor_dot` function"""
    e = 123
    d = 321
    # product of a scalar and a scalar
    c, c_axis = tensor_dot(d, e, ("a", "b"), ("c"))
    assert 123 * d == c
    assert c_axis == ()

    c, c_axis = tensor_dot(d, e, ("a", "b"), ("c"), preserve_a_axis=True)
    assert e * d == c
    assert c_axis == ("a", "b")

    # product of a tensor and a scalar
    a = torch.tensor([1.0])
    b = torch.tensor([[1.0, 2], [3.0, 4.0]])

    c, c_axis = tensor_dot(a, e, (), ())
    assert e * a == c
    assert c_axis == ()

    c, c_axis = tensor_dot(b, e, (), ())
    assert torch.allclose(e * b, c)
    assert c_axis == ()

    c, c_axis = tensor_dot(a, e, ("a",), ())
    assert e * a == c
    assert c_axis == ("a",)

    c, c_axis = tensor_dot(b, e, ("a",), ())
    assert torch.allclose(e * b, c)
    assert c_axis == ("a",)

    c, c_axis = tensor_dot(a, e, ("a",), ("b", "c"))
    assert e * a == c
    assert c_axis == ("a",)

    c, c_axis = tensor_dot(b, e, ("a",), ("b", "c"))
    assert torch.allclose(e * b, c)
    assert c_axis == ("a",)

    c, c_axis = tensor_dot(b, e, ("a", "d"), ("b", "c"))
    assert torch.allclose(e * b, c)
    assert c_axis == ("a", "d")

    # product of a scalar and a tensor
    c, c_axis = tensor_dot(e, a, (), ())
    assert e * a == c
    assert c_axis == ()

    c, c_axis = tensor_dot(e, b, (), ())
    assert torch.allclose(e * b, c)
    assert c_axis == ()

    c, c_axis = tensor_dot(e, a, (), ("a"))
    assert e * a == c
    assert c_axis == ("a",)

    c, c_axis = tensor_dot(e, b, (), ("a"))
    assert torch.allclose(e * b, c)
    assert c_axis == ("a",)

    c, c_axis = tensor_dot(e, a, ("a",), ("a"))
    assert e * a == c
    assert c_axis == ("a",)

    c, c_axis = tensor_dot(e, a, ("a", "c"), ("a"))
    assert e * a == c
    assert c_axis == ("a",)

    c, c_axis = tensor_dot(e, a, ("a", "c"), ("a"), preserve_a_axis=True)
    assert e * a == c
    assert c_axis == ("a", "c")

    c, c_axis = tensor_dot(e, b, ("a", "c"), ("a"), preserve_a_axis=True)
    assert torch.allclose((e * b)[..., None], c)
    assert c_axis == ("a", "c")

    # product of a tensor and a tensor
    c, c_axis = tensor_dot(a, b, (), ())
    assert torch.allclose((a * b), c)
    assert c_axis == ()

    c, c_axis = tensor_dot(a, b, ("a"), ("a", "b"))
    d = b.clone()
    d[:] *= a[:]
    assert torch.allclose(c, d)
    assert c_axis == ("a", "b")

    c, c_axis = tensor_dot(a, b, ("a"), ("a"))
    d = b.clone()
    d[..., :] *= a[:]
    assert torch.allclose(c, d)
    assert c_axis == ("a",)

    c, c_axis = tensor_dot(b, a, ("a", "b"), ("a"))
    d = b.clone()
    d[:, ...] *= a[:]
    assert torch.allclose(c, d)
    assert c_axis == ("a", "b")

    c, c_axis = tensor_dot(b, a, ("a", "b"), ("c"))
    d = b.clone()[..., None]
    d[..., :] *= a[:]
    assert torch.allclose(c, d)
    assert c_axis == ("a", "b", "c")


def test_mul():
    """
    Tests the mul function with various Wavefront objects and tensors."""
    wf = Wavefront([[1.0 + 1j]])

    # test wf and non-tensor product
    assert mul(wf, 123, ()) == wf * 123

    # test default axes
    wf = Wavefront([[1.0, 2], [3, 4]])
    a = torch.tensor([10, 20])

    assert torch.allclose(mul(wf, a, ("H")), wf * a[:, None])
    assert torch.allclose(mul(wf, a, ("W")), wf * a[None, :])
    with pytest.raises(AssertionError):
        mul(wf, torch.tensor([123]), ("s"))

    # test non default axes
    sim_params1 = SimulationParameters(
        axes={
            "H": torch.linspace(-1, 1, 2),
            "W": torch.linspace(-1, 1, 2),
            "wavelength": torch.tensor([1]),
        }
    )

    wf1 = Wavefront([[[1.0, 2], [3, 4]]])
    assert torch.allclose(mul(wf1, 123, "wavelength", sim_params1), 123 * wf1)
    r = wf1 * a[None, :]
    assert torch.allclose(mul(wf1, a, "H", sim_params1), r)

    # test the same product but with other simulation parameters
    sim_params2 = SimulationParameters(
        axes={
            "wavelength": torch.tensor([1]),
            "W": torch.linspace(-1, 1, 2),
            "H": torch.linspace(-1, 1, 2),
        }
    )
    wf2 = Wavefront(wf1.swapaxes(0, 2))
    assert torch.allclose(mul(wf2, a, "H", sim_params2), r.swapaxes(0, 2))
