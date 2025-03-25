from svetlanna.axes_math import _append_slice, _axes_indices_to_sort, _swaps
from svetlanna.axes_math import cast_tensor, _new_axes, tensor_dot
from itertools import permutations
import torch
import pytest


def test_append_slice():
    """Test that append slice"""
    axes = ('a',)
    new_axes = ('a', 'b')
    full_slice = slice(None, None, None)

    # no additional axes
    assert _append_slice(axes, axes) == (..., full_slice)
    assert _append_slice(new_axes, new_axes) == (..., full_slice, full_slice)
    # only one additional axis should be at the end
    assert _append_slice(axes, new_axes) == (..., full_slice, None)

    # two additional axis should be at the end
    for new_axes in permutations(('a', 'b', 'c')):
        assert _append_slice(axes, new_axes) == (..., full_slice, None, None)


def test_axes_indices_to_sort():
    """Test for `_axes_indices_to_sort` function"""
    axes = ('a', 'b')
    new_axes = ('b', 'd', 'a', 'c')
    # axes of the tensor expanded with _append_slice
    appended_tensor_axes = ('a', 'b', 'd', 'c')

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
    a = torch.tensor([[1, 2], [3, 4]])

    # additional axes
    b = cast_tensor(a=a, axes=('a',), new_axes=('a', 'b', 'c'))
    assert len(b.shape) == 4
    assert b.shape[-1] == b.shape[-2] == 1

    b = cast_tensor(a=a, axes=('a', 'b'), new_axes=('a', 'b', 'c'))
    assert len(b.shape) == 3
    assert b.shape[-1] == 1

    b = cast_tensor(a=a, axes=('a', 'b'), new_axes=('a', 'b'))
    assert len(b.shape) == 2


def test_new_axes():
    """Axis algebra test
    ```
    (a, b), (a) -> (a, b)  # existing axis
    (a, b), (c) -> (a, b, c)  # non-existing axis
    (a, b), (c, b) -> (a, b, c)  # both cases
    ```
    """

    assert _new_axes(('a', 'b'), ('a',)) == ('a', 'b')

    assert _new_axes(('a', 'b'), ('c',)) == ('a', 'b', 'c')
    assert _new_axes(('a', 'b'), ('c', 'd')) == ('a', 'b', 'c', 'd')

    assert _new_axes(('a', 'b'), ('a', 'c')) == ('a', 'b', 'c')
    assert _new_axes(('a', 'b'), ('c', 'a')) == ('a', 'b', 'c')
    assert _new_axes(('a', 'b'), ('b', 'c')) == ('a', 'b', 'c')
    assert _new_axes(('a', 'b'), ('c', 'b')) == ('a', 'b', 'c')
    assert _new_axes(('a', 'b'), ('c', 'd', 'b', 'e')) == ('a', 'b', 'c', 'd', 'e')


@pytest.mark.skip
def test_tensor_dot():
    """Test `tensor_dot` function"""
    # TODO: write the test
