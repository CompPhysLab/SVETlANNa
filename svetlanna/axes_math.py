from typing import Iterable, Generator, cast
from functools import cache
from types import EllipsisType
import torch


def _append_slice_generator(
    axes_number: int,
    new_axes_number: int
):
    """Yields Ellipsis, then `axes_number` of full slices (`::`), then
    `new_axes_number - axes_number` of None (new axis).

    Parameters
    ----------
    axes_number : int
        number of existing axes
    new_axes_number : int
        number of new axes

    Yields
    ------
    EllipsisType | slice | None
    """
    full_slice = slice(None, None, None)
    yield ...
    for _ in range(axes_number):
        yield full_slice
    for _ in range(new_axes_number - axes_number):
        yield None


@cache
def _append_slice(
    axes: tuple[str, ...],
    new_axes: tuple[str, ...]
) -> tuple[EllipsisType | slice | None, ...]:
    """
    Slice tuple that can be used to add new axes to the end
    of the tensor if `new_axes` is larger than `axes`.
    """
    axes_number = len(axes)
    new_axes_number = len(new_axes)
    return tuple(_append_slice_generator(axes_number, new_axes_number))


@cache
def _axes_indices_to_sort(
    axes: tuple[str, ...],
    new_axes: tuple[str, ...]
) -> tuple[int, ...]:
    """
    Indices of each axis of `axes` in `new_axes`.
    For example:
    ```
    axes = ('a', 'b')
    new_axes = ('b', 'd', 'a', 'c')

    # one should think about axes to sort as ('a', 'b', 'd', 'c'),
    # the axes appended with all axes presented in `new_axes` and not in `axes`

    >>> _axes_indices_to_sort(axes, new_axes)
    >>> (2, 0, 1, 3)
    ```
    """

    axes_number = len(axes)
    axes_indices = [-1 for _ in range(axes_number)]

    for i, axis in enumerate(new_axes):
        try:
            axes_indices[axes.index(axis)] = i
        except ValueError:
            axes_indices.append(i)

    return tuple(axes_indices)


def _swaps_generator(
    axes_indices: tuple[int, ...]
) -> Generator[tuple[int, int], None, None]:
    """
    Generates swaps to sort indices array.
    Based on selection sort.
    """
    indices = list(axes_indices)

    L = len(indices)
    for i in range(L - 1):
        j_min = i

        for j in range(i+1, L):
            if indices[j] < indices[j_min]:
                j_min = j

        if j_min != i:
            indices[i], indices[j_min] = indices[j_min], indices[i]
            yield -L+i, -L+j_min


@cache
def _swaps(
    axes_indices: tuple[int, ...]
) -> tuple[tuple[int, int], ...]:
    """
    Swaps to sort indices array.
    """
    return tuple(_swaps_generator(axes_indices))


@cache
def _check_new_axes(
    axes: tuple[str, ...],
    new_axes: tuple[str, ...]
) -> None:
    """
    Check whether `new_axes` contain all names presented in `axes`.
    """
    if not set(new_axes).issuperset(axes):
        raise ValueError('new_axes should contain all names in axes')


def cast_tensor(
    a: torch.Tensor,
    axes: tuple[str, ...],
    new_axes: tuple[str, ...]
) -> torch.Tensor:
    """Cast tensor `a` with axes `(..., a, b, c)` to `(..., *new_axes)`.
    `new_axes` should contain all axes presented in `axes`.

    Parameters
    ----------
    a : torch.Tensor
        a tensor to cast
    axes : tuple[str, ...]
        last axes of the tensor
    new_axes : tuple[str, ...]
        last axes of the resulting tensor

    Returns
    -------
    torch.Tensor
        tensor with `new_axes` as last axes
    """

    _check_new_axes(axes, new_axes)
    axes_indices = _axes_indices_to_sort(axes, new_axes)

    # add new axes
    a = a[_append_slice(axes, new_axes)]

    # swap required axes
    for i, j in _swaps(axes_indices):
        a = a.swapaxes(i, j)

    return a


@cache
def _axis_to_tuple(
    axis: str | Iterable[str]
) -> tuple[str, ...]:
    """Creates tuple of `str` from `str` or `Iterable[str]`."""
    if isinstance(axis, str):
        axis = (axis, )
    return tuple(axis)


@cache
def _new_axes(
    a_axis: tuple[str, ...],
    b_axis: tuple[str, ...]
) -> tuple[str, ...]:
    """
    Generates tuple with new axes.
    ```
    (a, b), (a) -> (a, b)
    (a, b), (c) -> (a, b, c)
    (a, b), (c, b) -> (a, b, c)
    ```
    """
    new_axes = list(a_axis)

    for axis in b_axis:
        if axis not in new_axes:
            new_axes.append(axis)

    return tuple(new_axes)


def is_scalar(a: torch.Tensor | float) -> bool:
    """Check if the value scalar, meaning 0-dimensional tensor or float

    Parameters
    ----------
    a : torch.Tensor | float
        value to check

    Returns
    -------
    bool
        test result
    """
    if isinstance(a, (int, float, complex)):
        return True
    if isinstance(a, torch.Tensor) and not a.shape:
        return True
    return False


def _check_axis(
    a: torch.Tensor | float,
    a_axis: tuple[str, ...]
):
    """
    Check if each axis is unique.
    Check whether the number of axes not greater than the dimensionality
    of the tensor `a` if `a` is a tensor (of 1 or higher dimensionality).
    The last check is not performed if `a` is 0-dimensional tensor or float.
    """
    if len(a_axis) != len(set(a_axis)):
        raise ValueError("Each axis must be unique in the axes list!")

    if isinstance(a, torch.Tensor) and a.shape:  # if a not a scalar
        if len(a.shape) < len(a_axis):
            raise ValueError(f"Number of axes in the tensor ({len(a.shape)}) should be larger than number of provided axes' names ({len(a_axis)})!")


def tensor_dot(
    a: torch.Tensor | float,
    b: torch.Tensor | float,
    a_axis: str | Iterable[str],
    b_axis: str | Iterable[str],
    preserve_a_axis: bool = False
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Perform tensor dot product.

    Parameters
    ----------
    a : torch.Tensor | float
        first tensor
    b : torch.Tensor | float
        second tensor
    a_axis : str | Iterable[str]
        axis name of the first tensor
    b_axis : str | Iterable[str]
        axis name of the second tensor
    preserve_a_axis : bool, optional
        check if the resulting tensor axes are coincide with the `a` tensor axes, by default False

    Returns
    -------
    tuple[torch.Tensor, tuple[str, ...]]
        Product result and its axes names
    """

    # axis to tuple
    a_axis = _axis_to_tuple(a_axis)
    b_axis = _axis_to_tuple(b_axis)

    if is_scalar(a):
        if not preserve_a_axis:
            a_axis = tuple()
    if is_scalar(b):
        b_axis = tuple()

    # check axes dims
    _check_axis(a, a_axis)
    _check_axis(b, b_axis)

    # generate axes of the resulting tensor
    new_axes = _new_axes(a_axis, b_axis)

    if preserve_a_axis:
        assert len(new_axes) == len(a_axis), "Can't preserve axes of the first tensor"

    if not is_scalar(a):
        a = cast(torch.Tensor, a)
        a = cast_tensor(a, a_axis, new_axes)
    if not is_scalar(b):
        b = cast(torch.Tensor, b)
        b = cast_tensor(b, b_axis, new_axes)

    return a * b, new_axes
