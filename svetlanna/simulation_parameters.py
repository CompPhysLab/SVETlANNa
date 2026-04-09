from __future__ import annotations
from typing import Any, TYPE_CHECKING, Self, overload
from typing_extensions import deprecated
import torch
from torch import nn
import warnings
from collections.abc import Mapping


class AxisNotFound(Exception):
    """Raised when trying to access a non-existent axis."""

    pass


REQUIRED_AXES = ("x", "y", "wavelength")

# nn.Module attributes that would break if used as axis names
_RESERVED_AXIS_NAMES = frozenset(
    {
        "training",
        "forward",
        "extra_repr",
        "children",
        "modules",
        "named_modules",
        "parameters",
        "named_parameters",
        "buffers",
        "named_buffers",
        "state_dict",
        "load_state_dict",
        "register_buffer",
        "register_parameter",
        "register_module",
        "add_module",
        "apply",
        "zero_grad",
        "train",
        "eval",
        "requires_grad_",
        "to",
        "cpu",
        "cuda",
        "half",
        "float",
        "double",
        "bfloat16",
    }
)


def legacy_axis_support(name: str) -> str:
    if name == "W":  # legacy
        warnings.warn(
            "'W' axis is deprecated, use 'x' instead", DeprecationWarning, stacklevel=2
        )
        return "x"
    if name == "H":  # legacy
        warnings.warn(
            "'H' axis is deprecated, use 'y' instead", DeprecationWarning, stacklevel=2
        )
        return "y"
    return name


class SimulationParameters(nn.Module):

    @overload
    def __init__(
        self,
        axes: Mapping[str, torch.Tensor | float],
        /,
    ) -> None: ...

    @overload
    def __init__(
        self,
        /,
        *,
        x: torch.Tensor | float,
        y: torch.Tensor | float,
        wavelength: torch.Tensor | float,
        **additional_axes: torch.Tensor | float,
    ) -> None: ...

    def __init__(
        self,
        axes: Mapping[str, torch.Tensor | float] | None = None,
        /,
        **kwaxes: torch.Tensor | float,
    ) -> None:
        """
        Simulation parameters.
        Manages coordinate systems and physical parameters for optical simulations.
        Required axes: `x`, `y`, `wavelength`.
        Additional axes can be added.

        Inherits from ``nn.Module`` so that axes are registered as buffers and
        participate in automatic device management when used as submodules of
        Elements.

        Note
        ----
        Axes are registered as **non-persistent** buffers (``persistent=False``).
        This means they are **not included** in ``state_dict()`` and will not
        be saved during checkpointing. The simulation grid must be provided
        when constructing the model; it does not need to be restored from
        a checkpoint.

        Examples
        --------
        Let's define simalation grid of width and height of 1 mm with 512 points for both axes (`Nx=Ny=512`) and wavelength of 632.8 nm:
        ```python
        import svetlanna as sv
        from svetlanna.units import ureg
        import torch

        sim_params = sv.SimulationParameters(
            x=torch.linspace(-0.5, 0.5, 512) * ureg.mm,
            y=torch.linspace(-0.5, 0.5, 512) * ureg.mm,
            wavelength=632.8 * ureg.nm,
        )
        ```
        You can make `wavelength` an array for polychromatic simulations:
        ```python hl_lines="4"
        sim_params = sv.SimulationParameters(
            x=torch.linspace(-0.5, 0.5, 512) * ureg.mm,
            y=torch.linspace(-0.5, 0.5, 512) * ureg.mm,
            wavelength=torch.linspace(600, 800, 10) * ureg.nm,
        )
        ```

        **The order of axes matters!** It defines the order of dimensions in wavefront tensors.
        In first case above, all optical elements will expect wavefront tensors with shape `(..., Ny, Nx)`,
        while in the second case, the expected shape will be `(..., Nwavelength, Ny, Nx)`.
        `...` means any number of leading dimensions (e.g., for batch).

        If you change the order:
        ```python hl_lines="3 4"
        sim_params = sv.SimulationParameters(
            x=torch.linspace(-0.5, 0.5, 512) * ureg.mm,
            wavelength=torch.linspace(600, 800, 10) * ureg.nm,
            y=torch.linspace(-0.5, 0.5, 512) * ureg.mm,
        )
        ```
        the expected order of axes is `('y', 'wavelength', 'x')`, so all optical elements will expect wavefront tensors with shape `(..., Ny, Nwavelength, Nx)`.

        You can add custom axes as needed:
        ```python hl_lines="2 4"
        sim_params = sv.SimulationParameters(
            t=torch.linspace(0, 1, 5) * ureg.s,  # time axis
            x=torch.linspace(-0.5, 0.5, 512) * ureg.mm,
            wavelength=632.8 * ureg.nm,
            y=torch.linspace(-0.5, 0.5, 512) * ureg.mm,
        )
        ```
        In this case, the expected order of axes is `('y', 'x', 't')` as wavelength is scalar, so all optical elements will expect wavefront tensors with shape `(..., Ny, Nx, Nt)`.


        """
        super().__init__()

        # Handle backward compatibility
        if axes is not None:
            if kwaxes:
                raise ValueError(
                    'Cannot use both "axes" dict and keyword arguments. '
                    'Use either {"x": ..., "y": ..., "wavelength": ...} or x=..., y=..., wavelength=...'
                )
            all_axes = dict(axes)
        else:
            # New style initialization
            if (
                ("x" not in kwaxes)
                or ("y" not in kwaxes)
                or ("wavelength" not in kwaxes)
            ):
                raise ValueError(
                    "x, y, and wavelength are required when not using 'axes' dict"
                )
            all_axes = kwaxes

        all_axes = {
            legacy_axis_support(name): value for name, value in all_axes.items()
        }

        # Validate axis names
        for name in all_axes.keys():
            if not isinstance(name, str):
                raise TypeError(
                    f"Axis names must be strings, but got {type(name)}: {name})"
                )
            if name.startswith("_"):
                raise ValueError(
                    f"Axis name '{name}' cannot start with underscore "
                    "(reserved for internal attributes)"
                )
            if name in _RESERVED_AXIS_NAMES:
                raise ValueError(
                    f"Axis name '{name}' conflicts with nn.Module attribute"
                )

        # Check required axes presence
        if not all(name in all_axes.keys() for name in REQUIRED_AXES):
            missing = set(REQUIRED_AXES).difference(all_axes.keys())
            raise ValueError(
                f"Missing required axes: {missing}. " f"Required: {REQUIRED_AXES}"
            )

        # Convert all values to tensors and ensure device consistency
        device = None
        converted_axes: dict[str, torch.Tensor] = {}

        for name, value in all_axes.items():
            if isinstance(value, torch.Tensor):
                if device is None:
                    device = value.device
                elif value.device != device:
                    raise ValueError(
                        f"All axes must be on the same device. "
                        f"Axis '{name}' is on {value.device}, expected {device}"
                    )
                converted_axes[name] = value
            else:
                # Convert scalar/list to tensor
                tensor_value = torch.tensor(value)
                if device is not None:
                    tensor_value = tensor_value.to(device)
                converted_axes[name] = tensor_value

        # Validate x and y are 1-dimensional
        for name in ("x", "y"):
            if converted_axes[name].dim() != 1:
                raise ValueError(
                    f"Axis '{name}' must be 1-dimensional, "
                    f"got {converted_axes[name].dim()}-dimensional"
                )

        # Validate all axes are 0- or 1-dimensional
        non_scalar_names = []
        scalar_names = []

        for name, value in converted_axes.items():
            dim = value.dim()
            if dim not in (0, 1):
                raise ValueError(
                    f"Axis '{name}' must be 0- or 1-dimensional, "
                    f"got {dim}-dimensional"
                )
            if dim == 1:
                non_scalar_names.append(name)
            else:
                scalar_names.append(name)

        # Set default device if none specified
        if device is None:
            device = torch.get_default_device()
            # Move all tensors to default device
            converted_axes = {
                name: tensor.to(device) for name, tensor in converted_axes.items()
            }

        # Register axes as non-persistent buffers
        for name, tensor in converted_axes.items():
            self.register_buffer(name, tensor, persistent=False)

        # Track axis names in plain instance attributes
        self._non_scalar_names_insertion_order: tuple[str, ...] = tuple(
            non_scalar_names
        )
        self._non_scalar_names: tuple[str, ...] = tuple(reversed(non_scalar_names))
        self._scalar_names: tuple[str, ...] = tuple(scalar_names)

        # Initialize dict-based caches (replaces @lru_cache)
        self._cache_axis_sizes: dict[tuple[str, ...] | None, torch.Size] = {}
        self._cache_cast_info: dict[
            tuple[str, ...], tuple[tuple[str, ...], tuple[int, ...]]
        ] = {}

        # Enable read-only protection — MUST be last
        self._all_axis_names: frozenset[str] = frozenset(converted_axes.keys())

        if TYPE_CHECKING:
            self.x: torch.Tensor
            self.y: torch.Tensor
            self.W: torch.Tensor  # legacy
            self.H: torch.Tensor  # legacy
            self.wavelength: torch.Tensor

    def _clear_caches(self) -> None:
        """Clear all cached method results.

        Since axes are immutable after ``__init__``, caches never go stale
        during normal use. This method exists only for testing.
        """
        self._cache_axis_sizes.clear()
        self._cache_cast_info.clear()

    ###########################################################################
    # Initializers
    ###########################################################################

    @classmethod
    def from_ranges(
        cls,
        *,
        x_range: tuple[float, float],
        x_points: int,
        y_range: tuple[float, float],
        y_points: int,
        wavelength: torch.Tensor | float,
        **additional_axes: torch.Tensor | float,
    ) -> Self:
        """
        Create SimulationParameters from coordinate ranges.

        Parameters
        ----------
        x_range : tuple[float, float]
            (min, max) range for x-axis. Use `ureg` for units.
        x_points : int
            Number of points along x-axis.
        y_range : tuple[float, float]
            (min, max) range for y-axis. Use `ureg` for units.
        y_points : int
            Number of points along y-axis.
        wavelength : torch.Tensor | float
            Optical wavelength. Use `ureg` for units.
        **additional_axes : torch.Tensor | float
            Additional axes.

        Examples
        --------
        >>> from svetlanna.units import ureg
        >>> params = SimulationParameters.from_ranges(
        ...     x_range=(-1*ureg.mm, 1*ureg.mm), x_points=256,
        ...     y_range=(-1*ureg.mm, 1*ureg.mm), y_points=256,
        ...     wavelength=632.8*ureg.nm
        ... )
        """
        return cls(
            x=torch.linspace(x_range[0], x_range[1], x_points),
            y=torch.linspace(y_range[0], y_range[1], y_points),
            wavelength=wavelength,
            **additional_axes,
        )

    @classmethod
    def from_dict(
        cls,
        axes_dict: Mapping[str, torch.Tensor | float],
    ) -> Self:
        """
        Create SimulationParameters from a dictionary.

        Parameters
        ----------
        axes_dict : Mapping[str, torch.Tensor | float]
            Dictionary with axis names as keys and tensor/scalar values.
        """
        return cls(dict(axes_dict))

    def clone(self) -> "SimulationParameters":
        """
        Create a deep copy of the SimulationParameters instance.

        Returns
        -------
        SimulationParameters
            A new instance with cloned axes.
        """
        # _buffers is an OrderedDict — preserves axis insertion order
        cloned_axes = {
            name: buf.clone() for name, buf in self._buffers.items() if buf is not None
        }
        return SimulationParameters(cloned_axes)

    ###########################################################################
    # Equality
    ###########################################################################

    def equal(self, value: SimulationParameters) -> bool:
        """Check equality with another SimulationParameters instance.
        The comparison between tensor axes is based on `torch.equal`,
        see [documentation](https://docs.pytorch.org/docs/2.10/generated/torch.equal.html) for more details.
        Comparing instances on diffrent devices will raise `RuntimeError` because `torch.equal` requires tensors to be on the same device.

        Parameters
        ----------
        value : SimulationParameters
            SimulationParameters instance to compare with.

        Returns
        -------
        bool
            `True` if all axes are equal, `False` otherwise.
        """

        if self._all_axis_names != value._all_axis_names:
            return False

        # Axis ordering matters — different order means incompatible tensor layouts
        if self._non_scalar_names != value._non_scalar_names:
            return False

        for name in self._all_axis_names:
            if not torch.equal(getattr(self, name), getattr(value, name)):
                return False

        return True

    ###########################################################################
    # Axes related properties and methods
    ###########################################################################

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Get names of non-scalar axes (those with length > 1)."""
        return self._non_scalar_names

    @property
    def _axis_names_scalar(self) -> tuple[str, ...]:
        """Get names of scalar (0-dimensional) axes."""
        return self._scalar_names

    def __getattr__(self, name: str) -> Any:
        """Get axis value by name, with legacy W/H remapping."""
        resolved = legacy_axis_support(name)
        if resolved != name:
            return super().__getattr__(resolved)
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Block writes to axis names, delegate rest to nn.Module."""
        if (
            hasattr(self, "_all_axis_names")
            and legacy_axis_support(name) in self._all_axis_names
        ):
            warnings.warn(f"Axis '{name}' is read-only")
            return
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """Block deletion of axis attributes."""
        resolved = legacy_axis_support(name)
        if hasattr(self, "_all_axis_names") and resolved in self._all_axis_names:
            raise AttributeError(f"Cannot delete axis '{name}': axes are read-only")
        super().__delattr__(resolved)

    def __contains__(self, name: str) -> bool:
        """Check if an axis exists using 'in' operator."""
        return name in self._all_axis_names

    def __getitem__(self, name: str) -> torch.Tensor:
        """Get axis by name using bracket notation."""
        name = legacy_axis_support(name)
        if name in self._all_axis_names:
            return getattr(self, name)
        raise AxisNotFound(f"Axis '{name}' does not exist")

    def __setitem__(self, name: str, value: Any) -> None:
        """Prevent modification of axes via bracket notation."""
        raise RuntimeError(f"Axis '{name}' is read-only")

    ###########################################################################
    # Utils
    ###########################################################################

    def meshgrid(self, x_axis: str, y_axis: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create a coordinate meshgrid from two axes.

        Parameters
        ----------
        x_axis : str
            Name of the axis for x-coordinates (typically 'x').
        y_axis : str
            Name of the axis for y-coordinates (typically 'y').

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            2D coordinate grids with 'xy' indexing convention.

        Examples
        --------
        ```python
        import svetlanna as sv
        import torch

        sim_params = sv.SimulationParameters(
            x=torch.linspace(-0.5, 0.5, 10),
            y=torch.linspace(-0.5, 0.5, 12),
            wavelength=1,
        )

        X, Y = sim_params.meshgrid("x", "y")
        print(X.shape)  # torch.Size([12, 10])
        ```
        """

        x_axis = legacy_axis_support(x_axis)
        y_axis = legacy_axis_support(y_axis)

        missing = [ax for ax in [x_axis, y_axis] if ax not in self._all_axis_names]
        if missing:
            raise AxisNotFound(f"Axes not found: {', '.join(missing)}")

        x = self[x_axis]
        y = self[y_axis]

        # Handle scalar axes by unsqueezing to 1D
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)

        X, Y = torch.meshgrid(x, y, indexing="xy")
        return X, Y

    def axis_sizes(self, axs: tuple[str, ...] | None = None) -> torch.Size:
        """
        Get the size of specified axes in order (cached for performance).

        Parameters
        ----------
        axs : tuple[str, ...] | None
            Tuple of axis names in the desired order.

        Returns
        -------
        torch.Size
            Size object with lengths of specified axes.

        Examples
        --------
        ```python
        import svetlanna as sv
        import torch

        sim_params = sv.SimulationParameters(
            x=torch.linspace(-0.5, 0.5, 10),
            y=torch.linspace(-0.5, 0.5, 12),
            wavelength=1,
        )

        print(sim_params.axis_sizes(('y', 'x')))  # torch.Size([12, 10])
        ```
        """

        if axs is None:
            axs = self._non_scalar_names

        cached = self._cache_axis_sizes.get(axs)
        if cached is not None:
            return cached

        sizes = []
        for axis in axs:

            axis_tensor = self[axis]
            axis_len = len(axis_tensor) if axis_tensor.dim() > 0 else 1

            sizes.append(axis_len)

        result = torch.Size(sizes)
        self._cache_axis_sizes[axs] = result
        return result

    def index(self, name: str) -> int:
        """
        Get the negative index of an axis in tensors.

        Parameters
        ----------
        name : str
            Name of the axis.

        Returns
        -------
        int
            Negative index for use in tensor operations.

        Raises
        ------
        AxisNotFound
            If the axis doesn't exist or is scalar.
        """
        name = legacy_axis_support(name)
        if name in self._non_scalar_names:
            return -self._non_scalar_names_insertion_order.index(name) - 1
        raise AxisNotFound(f"Axis '{name}' does not exist or is scalar")

    ###########################################################################
    # Casting and reordering
    ###########################################################################

    def _cast_info(
        self, axes: tuple[str, ...]
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        """
        Cached helper for cast(). Returns (tensor_axes, tensor_sizes).
        tensor_axes is  tuple of (axis_name),
        tensor_sizes is tuple of (axis_size).
        """
        cached = self._cache_cast_info.get(axes)
        if cached is not None:
            return cached

        tensor_axes: list[str] = []
        tensor_sizes: list[int] = []

        for axis_name, axis_size in zip(axes, self.axis_sizes(axes)):
            # Skip scalar axes - they don't correspond to tensor dimensions
            if axis_name in self._axis_names_scalar:
                continue

            tensor_axes.append(axis_name)
            tensor_sizes.append(axis_size)

        result = tuple(tensor_axes), tuple(tensor_sizes)
        self._cache_cast_info[axes] = result
        return result

    def cast(
        self, tensor: torch.Tensor, *axes: str, shape_check: bool = True
    ) -> torch.Tensor:
        """
        Cast tensor to match simulation parameters axes for broadcasting.

        Reshapes tensor so it can be broadcast with wavefront tensors.
        Scalar axes are skipped (they don't affect tensor shape).

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor whose trailing dimensions correspond to `axes`.
        *axes : str
            Axes names corresponding to tensor's trailing dimensions.

        Returns
        -------
        torch.Tensor
            Tensor reshaped for broadcasting with wavefront.

        Examples
        --------
        ```python
        import svetlanna as sv
        import torch

        sim_params = sv.SimulationParameters(
            x=torch.linspace(-0.5, 0.5, 3),
            y=torch.linspace(-0.5, 0.5, 2),
            wavelength=torch.linspace(1, 2, 5),
        )
        # axes: (wavelength, y, x)
        print(sim_params.axis_sizes(("wavelength", "y", "x")))  # torch.Size([5, 2, 3])

        a = torch.rand(2, 3)  # y, x
        a = sim_params.cast(a, "y", "x")
        print(a.shape)  # torch.Size([1, 2, 3])
        # a is now ready to broadcast with tensor of shape (5, 2, 3)
        ```
        """
        # avoid circular import
        from svetlanna.axes_math import cast_tensor

        # Get cached preprocessing
        tensor_axes, tensor_sizes = self._cast_info(axes)

        if shape_check and tuple(tensor.shape) != tensor_sizes:
            raise ValueError(
                f"Tensor has shape {tuple(tensor.shape)}, "
                f"expected shape {tensor_sizes} for axes {tensor_axes}."
            )

        return cast_tensor(tensor, tensor_axes, self.axis_names)

    ###########################################################################
    # Device management — nn.Module.to() handles buffer transfer automatically
    ###########################################################################

    @property
    def device(self) -> torch.device:
        """Get the device where all axes are stored."""
        x = self.x
        return x.device

    ###########################################################################
    # Sugar
    ###########################################################################

    def __dir__(self) -> list[str]:
        """List all available attributes including axes for autocompletion."""
        attrs = list(self._all_axis_names)
        attrs.extend(super().__dir__())
        return attrs

    def __repr__(self) -> str:
        """Concise string representation showing all axes."""
        axes_info = []
        for name in sorted(self._all_axis_names):
            tensor = getattr(self, name)
            if tensor.dim() == 0:
                # Scalar: show value
                value = tensor.item()
                axes_info.append(f"{name}={value:.3g}")
            else:
                # Vector: show shape
                axes_info.append(f"{name}={tuple(tensor.shape)}")

        return f"SimulationParameters({', '.join(axes_info)})"

    ###########################################################################
    # LEGACY SUPPORT
    ###########################################################################
    @property
    @deprecated(
        "axes is deprecated, use SimulationParameters instance directly for axis access instead"
    )
    def axes(self):
        warnings.warn(
            "axes is deprecated, use SimulationParameters instance directly for axis access instead",
            DeprecationWarning,
            stacklevel=1,
        )
        return self

    @property
    @deprecated("names is deprecated, use axis_names instead")
    def names(self) -> tuple[str, ...]:
        warnings.warn(
            "names is deprecated, use axis_names instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.axis_names

    @deprecated("axes_size() is deprecated, use axis_sizes() instead")
    def axes_size(self, *args, **kwargs):
        warnings.warn(
            "axes_size() is deprecated, use axis_sizes() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.axis_sizes(*args, **kwargs)
