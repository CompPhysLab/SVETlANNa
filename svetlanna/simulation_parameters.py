from __future__ import annotations
from typing import Any, TYPE_CHECKING, Self
import torch
import warnings
import functools
from collections.abc import Mapping


class AxisNotFound(Exception):
    """Raised when trying to access a non-existent axis."""

    pass


REQUIRED_AXES = {"x", "y", "wavelength"}


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


class SimulationParameters:
    """
    Simulation parameters.

    Manages coordinate systems and physical parameters for optical simulations.
    Required axes: x, y, wavelength.
    Additional axes can be added dynamically.

    Examples
    --------
    >>> from svetlanna.units import ureg
    >>>
    >>> # Basic setup with units
    >>> params = SimulationParameters(
    ...     x=torch.linspace(-0.5*ureg.mm, 0.5*ureg.mm, 512),
    ...     y=torch.linspace(-0.5*ureg.mm, 0.5*ureg.mm, 512),
    ...     wavelength=632.8*ureg.nm
    ... )
    >>>
    >>> # Convenient constructor
    >>> params = SimulationParameters.from_ranges(
    ...     x_range=(-0.5*ureg.mm, 0.5*ureg.mm), x_points=512,
    ...     y_range=(-0.5*ureg.mm, 0.5*ureg.mm), y_points=512,
    ...     wavelength=632.8*ureg.nm
    ... )
    >>>
    >>> # Add polarization
    >>> params.pol = torch.tensor([1., 0.])  # x-polarized
    """

    def __init__(
        self,
        axes: dict[str, torch.Tensor | float] | None = None,
        *,
        x: torch.Tensor | float | None = None,
        y: torch.Tensor | float | None = None,
        wavelength: torch.Tensor | float | None = None,
        **additional_axes: torch.Tensor | float,
    ) -> None:
        """
        Initialize simulation parameters with coordinate axes.

        Parameters
        ----------
        axes : dict[str, torch.Tensor | float] | None, optional
            Dictionary mapping axis names to values (legacy API).
            Cannot be used together with keyword arguments.
        x : torch.Tensor | float | None, optional
            Width/x-axis coordinates in meters. Required if `axes` not provided.
        y : torch.Tensor | float | None, optional
            Height/y-axis coordinates in meters. Required if `axes` not provided.
        wavelength : torch.Tensor | float | None, optional
            Optical wavelength in meters. Required if `axes` not provided.
        **additional_axes : torch.Tensor | float
            Additional axes (e.g., pol=torch.tensor([1., 0.])).

        Raises
        ------
        ValueError
            If both `axes` and keyword arguments are provided, or if required
            axes are missing, or if axes are on different devices.
        """
        # Handle backward compatibility
        if axes is not None:
            if any(x is not None for x in [x, y, wavelength]) or additional_axes:
                raise ValueError(
                    "Cannot use both 'axes' dict and keyword arguments. "
                    "Use either axes={...} or x=..., y=..., wavelength=..."
                )
            all_axes = dict(axes)
        else:
            # New style initialization
            if any(x is None for x in [x, y, wavelength]):
                raise ValueError(
                    "x, y, and wavelength are required when not using 'axes' dict"
                )
            all_axes = {"x": x, "y": y, "wavelength": wavelength, **additional_axes}

        all_axes = {
            legacy_axis_support(name): value for name, value in all_axes.items()
        }

        # Check required axes presence
        if not all(name in all_axes.keys() for name in REQUIRED_AXES):
            missing = REQUIRED_AXES - set(all_axes.keys())
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

        self.__names_inversed = tuple(non_scalar_names)
        self.__names = tuple(reversed(non_scalar_names))
        self.__scalar_names = tuple(scalar_names)
        self.__axes_dict = converted_axes
        self.__device = device

        # Lazy initialization and caching
        self._clear_caches()

        if TYPE_CHECKING:
            self.x: torch.Tensor
            self.y: torch.Tensor
            self.W: torch.Tensor  # legacy
            self.H: torch.Tensor  # legacy
            self.wavelength: torch.Tensor

    def _clear_caches(self) -> None:
        """Clear all cached method results when axes change."""
        # Clear LRU caches
        if hasattr(self.axes_size, "cache_clear"):
            self.axes_size.cache_clear()
        if hasattr(self._cast_info, "cache_clear"):
            self._cast_info.cache_clear()
        # Reset lazy axes

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
        wavelength: float,
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
        wavelength : float
            Optical wavelength. Use `ureg` for units.
        **additional_axes : torch.Tensor | float
            Additional axes.

        Examples
        --------
        >>> from svetlanna.units import ureg
        >>> params = SimulationParameters.from_ranges(
        ...     w_range=(-1*ureg.mm, 1*ureg.mm), w_points=256,
        ...     h_range=(-1*ureg.mm, 1*ureg.mm), h_points=256,
        ...     wavelength=632.8*ureg.nm
        ... )
        """
        return cls(
            axes=None,
            x=torch.linspace(x_range[0], x_range[1], x_points),
            y=torch.linspace(y_range[0], y_range[1], y_points),
            wavelength=wavelength,
            **additional_axes,
        )

    @classmethod
    def from_dict(cls, axes_dict: Mapping[str, torch.Tensor | float]) -> Self:
        """
        Create SimulationParameters from a dictionary.

        Parameters
        ----------
        axes_dict : Mapping[str, torch.Tensor | float]
            Dictionary with axis names as keys and tensor/scalar values.
        """
        return cls(axes=dict(axes_dict))

    ###########################################################################
    # Axes related properties and methods
    ###########################################################################

    @property
    def names(self) -> tuple[str, ...]:
        """Get names of non-scalar axes (those with length > 1)."""
        return self.__names

    @property
    def names_scalar(self) -> tuple[str, ...]:
        """Get names of scalar (0-dimensional) axes."""
        return self.__scalar_names

    @property
    def names_additional(self) -> frozenset[str]:
        """Get names of additional (non-required) axes."""
        return frozenset(self.__axes_dict.keys()) - REQUIRED_AXES

    def __getattribute__(self, name: str) -> torch.Tensor:
        """Get axis value by name using attribute syntax."""
        # Avoid infinite recursion for private attributes
        if name == "_SimulationParameters__axes_dict":
            # Avoid infinite recursion for private attributes
            return super().__getattribute__(name)

        name = legacy_axis_support(name)

        if (value := self.__axes_dict.get(name)) is not None:
            return value

        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set axis value by name using attribute syntax."""
        if hasattr(self, "_SimulationParameters__axes_dict"):
            # __setattr__ is called during __init__ before __axes_dict exists
            name = legacy_axis_support(name)
            if name in self.__axes_dict:
                warnings.warn(f"Axis '{name}' is read-only")
                return

        return super().__setattr__(name, value)

    def __contains__(self, name: str) -> bool:
        """Check if an axis exists using 'in' operator."""
        return name in self.__axes_dict

    def __getitem__(self, name: str) -> torch.Tensor:
        """Get axis by name using bracket notation."""

        name = legacy_axis_support(name)

        if name in self.__axes_dict:
            return self.__axes_dict[name]
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
            Name of the axis for x-coordinates (typically 'W').
        y_axis : str
            Name of the axis for y-coordinates (typically 'H').

        Returns
        -------
        x_grid, y_grid : tuple[torch.Tensor, torch.Tensor]
            2D coordinate grids with 'xy' indexing convention.
        """
        # Validate input
        if not isinstance(x_axis, str) or not isinstance(y_axis, str):
            raise TypeError("Axis names must be strings")
        x_axis = legacy_axis_support(x_axis)
        y_axis = legacy_axis_support(y_axis)

        if x_axis not in self.__axes_dict or y_axis not in self.__axes_dict:
            missing = [ax for ax in [x_axis, y_axis] if ax not in self.__axes_dict]
            raise AxisNotFound(f"Axes not found: {missing}")

        x_data = self[x_axis]
        y_data = self[y_axis]

        # Handle scalar axes by unsqueezing to 1D
        if x_data.dim() == 0:
            x_data = x_data.unsqueeze(0)
        if y_data.dim() == 0:
            y_data = y_data.unsqueeze(0)

        # Validate axes are 1D
        if x_data.dim() != 1 or y_data.dim() != 1:
            raise ValueError(
                f"Both axes must be 0- or 1-dimensional. "
                f"Got {x_axis}: {x_data.dim()}D, {y_axis}: {y_data.dim()}D"
            )

        # Only transfer to device if necessary (optimization)
        if x_data.device != self.__device:
            x_data = x_data.to(self.__device)
        if y_data.device != self.__device:
            y_data = y_data.to(self.__device)

        x, y = torch.meshgrid(x_data, y_data, indexing="xy")
        return x, y

    @functools.lru_cache(maxsize=128)
    def axes_size(self, axs: tuple[str, ...] | None = None) -> torch.Size:
        """
        Get the size of specified axes in order (cached for performance).

        Parameters
        ----------
        axs : tuple[str, ...] | None
            Tuple of axis names in the desired order.
            For legacy compatibility, also accepts axs=(...) keyword argument.

        Returns
        -------
        torch.Size
            Size object with lengths of specified axes.

        Examples
        --------
        >>> size = params.axes_size(('y', 'x'))  # New API (cached)
        >>> size = params.axes_size(axs=('y', 'x'))  # Legacy API
        """

        if axs is None:
            axs = self.__names

        sizes = []
        for axis in axs:
            try:
                axis_tensor = self[axis]
                axis_len = len(axis_tensor) if axis_tensor.dim() > 0 else 1
            except (TypeError, AxisNotFound):
                warnings.warn(f"Axis '{axis}' not found. Using size 0.")
                axis_len = 0
            sizes.append(axis_len)

        return torch.Size(sizes)

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
        if name in self.__names:
            return -self.__names_inversed.index(name) - 1
        raise AxisNotFound(f"Axis '{name}' does not exist or is scalar")

    ###########################################################################
    # Casting and reordering
    ###########################################################################

    @functools.lru_cache(maxsize=64)
    def _cast_info(
        self, axes: tuple[str, ...]
    ) -> tuple[tuple[str, ...], tuple[tuple[int, int, str], ...]]:
        """
        Cached helper for cast(). Returns (tensor_axes, validations).

        validations is tuple of (input_offset, expected_shape, axis_name).
        """
        tensor_axes: list[str] = []
        validations: list[tuple[int, int, str]] = []

        for i, axis_name in enumerate(axes):
            # Skip scalar axes - they don't correspond to tensor dimensions
            if axis_name in self.names_scalar:
                continue

            # Check axis exists
            if axis_name not in self.names:
                raise ValueError(
                    f"Axis '{axis_name}' not found in simulation parameters"
                )

            tensor_axes.append(axis_name)

            # Store validation info: offset from end, expected shape, name
            (expected_shape,) = self.axes_size((axis_name,))
            validations.append((i - len(axes), expected_shape, axis_name))

        return tuple(tensor_axes), tuple(validations)

    def cast(self, tensor: torch.Tensor, *axes: str) -> torch.Tensor:
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
        >>> # axes: (wavelength, H, W), shapes: (5, 2, 3)
        >>> a = torch.rand(2, 3)  # H, W
        >>> a = sim_params.cast(a, "H", "W")
        >>> a.shape
        torch.Size([1, 2, 3])  # ready to broadcast with (5, 2, 3)
        """
        # avoid circular import
        from svetlanna.axes_math import cast_tensor

        # Get cached preprocessing
        tensor_axes, validations = self._cast_info(axes)

        # Validate shapes
        for offset, expected_shape, axis_name in validations:
            actual_shape = tensor.shape[offset]
            if expected_shape != actual_shape:
                raise ValueError(
                    f"Axis '{axis_name}' has shape {actual_shape}, "
                    f"expected {expected_shape}"
                )

        return cast_tensor(tensor, tensor_axes, self.names)

    def reorder(self, tensor: torch.Tensor, *trailing_axes: str) -> torch.Tensor:
        """
        Permute tensor so that specified axes are last, in the given order.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor whose trailing dimensions correspond to axes.names.
        *trailing_axes : str
            Axis names that should become the last dimensions (in order).

        Returns
        -------
        torch.Tensor
            Permuted tensor with trailing_axes as the last dimensions.

        Examples
        --------
        >>> # axes: (wavelength, H, W), tensor shape: (batch, wavelength, H, W)
        >>> t = sim_params.reorder(tensor, "H", "W")      # no change
        >>> t = sim_params.reorder(tensor, "W", "H")      # -> (batch, wavelength, W, H)
        >>> t = sim_params.reorder(tensor, "wavelength")  # -> (batch, H, W, wavelength)
        """
        current = self.__names  # physical order in tensor (left to right)
        n_batch = tensor.ndim - len(current)

        trailing_set = set(trailing_axes)
        missing = trailing_set - set(current)
        if missing:
            raise AxisNotFound(f"Axes not found: {missing}")

        other = tuple(n for n in current if n not in trailing_set)
        target = other + trailing_axes

        if current == target:
            return tensor

        perm = [*range(n_batch), *(n_batch + current.index(n) for n in target)]
        return tensor.permute(perm)

    ###########################################################################
    # Device management
    ###########################################################################

    def to(self, device: str | torch.device | int) -> Self:
        """
        Move all axes to a different device (inplace).

        Unlike tensor.to() which returns a new tensor, this method
        mutates the instance inplace (like nn.Module.to()) to ensure
        all Elements sharing this SimulationParameters stay in sync.

        Parameters
        ----------
        device : str | torch.device | int
            Target device.

        Returns
        -------
        Self
            The same instance (for chaining).
        """
        target_device = torch.device(device)
        if self.__device == target_device:
            return self

        # Mutate inplace — all references stay valid
        for name in self.__axes_dict:
            self.__axes_dict[name] = self.__axes_dict[name].to(target_device)

        # Use actual device from tensor (e.g., 'cuda' → 'cuda:0')
        first_tensor = next(iter(self.__axes_dict.values()))
        self.__device = first_tensor.device

        return self

    @property
    def device(self) -> torch.device:
        """Get the device where all axes are stored."""
        return self.__device

    ###########################################################################
    # Sugar
    ###########################################################################

    def __dir__(self) -> list[str]:
        """List all available attributes including axes for autocompletion."""
        attrs = list(self.__axes_dict.keys())
        attrs.extend(super().__dir__())
        return attrs

    def __repr__(self) -> str:
        """Concise string representation showing all axes."""
        axes_info = []
        for name in sorted(self.__axes_dict.keys()):
            tensor = self.__axes_dict[name]
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
    def axes(self):
        return self
