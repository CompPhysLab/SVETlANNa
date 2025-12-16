from __future__ import annotations
from typing import Any, Iterable, TYPE_CHECKING, Self
import torch
import warnings
import functools
from collections.abc import Mapping


class AxisNotFound(Exception):
    """Raised when trying to access a non-existent axis."""

    pass


_AXES_INNER_ATTRS = tuple(
    f"_Axes{i}" for i in ("__axes_dict", "__names", "__names_inversed")
)


class Axes:
    """
    Storage for simulation axes with dynamic access support.

    Manages coordinate systems and parameter axes for optical simulations.
    Must contain required axes: 'W' (width/x), 'H' (height/y), and 'wavelength'.

    Examples
    --------
    >>> import torch
    >>> from svetlanna.units import ureg
    >>> axes_dict = {
    ...     'W': torch.linspace(-1*ureg.mm, 1*ureg.mm, 100),
    ...     'H': torch.linspace(-1*ureg.mm, 1*ureg.mm, 100),
    ...     'wavelength': torch.tensor(632.8*ureg.nm)
    ... }
    >>> axes = Axes(axes_dict)
    >>> print(axes.W.shape)  # torch.Size([100])
    """

    def __init__(self, axes: dict[str, torch.Tensor]) -> None:
        # Validate required axes presence
        required_axes = ("W", "H", "wavelength")
        if not all(name in axes.keys() for name in required_axes):
            missing = set(required_axes) - set(axes.keys())
            raise ValueError(
                f"Missing required axes: {missing}. " f"Required: {required_axes}"
            )

        # Validate W and H are 1-dimensional
        for ax_name in ("W", "H"):
            if axes[ax_name].dim() != 1:
                raise ValueError(
                    f"Axis '{ax_name}' must be 1-dimensional, "
                    f"got {axes[ax_name].dim()}-dimensional"
                )

        # Validate all axes are 0- or 1-dimensional
        non_scalar_names = []
        for axis_name, value in axes.items():
            dim = value.dim()
            if dim not in (0, 1):
                raise ValueError(
                    f"Axis '{axis_name}' must be 0- or 1-dimensional, "
                    f"got {dim}-dimensional"
                )
            if dim == 1:
                non_scalar_names.append(axis_name)

        self.__axes_dict = axes
        self.__names_inversed = tuple(non_scalar_names)
        self.__names = tuple(reversed(non_scalar_names))

        if TYPE_CHECKING:
            self.W: torch.Tensor
            self.H: torch.Tensor
            self.wavelength: torch.Tensor

    @property
    def names(self) -> tuple[str, ...]:
        """Get names of non-scalar axes (those with length > 1)."""
        return self.__names

    @property
    def scalar_names(self) -> tuple[str, ...]:
        """Get names of scalar (0-dimensional) axes."""
        return tuple(
            name for name, value in self.__axes_dict.items() if value.dim() == 0
        )

    @property
    def shapes(self) -> tuple[int, ...]:
        """Get shapes of non-scalar axes in physical (tensor) order."""
        return tuple(len(self.__axes_dict[name]) for name in self.__names_inversed)

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
        if name in self.__names:
            return -self.__names_inversed.index(name) - 1
        raise AxisNotFound(f"Axis '{name}' does not exist or is scalar")

    def ensure_order(self, tensor: torch.Tensor, *trailing_axes: str) -> torch.Tensor:
        """
        Permute tensor so that specified axes are last, in the given order.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor whose trailing dimensions correspond to self.names.
        *trailing_axes : str
            Axis names that should become the last dimensions (in order).

        Returns
        -------
        torch.Tensor
            Permuted tensor with trailing_axes as the last dimensions.

        Examples
        --------
        If axes are (wavelength, H, W) and tensor has shape (batch, wavelength, H, W):

        >>> axes.ensure_order(tensor, 'H', 'W')      # no change, already (..., H, W)
        >>> axes.ensure_order(tensor, 'W', 'H')      # -> (batch, wavelength, W, H)
        >>> axes.ensure_order(tensor, 'wavelength')  # -> (batch, H, W, wavelength)
        """
        current = self.__names_inversed  # physical order in tensor
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

    def __getattribute__(self, name: str) -> Any:
        if name in _AXES_INNER_ATTRS:
            return super().__getattribute__(name)

        axes = self.__axes_dict
        if name in axes:
            return axes[name]

        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _AXES_INNER_ATTRS:
            return super().__setattr__(name, value)

        if hasattr(self, "_Axes__axes_dict") and name in self.__axes_dict:
            warnings.warn(f"Axis '{name}' is read-only and cannot be modified")

        return super().__setattr__(name, value)

    def __getitem__(self, name: str) -> torch.Tensor:
        """Get axis by name using bracket notation."""
        if name in self.__axes_dict:
            return self.__axes_dict[name]
        raise AxisNotFound(f"Axis '{name}' does not exist")

    def __setitem__(self, name: str, value: Any) -> None:
        """Prevent modification of axes via bracket notation."""
        raise RuntimeError(
            "Axes are read-only. Use SimulationParameters.add_axis() instead"
        )

    def __dir__(self) -> Iterable[str]:
        """List all available axis names."""
        return self.__axes_dict.keys()


class SimulationParameters:
    """
    Simulation parameters.

    Manages coordinate systems and physical parameters for optical simulations.
    Required axes: W (width/x), H (height/y), wavelength.
    Additional axes can be added dynamically.

    Examples
    --------
    >>> from svetlanna.units import ureg
    >>>
    >>> # Basic setup with units
    >>> params = SimulationParameters(
    ...     W=torch.linspace(-0.5*ureg.mm, 0.5*ureg.mm, 512),
    ...     H=torch.linspace(-0.5*ureg.mm, 0.5*ureg.mm, 512),
    ...     wavelength=632.8*ureg.nm
    ... )
    >>>
    >>> # Convenient constructor
    >>> params = SimulationParameters.from_ranges(
    ...     w_range=(-0.5*ureg.mm, 0.5*ureg.mm), w_points=512,
    ...     h_range=(-0.5*ureg.mm, 0.5*ureg.mm), h_points=512,
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
        W: torch.Tensor | float | None = None,
        H: torch.Tensor | float | None = None,
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
        W : torch.Tensor | float | None, optional
            Width/x-axis coordinates in meters. Required if `axes` not provided.
        H : torch.Tensor | float | None, optional
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
            if any(x is not None for x in [W, H, wavelength]) or additional_axes:
                raise ValueError(
                    "Cannot use both 'axes' dict and keyword arguments. "
                    "Use either axes={...} or W=..., H=..., wavelength=..."
                )
            all_axes = dict(axes)
        else:
            # New style initialization
            if any(x is None for x in [W, H, wavelength]):
                raise ValueError(
                    "W, H, and wavelength are required when not using 'axes' dict"
                )
            all_axes = {"W": W, "H": H, "wavelength": wavelength, **additional_axes}

        # Convert all values to tensors and ensure device consistency
        device = None
        converted_axes = {}

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
                # Convert scalar to tensor
                tensor_value = torch.tensor(float(value))
                if device is not None:
                    tensor_value = tensor_value.to(device)
                converted_axes[name] = tensor_value

        # Set default device if none specified
        if device is None:
            device = torch.get_default_device()
            # Move all tensors to default device
            converted_axes = {
                name: tensor.to(device) for name, tensor in converted_axes.items()
            }

        self.__axes_dict = converted_axes
        self.__device = device
        self._frozen = False

        # Lazy initialization and caching
        self._axes = None
        self._clear_caches()

    @property
    def axes(self) -> Axes:
        """
        Get axes object with convenient access methods.

        The Axes object provides attribute-style access to individual axes
        and methods for working with the coordinate system.
        """
        if self._axes is None:
            self._axes = Axes(self.__axes_dict)
        return self._axes

    def _clear_caches(self) -> None:
        """Clear all cached method results when axes change."""
        # Clear LRU cache
        if hasattr(self.axes_size, "cache_clear"):
            self.axes_size.cache_clear()
        # Reset lazy axes
        self._axes = None

    def freeze(self) -> Self:
        """
        Freeze the simulation parameters to prevent modification.

        After freezing, add_axis(), remove_axis(), and attribute assignment
        for new axes will raise RuntimeError. This is useful after creating
        Elements that cache tensors based on current axes.

        Returns
        -------
        Self
            The same instance (for chaining).

        Examples
        --------
        >>> params = SimulationParameters(W=..., H=..., wavelength=...)
        >>> params.freeze()
        >>> params.pol = torch.tensor([1, 0])  # raises RuntimeError
        """
        self._frozen = True
        return self

    @property
    def frozen(self) -> bool:
        """Check if the simulation parameters are frozen."""
        return self._frozen

    def _check_frozen(self, operation: str) -> None:
        """Raise if frozen."""
        if self._frozen:
            raise RuntimeError(
                f"Cannot {operation}: SimulationParameters is frozen. "
                "Create a new instance or call copy() first."
            )

    @classmethod
    def from_ranges(
        cls,
        *,
        w_range: tuple[float, float],
        w_points: int,
        h_range: tuple[float, float],
        h_points: int,
        wavelength: float,
        **additional_axes: torch.Tensor | float,
    ) -> Self:
        """
        Create SimulationParameters from coordinate ranges.

        Parameters
        ----------
        w_range : tuple[float, float]
            (min, max) range for W-axis. Use `ureg` for units.
        w_points : int
            Number of points along W-axis.
        h_range : tuple[float, float]
            (min, max) range for H-axis. Use `ureg` for units.
        h_points : int
            Number of points along H-axis.
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
            W=torch.linspace(w_range[0], w_range[1], w_points),
            H=torch.linspace(h_range[0], h_range[1], h_points),
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

    def add_axis(self, name: str, values: torch.Tensor | float) -> None:
        """
        Add a new axis dynamically.

        Parameters
        ----------
        name : str
            Name of the new axis. Should be descriptive (e.g., 'pol', 'time').
        values : torch.Tensor | float
            Values for the axis. Scalars become 0-d tensors, arrays become 1-d.

        Raises
        ------
        RuntimeError
            If the instance is frozen.
        ValueError
            If the tensor has more than 1 dimension.
        """
        self._check_frozen(f"add axis '{name}'")

        if not isinstance(name, str) or not name:
            raise ValueError("Axis name must be a non-empty string")

        if name in self.__axes_dict:
            warnings.warn(f"Axis '{name}' already exists and will be replaced")

        # Convert to tensor and move to correct device
        if isinstance(values, torch.Tensor):
            tensor = values.to(self.__device)
        else:
            tensor = torch.tensor(float(values), device=self.__device)

        # Validate dimensionality
        if tensor.dim() > 1:
            raise ValueError(
                f"Axis '{name}' must be 0- or 1-dimensional, "
                f"got {tensor.dim()}-dimensional tensor"
            )

        # Update internal state
        self.__axes_dict[name] = tensor
        self._clear_caches()

    def remove_axis(self, name: str) -> None:
        """
        Remove an axis from the simulation parameters.

        Required axes (W, H, wavelength) cannot be removed.

        Parameters
        ----------
        name : str
            Name of the axis to remove.

        Raises
        ------
        RuntimeError
            If the instance is frozen.
        ValueError
            If trying to remove a required axis.
        AxisNotFound
            If the axis doesn't exist.
        """
        self._check_frozen(f"remove axis '{name}'")

        required_axes = {"W", "H", "wavelength"}
        if name in required_axes:
            raise ValueError(
                f"Cannot remove required axis '{name}'. "
                f"Required axes: {required_axes}"
            )

        if name not in self.__axes_dict:
            raise AxisNotFound(f"Axis '{name}' does not exist")

        del self.__axes_dict[name]
        self._clear_caches()

    @property
    def axis_names(self) -> frozenset[str]:
        """Get names of all axes as a frozen set."""
        return frozenset(self.__axes_dict.keys())

    @property
    def additional_axes(self) -> frozenset[str]:
        """Get names of additional (non-required) axes."""
        required = {"W", "H", "wavelength"}
        return frozenset(self.__axes_dict.keys()) - required

    def __getattr__(self, name: str) -> torch.Tensor:
        """Get axis value by name using attribute syntax."""
        # Avoid infinite recursion for private attributes
        if name.startswith("_") or name in ("axes", "axis_names", "additional_axes"):
            return super().__getattribute__(name)

        if name in self.__axes_dict:
            return self.__axes_dict[name]

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: torch.Tensor | float) -> None:
        """Set axis value by name using attribute syntax."""
        # Handle private attributes and initialization
        if name.startswith("_") or name in ("axes",):
            super().__setattr__(name, value)
            return

        # Check if we're still in initialization
        if not hasattr(self, "_SimulationParameters__axes_dict"):
            super().__setattr__(name, value)
            return

        # Dynamic axis assignment with warning for required axes
        required_axes = {"W", "H", "wavelength"}
        if name in required_axes:
            warnings.warn(
                f"Modifying required axis '{name}' may break compatibility. "
                f"Consider creating a new SimulationParameters instance instead."
            )

        self.add_axis(name, value)

    def __contains__(self, name: str) -> bool:
        """Check if an axis exists using 'in' operator."""
        return name in self.__axes_dict

    def __dir__(self) -> list[str]:
        """List all available attributes including axes for autocompletion."""
        attrs = list(super().__dir__())
        attrs.extend(self.__axes_dict.keys())
        return sorted(set(attrs))

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

    def __getitem__(self, axis: str) -> torch.Tensor:
        """Get axis by name using bracket notation."""
        return self.axes[axis]

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
        if x_axis not in self.__axes_dict or y_axis not in self.__axes_dict:
            missing = [ax for ax in [x_axis, y_axis] if ax not in self.__axes_dict]
            raise AxisNotFound(f"Axes not found: {missing}")

        x_data = self.axes[x_axis]
        y_data = self.axes[y_axis]

        # Validate axes are 1D
        if x_data.dim() != 1 or y_data.dim() != 1:
            raise ValueError(
                f"Both axes must be 1-dimensional. "
                f"Got {x_axis}: {x_data.dim()}D, {y_axis}: {y_data.dim()}D"
            )

        # Only transfer to device if necessary (optimization)
        if x_data.device != self.__device:
            x_data = x_data.to(self.__device)
        if y_data.device != self.__device:
            y_data = y_data.to(self.__device)

        return torch.meshgrid(x_data, y_data, indexing="xy")

    @functools.lru_cache(maxsize=128)
    def axes_size(self, axs: tuple[str, ...] | None = None, **kwargs) -> torch.Size:
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
        >>> size = params.axes_size(('H', 'W'))  # New API (cached)
        >>> size = params.axes_size(axs=('H', 'W'))  # Legacy API
        """
        # Handle both new and legacy API
        if axs is None and "axs" in kwargs:
            axs = kwargs["axs"]
            if not isinstance(axs, tuple):
                axs = tuple(axs)
        elif axs is None:
            raise ValueError("axes must be specified")
        elif not isinstance(axs, tuple):
            axs = tuple(axs)

        sizes = []
        for axis in axs:
            try:
                axis_tensor = self.axes[axis]
                axis_len = len(axis_tensor) if axis_tensor.dim() > 0 else 1
            except (TypeError, AxisNotFound):
                warnings.warn(f"Axis '{axis}' not found. Using size 0.")
                axis_len = 0
            sizes.append(axis_len)

        return torch.Size(sizes)

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

        sim_axes = self.axes

        # Filter out scalar axes and validate shapes
        tensor_axes: list[str] = []
        skipped_scalar_axes: list[str] = []

        for i, axis_name in enumerate(axes):
            # Skip scalar axes - they don't correspond to tensor dimensions
            if axis_name in sim_axes.scalar_names:
                skipped_scalar_axes.append(axis_name)
                continue

            # Check axis exists
            if axis_name not in sim_axes.names:
                raise ValueError(
                    f"Axis '{axis_name}' not found in simulation parameters"
                )

            tensor_axes.append(axis_name)

            # Validate shape matches
            axis_idx = sim_axes.index(axis_name)  # negative index
            expected_shape = sim_axes.shapes[axis_idx]
            actual_shape = tensor.shape[i - len(axes)]

            if expected_shape != actual_shape:
                raise ValueError(
                    f"Axis '{axis_name}' has shape {actual_shape}, "
                    f"expected {expected_shape}. "
                    f"(Skipped scalar axes: {skipped_scalar_axes})"
                )

        return cast_tensor(tensor, tuple(tensor_axes), sim_axes.names)

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
        return self.axes.ensure_order(tensor, *trailing_axes)

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

        self.__device = target_device
        self._clear_caches()
        return self

    @property
    def device(self) -> torch.device:
        """Get the device where all axes are stored."""
        return self.__device

    def copy(self) -> Self:
        """Create a deep copy of this instance."""
        new_axes = {name: tensor.clone() for name, tensor in self.__axes_dict.items()}
        return SimulationParameters(axes=new_axes)
