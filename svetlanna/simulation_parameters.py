from typing import Any, Iterable, TYPE_CHECKING
import torch
import warnings


class AxisNotFound(Exception):
    """
    Raised when an axis is requested that does not exist."""

    pass


_AXES_INNER_ATTRS = tuple(
    f"_Axes{i}" for i in ("__axes_dict", "__names", "__names_inversed")
)


class Axes:
    """Axes storage"""

    def __init__(self, axes: dict[str, torch.Tensor]) -> None:
        """
        Initializes the AxisInfo object with a dictionary of axes.

            Args:
                axes: A dictionary where keys are axis names (e.g., 'W', 'H') and
                    values are PyTorch tensors representing the axis values.  Must contain
                    'W', 'H', and 'wavelength'.

            Returns:
                None
        """
        # TODO: set default values for the new axis if needed (ex. pol = 0)

        # check if required axes are presented
        required_axes = ("W", "H", "wavelength")
        if not all(name in axes.keys() for name in required_axes):
            raise ValueError("Axes 'W', 'H', and 'wavelength' are required!")

        # check if W and H axes are 1-d
        if not len(axes["W"].shape) == 1:
            raise ValueError("'W' axis should be 1-dimensional")
        if not len(axes["H"].shape) == 1:
            raise ValueError("'H' axis should be 1-dimensional")

        # check if axes are 0- or 1-dimensional
        non_scalar_names = []
        for axis_name, value in axes.items():
            tensor_dimensionality = len(value.shape)

            if tensor_dimensionality not in (0, 1):
                raise ValueError(
                    "All axes should be 0- or 1-dimensional tensors. "
                    f"Axis {axis_name} is {tensor_dimensionality}-dimensional"
                )

            if tensor_dimensionality == 1:
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
        """Non-scalar axes' names"""
        return self.__names

    def index(self, name: str) -> int:
        """Index of specific axis in the tensor.
        The index is negative.

        Parameters
        ----------
        name : str
            name of the axis

        Returns
        -------
        int
            index of the axis
        """
        if name in self.__names:
            return -self.__names_inversed.index(name) - 1
        raise AxisNotFound(f"Axis with name {name} does not exist.")

    def __getattribute__(self, name: str) -> Any:
        """
        Retrieves an attribute from the object.

            This method intercepts attribute access and first checks for inner attributes
            defined in _AXES_INNER_ATTRS. If not found there, it looks within the internal
            axes dictionary (__axes_dict).  If still not found, it falls back to the
            default attribute retrieval behavior of the superclass.

            Args:
                name: The name of the attribute to retrieve.

            Returns:
                The value of the attribute if found, otherwise the result of the
                superclass's __getattribute__ method.
        """

        if name in _AXES_INNER_ATTRS:
            return super().__getattribute__(name)

        axes = self.__axes_dict

        if name in axes:
            return axes[name]

        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets an attribute on the object.

            This method intercepts attribute assignments to handle special cases for
            inner attributes and axes that already exist. It issues a warning if an
            existing axis is being reassigned without modification.

            Args:
                name: The name of the attribute to set.
                value: The value to assign to the attribute.

            Returns:
                None
        """

        if name in _AXES_INNER_ATTRS:
            return super().__setattr__(name, value)

        if name in self.__axes_dict:
            warnings.warn(f"Axis {name} has not been changed")

        return super().__setattr__(name, value)

    def __getitem__(self, name: str) -> Any:
        """
        Retrieves an axis by its name.

          Args:
            name: The name of the axis to retrieve.

          Returns:
            The axis object associated with the given name.

          Raises:
            AxisNotFound: If no axis with the specified name exists.
        """
        axes = self.__axes_dict
        if name in axes:
            return axes[name]

        raise AxisNotFound(f"Axis with name {name} does not exist.")

    def __setitem__(self, name: str, value: Any) -> None:
        """
        Sets an item in the axis.

          This method is overridden to prevent modification of the axis after creation.

          Args:
            name: The name of the item to set.
            value: The value to assign to the item.

          Returns:
            None

          Raises:
            RuntimeError: Always raised, as axis items cannot be changed.
        """
        raise RuntimeError("Axis can not be changed")

    def __dir__(self) -> Iterable[str]:
        """
        Returns the names of the axes in the coordinate system.

          Args:
            None

          Returns:
            Iterable[str]: An iterable of strings representing the axis names.
        """
        return self.__axes_dict.keys()


class SimulationParameters:
    """
    A class which describes characteristic parameters of the system
    """

    def __init__(self, axes: dict[str, torch.Tensor | float]) -> None:
        """
        Initializes the object with a dictionary of axes.

            Args:
                axes: A dictionary where keys are axis names (strings) and values are either
                    PyTorch tensors or floats representing the axis values.  All tensor values
                    must be on the same device.

            Returns:
                None
        """
        device = None

        def value_to_tensor(x):
            nonlocal device
            if isinstance(x, torch.Tensor):
                if device is None:
                    device = x.device
                if x.device != device:
                    raise ValueError("All axes should be on the same device")
                return x
            return torch.tensor(x)

        # create a copy of the dict
        self.__axes_dict = {
            name: value_to_tensor(value) for name, value in axes.items()
        }

        if device is None:
            device = torch.get_default_device()

        self.__device = device
        self.to(device=device)

        self.axes = Axes(self.__axes_dict)

    def __getitem__(self, axis: str) -> torch.Tensor:
        """
        Returns the tensor associated with a given axis.

          Args:
            axis: The name of the axis to retrieve.

          Returns:
            torch.Tensor: The tensor corresponding to the specified axis.
        """
        return self.axes[axis]

    def meshgrid(self, x_axis: str, y_axis: str):
        """
        Returns a meshgrid for a selected pair of axes.
        ...

        Parameters
        ----------
        x_axis, y_axis : str
            Axis names to compose a meshgrid.

        Returns
        -------
        x_grid, y_grid: torch.Tensor
            A torch.meshgrid of selected axis.
            Comment: indexing='xy'
                the first dimension corresponds to the cardinality
                of the second axis (`y_axis`) and the second dimension
                corresponds to the cardinality of the first axis (`x_axis`).
        """
        a, b = torch.meshgrid(self.axes[x_axis], self.axes[y_axis], indexing="xy")
        return a.to(self.__device), b.to(self.__device)

    def axes_size(self, axs: Iterable[str]) -> torch.Size:
        """
        Returns a size of axes in specified order.

        Parameters
        ----------
        axs : Iterable[str]
            An order of axis.

        Returns
        -------
        torch.Size()
            Size of axes in a specified order.
        """
        sizes = []
        for axis in axs:

            try:
                axis_len = len(self.axes[axis])
            except TypeError:  # float has no len()
                axis_len = 1
            except AxisNotFound:  # axis not in self.__axes_dict.keys()
                warnings.warn(
                    f"There is no '{axis}' in axes! "
                    f"Zero returned as a dimension for '{axis}'-axis."
                )
                axis_len = 0

            sizes.append(axis_len)

        return torch.Size(sizes)

    def to(self, device: str | torch.device | int) -> "SimulationParameters":
        if self.__device == torch.device(device):
            return self

        new_axes_dict = {}
        for axis_name, axis in self.__axes_dict.items():
            new_axes_dict[axis_name] = axis.to(device=device)
        return SimulationParameters(axes=new_axes_dict)

    @property
    def device(self) -> str | torch.device | int:
        """
        Returns the device on which tensors are allocated.

          Args:
            None

          Returns:
            The device string, torch.device object or integer representing the device.
        """
        return self.__device

    # def check_wf(self, wf: 'Wavefront'):
    #     # TODO: check if wf has a right dimensionality
    #     ...
