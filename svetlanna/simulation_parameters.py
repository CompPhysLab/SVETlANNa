from typing import Any, Iterable, TYPE_CHECKING
import torch
import warnings


class Axes:
    """Axes storage"""
    def __init__(self, axes: dict[str, torch.Tensor]) -> None:
        # TODO: set default values for the new axis if needed (ex. pol = 0)

        # check if required axes are presented
        required_axes = (
            'W', 'H', 'wavelength'
        )
        if not all(name in axes.keys() for name in required_axes):
            raise ValueError("Axes 'W', 'H', and 'wavelength' are required!")

        # check if W and H axes are 1-d
        if not len(axes['W'].shape) == 1:
            raise ValueError("'W' axis should be 1-dimensional")
        if not len(axes['H'].shape) == 1:
            raise ValueError("'H' axis should be 1-dimensional")

        # check if axes are 0- or 1-dimensional
        non_scalar_names = []
        for axis_name, value in axes.items():
            tensor_dimensionality = len(value.shape)

            if tensor_dimensionality not in (0, 1):
                raise ValueError(f"All axes should be 0- or 1-dimensional tensors. Axis {axis_name} is {tensor_dimensionality}-dimensional")

            if tensor_dimensionality == 1:
                non_scalar_names.append(axis_name)

        self.__axes_dict = axes
        self.__names = tuple(reversed(non_scalar_names))

        if TYPE_CHECKING:
            self.W: torch.Tensor
            self.H: torch.Tensor
            self.wavelength: torch.Tensor

    @property
    def names(self) -> tuple[str, ...]:
        """Non-scalar axes' names"""
        return self.__names

    def __getattribute__(self, name: str) -> Any:

        if name in (f'_Axes{i}' for i in ('__axes_dict', '__names')):
            return super().__getattribute__(name)

        axes = self.__axes_dict

        if name in axes:
            return axes[name]

        return super().__getattribute__(name)

    def __getitem__(self, name: str) -> Any:
        axes = self.__axes_dict
        if name in axes:
            return axes[name]

        raise KeyError(name)

    def __setitem__(self, name: str) -> None:
        raise RuntimeError('Axis can not be changed')

    def __dir__(self) -> Iterable[str]:
        return self.__axes_dict.keys()


class SimulationParameters:
    """
    A class which describes characteristic parameters of the system
    """
    def __init__(
        self,
        axes: dict[str, torch.Tensor | float]
    ) -> None:
        device = None

        def value_to_tensor(x):
            nonlocal device
            if isinstance(x, torch.Tensor):
                if device is None:
                    device = x.device
                if x.device != device:
                    raise ValueError('All axes should be on the same device')
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
                the first dimension corresponds to the cardinality of the second axis (`y_axis`) and
                the second dimension corresponds to the cardinality of the first axis (`x_axis`).
        """
        a, b = torch.meshgrid(
            self.axes[x_axis], self.axes[y_axis],
            indexing='xy'
        )
        return a.to(self.__device), b.to(self.__device)

    def axes_size(self, axs: str | Iterable[str]) -> torch.Size:
        """
        Returns a size of axes in specified order.

        Parameters
        ----------
        axs : str | Iterable[str]
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
            except KeyError:  # axis not in self.__axes_dict.keys()
                warnings.warn(
                    message=f"There is no '{axis}' in axes! Zero returned as a dimension for '{axis}'-axis."
                )
                axis_len = 0

            sizes.append(axis_len)

        return torch.Size(sizes)

    def to(self, device: str | torch.device | int) -> 'SimulationParameters':
        if self.__device == torch.device(device):
            return self

        new_axes_dict = {}
        for axis_name in self.__axes_dict.keys():
            new_axes_dict[axis_name] = self.__axes_dict[axis_name].to(device=device)
        return SimulationParameters(axes=new_axes_dict)

    @property
    def device(self) -> str | torch.device | int:
        return self.__device

    # def check_wf(self, wf: 'Wavefront'):
    #     # TODO: check if wf has a right dimensionality
    #     ...
