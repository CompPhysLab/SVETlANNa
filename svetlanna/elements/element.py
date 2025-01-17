from abc import ABCMeta, abstractmethod
from torch import nn
from torch import Tensor
from ..simulation_parameters import SimulationParameters
from ..specs import ReprRepr, ParameterSpecs
from typing import Iterable, TypeVar
from ..parameters import BoundedParameter, Parameter
from ..wavefront import Wavefront


INNER_PARAMETER_SUFFIX = '_svtlnn_inner_parameter'

_T = TypeVar('_T', bound=Tensor)
_V = TypeVar('_V')


# TODO: check docstring
class Element(nn.Module, metaclass=ABCMeta):
    """A class that describes each element of the system

    Parameters
    ----------
    nn : _type_
        _description_
    metaclass : _type_, optional
        _description_, by default ABCMeta
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters
    ) -> None:
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Class exemplar that describes the optical system
        """

        super().__init__()

        self.simulation_parameters = simulation_parameters

        self._x_nodes = self.simulation_parameters.axes.W.shape[0]
        self._y_nodes = self.simulation_parameters.axes.H.shape[0]
        self._wavelength = self.simulation_parameters.axes.wavelength

        self._x_linspace = self.simulation_parameters.axes.W
        self._y_linspace = self.simulation_parameters.axes.H

        self._x_grid, self._y_grid = self.simulation_parameters.meshgrid(x_axis='W', y_axis='H')    # noqa: E501

    # TODO: check doctrings
    @abstractmethod
    def forward(self, input_field: Wavefront) -> Wavefront:

        """Forward propagation through the optical element"""

    def to_specs(self) -> Iterable[ParameterSpecs]:

        """Create specs"""

        for (name, parameter) in self.named_parameters():

            # BoundedParameter and Parameter support
            if name.endswith(INNER_PARAMETER_SUFFIX):
                name = name.removesuffix(INNER_PARAMETER_SUFFIX)
                parameter = self.__getattribute__(name)

            yield ParameterSpecs(
                parameter_name=name,
                representations=(ReprRepr(value=parameter),)
            )

    # TODO: create docstrings
    def __setattr__(self, name: str, value: Tensor | nn.Module) -> None:

        # BoundedParameter and Parameter are handled by pointing
        # auxiliary attribute on them with a name plus INNER_PARAMETER_SUFFIX
        if isinstance(value, (BoundedParameter, Parameter)):
            super().__setattr__(
                name + INNER_PARAMETER_SUFFIX, value.inner_storage
            )

        return super().__setattr__(name, value)

    def make_buffer(
        self,
        name: str,
        value: _T,
        persistent: bool = False
    ) -> _T:
        """Make buffer for internal use.

        Use case:
        ```
        self.mask = make_buffer('mask', some_tensor)
        ```
        This allow torch to properly process `.to` method on Element
        by marking that `mask` should be transferred to required device.

        Parameters
        ----------
        name : str
            name of the new buffer
            (it is more convenient to use name of new attribute)
        value : _T
            tensor to be buffered
        persistent : bool, optional
            see torch docs on buffers, by default False

        Returns
        -------
        _T
            the value passed to the method
        """
        self.register_buffer(
            name, value, persistent=persistent
        )
        return self.__getattr__(name)

    def process_parameter(
        self,
        name: str,
        value: _V
    ) -> _V:
        """Process element parameter passed by user.
        Automatically registers buffer for non-parametric tensors.

        Use case:
        ```
        self.mask = process_parameter('mask', some_tensor)
        ```

        Parameters
        ----------
        name : str
            name of the new buffer
            (it is more convenient to use name of new attribute)
        value : _V
            the value of the element parameter

        Returns
        -------
        _V
            the value passed to the method
        """
        if isinstance(value, (nn.Parameter, Parameter)):
            return value
        if isinstance(value, Tensor):
            return self.make_buffer(name, value, persistent=True)
        return value
