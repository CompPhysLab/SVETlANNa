from abc import ABCMeta, abstractmethod
from torch import nn
from torch import Tensor
from ..simulation_parameters import SimulationParameters
from ..specs import PrettyReprRepr, ParameterSpecs
from ..specs.specs_writer import write_specs_to_html
from io import StringIO
from typing import Iterable, TypeVar, TYPE_CHECKING
from ..parameters import ConstrainedParameter, Parameter
from ..wavefront import Wavefront


INNER_PARAMETER_SUFFIX = '_svtlnn_inner_parameter'

_T = TypeVar('_T', Tensor, None)
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

    # TODO: check doctrings
    @abstractmethod
    def forward(self, incident_wavefront: Wavefront) -> Wavefront:

        """Forward propagation through the optical element"""

    def to_specs(self) -> Iterable[ParameterSpecs]:

        """Create specs"""

        for (name, parameter) in self.named_parameters():

            yield ParameterSpecs(
                parameter_name=name,
                representations=(PrettyReprRepr(value=parameter),)
            )

    # TODO: create docstrings
    def __setattr__(self, name: str, value: Tensor | nn.Module) -> None:

        # BoundedParameter and Parameter are handled by pointing
        # auxiliary attribute on them with a name plus INNER_PARAMETER_SUFFIX
        if isinstance(value, (ConstrainedParameter, Parameter)):
            super().__setattr__(
                name + INNER_PARAMETER_SUFFIX, value.inner_storage
            )

        return super().__setattr__(name, value)

    def _repr_html_(self) -> str:
        stream = StringIO('')
        write_specs_to_html(self, 0, '', stream)
        return stream.getvalue()

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

        if value is not None:
            if value.device != self.simulation_parameters.device:
                raise ValueError(
                    f"Tensor to be buffered as {name} must be on "
                    "the simulation parameters device."
                )

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
        if isinstance(value, Tensor):
            if value.device != self.simulation_parameters.device:
                raise ValueError(
                    f"Parameter {name} must be on "
                    "the simulation parameters device."
                )
        if isinstance(value, (nn.Parameter, Parameter)):
            return value
        if isinstance(value, Tensor):
            return self.make_buffer(name, value, persistent=True)
        return value

    # === methods below are added for typing only ===

    if TYPE_CHECKING:
        def __call__(self, incident_wavefront: Wavefront) -> Wavefront:
            ...
