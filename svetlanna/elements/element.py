from abc import ABCMeta, abstractmethod
from torch import nn
from torch import Tensor
from ..simulation_parameters import SimulationParameters
from ..specs import PrettyReprRepr, ParameterSpecs, SubelementSpecs, Specsable
from ..specs.specs_writer import write_specs_to_html, context_generator
from io import StringIO
from typing import Iterable, TypeVar, TYPE_CHECKING, Self
from ..parameters import Parameter
from ..wavefront import Wavefront
from warnings import warn


INNER_PARAMETER_SUFFIX = "_svtlnn_inner_parameter"

_T = TypeVar("_T", Tensor, None)
_V = TypeVar("_V")


class _BufferedValueContainer(tuple):
    """Internal class that marks buffered values.

    It is used to prevent double __setattr__ calls with the same value in
    patterns like `self.x = self.make_buffer('x', x_value)`.
    Inheriting from tuple is used for performance reasons, hence `__slots__`.
    This approach was identified by GPT as the fastest one.
    """

    __slots__ = ()

    def __new__(cls, obj: Tensor | None):
        return super().__new__(cls, (obj,))


class Element(nn.Module, metaclass=ABCMeta):
    def __init__(self, simulation_parameters: SimulationParameters) -> None:
        """
        This is the abstract class for all optical elements in SVETlANNa.
        It is inherited from `torch.nn.Module`, so it is PyTorch-compatible.
        Each element takes an incident wavefront and produces a transmitted wavefront.

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Simulation parameters.
        """

        super().__init__()

        self.simulation_parameters = simulation_parameters.clone()

    @abstractmethod
    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        """Forward propagation through the optical element."""

    def to(self, *args, **kwargs) -> Self:
        """
        Move element to a different device/dtype.

        Overrides `torch.nn.Module.to()` to also transfer simulation_parameters.

        Returns
        -------
        Self
            The element itself, not a copy.
        """
        self.simulation_parameters.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def to_specs(self) -> Iterable[ParameterSpecs | SubelementSpecs]:
        for name, parameter in self.named_parameters():

            yield ParameterSpecs(
                parameter_name=name, representations=(PrettyReprRepr(value=parameter),)
            )

    def __setattr__(
        self, name: str, value: Tensor | nn.Module | _BufferedValueContainer
    ) -> None:

        if isinstance(value, _BufferedValueContainer):
            # In the case of pattern `self.x = self.make_buffer('x', x_value)`
            # the attribute value is already set, the __setattr__ is ignored
            try:
                # Check if a buffer with the given name is already registered
                self.get_buffer(name)
                return
            except AttributeError:
                warn(
                    f"You set the attribute {name} with an object of "
                    "internal type _BufferedValueContainer. "
                    "Make sure this is the intended behavior."
                )
                return super().__setattr__(name, value)  # type: ignore

        # BoundedParameter and Parameter are handled by pointing
        # auxiliary attribute on them with a name plus INNER_PARAMETER_SUFFIX
        if isinstance(value, Parameter):
            super().__setattr__(name + INNER_PARAMETER_SUFFIX, value.inner_storage)

        return super().__setattr__(name, value)

    def _repr_html_(self) -> str:
        stream = StringIO("")

        def write_element_details(element: Specsable):
            subelements: list[SubelementSpecs] = []
            writer_context_generator = context_generator(
                element=element, element_index=0, directory="", subelements=subelements
            )
            # Write element's parameter specs to the stream
            write_specs_to_html(element, 0, writer_context_generator, stream)

            # Iterate over element's subelements (SubelementSpecs)
            for subelement in subelements:
                # Create details tag for the element and start summary tag
                stream.write(
                    '<details style="border: 1px solid gray;margin: 0.3rem 0">'
                    '<summary style="font-family:monospace;'
                    'background-color:#cff1f0;color:black;padding:0.3rem">'
                )
                element_name = subelement.subelement.__class__.__name__
                # Write the element's name to the summary tag
                stream.write(f"[{subelement.subelement_type}] <b>{element_name}</b>")

                # Close summary tag and open a new div for the subelement
                stream.write(
                    "</summary>" '<div style="margin-left:2rem;margin-right: 0.3rem">'
                )
                # Repeat the process for the subelement
                write_element_details(subelement.subelement)
                # Close the div and the details tags
                stream.write("</div>" "</details>")

        write_element_details(self)
        return stream.getvalue()

    def make_buffer(self, name: str, value: _T, persistent: bool = False) -> _T:
        """Make buffer for internal use.

        Use case in `__init__` method:
        ```python linenums="0"
        self.mask = self.make_buffer('mask', some_tensor)
        ```
        This allow torch to properly process the `.to` method on the element, since the buffer `maask` will be transferred to the required device along with simulation parameters.
        This allows torch to properly process the `.to` method on the element, since the buffer `mask` will be transferred to the required device along with simulation parameters.

        Parameters
        ----------
        name : str
            Name of the new buffer
            (it is more convenient to use the name of the new attribute).
        value : _T
            Tensor to be buffered.
        persistent : bool, optional
            See torch docs on buffers, by default `False`.

        Returns
        -------
        _T
            The value passed to the method.
        """

        if value is not None:
            if value.device != self.simulation_parameters.device:
                raise ValueError(
                    f"Tensor to be buffered as {name} must be on "
                    "the simulation parameters device."
                )

        self.register_buffer(name, value, persistent=persistent)

        # The instance of _BufferedValueContainer is returned
        # to support `self.x = self.make_buffer('x', x_value)` pattern
        return _BufferedValueContainer(self.__getattr__(name))  # type: ignore

    def process_parameter(self, name: str, value: _V) -> _V:
        """Process element parameter passed by user.
        Automatically registers buffer for non-parametric tensors.

        Use case in `__init__` method:
        ```python linenums="0"
        class SomeElement(Element):
            def __init__(self, simulation_parameters, mask, a):
                super().__init__(simulation_parameters)

                self.mask = self.process_parameter('mask', mask)
                self.a = self.process_parameter('a', a)

                ...
        ```

        Parameters
        ----------
        name : str
            Name of the new buffer
            (it is more convenient to use the name of the new attribute).
        value : _V
            The value of the element parameter.

        Returns
        -------
        _V
            The value passed to the method.
        """
        if isinstance(value, Tensor):
            if value.device != self.simulation_parameters.device:
                raise ValueError(
                    f"Parameter {name} must be on " "the simulation parameters device."
                )
        if isinstance(value, (nn.Parameter, Parameter)):
            return value
        if isinstance(value, Tensor):
            return self.make_buffer(name, value, persistent=True)  # type: ignore
        return value

    # === methods below are added for typing only ===

    if TYPE_CHECKING:

        def __call__(self, incident_wavefront: Wavefront) -> Wavefront: ...
