import torch
from typing import TYPE_CHECKING, TypeVar, Generic
from typing import ParamSpec, Concatenate, Callable

from .parameters import Parameter


_I = TypeVar("_I")
_O = TypeVar("_O")
_P = ParamSpec("_P")


class PartialWithParameters(torch.nn.Module, Generic[_I, _O]):
    def __init__(
        self,
        function: Callable[Concatenate[_I, _P], _O],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> None:
        """This class wraps an arbitrary function with trainable keyword arguments.
        It works similarly to `functools.partial`, but only keyword arguments are supported.

        Example:
        > sv.elements.NonlinearElement(
        >     simulation_parameters,
        >     sv.PartialWithParameters(
        >         lambda x, alpha: torch.polar(x.abs()**alpha, x.angle()),
        >         alpha=torch.nn.Parameter(torch.tensor(1.0)),
        >     )
        > )

        Parameters
        ----------
        function : Callable[Concatenate[_I, _P], _O]
            Arbitrary function with parameters
        *args : _P.args
            Positional arguments for the function (not supported, should be empty)
        **kwargs : _P.kwargs
            Keyword arguments for the function, these will be trainable parameters of the module
        """
        super().__init__()
        self.function = function

        if len(args) > 0:
            raise ValueError(
                "PartialWithParameters does not support positional arguments. "
                "Please use only keyword arguments."
            )

        self.__function_args = args  # for typing purposes
        self.__function_kwargs = kwargs  # for typing purposes

        for name, value in dict(kwargs).items():
            # SVETlANNa's Parameter is handled by pointing auxiliary attribute on
            # their inner_storage with a name plus _svtlnn_inner_storage suffix:
            if isinstance(value, Parameter):
                setattr(self, name + "_svtlnn_inner_storage", value.inner_storage)

            setattr(self, name, value)

    def forward(self, function_argument: _I) -> _O:
        return self.function(
            function_argument, *self.__function_args, **self.__function_kwargs
        )

    if TYPE_CHECKING:

        def __call__(self, function_argument: _I) -> _O: ...
