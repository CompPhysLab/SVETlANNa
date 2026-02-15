import torch
from typing import TYPE_CHECKING, TypeVar, Generic
from typing import ParamSpec, Concatenate, Callable

from .parameters import Parameter


_Input = TypeVar("_Input")
_Output = TypeVar("_Output")
_Params = ParamSpec("_Params")


class PartialWithParameters(torch.nn.Module, Generic[_Input, _Output]):
    def __init__(
        self,
        function: Callable[Concatenate[_Input, _Params], _Output],
        *args: _Params.args,
        **kwargs: _Params.kwargs,
    ) -> None:
        """Wrap an arbitrary function with trainable keyword arguments.

        This behaves like `functools.partial`, but only keyword arguments are
        supported. Use this wrapper when you want keyword arguments to be
        registered as trainable parameters or as buffers (for tensor-valued
        constants). This is especially useful for multi-device workflows, since
        parameters and buffers move with the module.

        Parameters
        ----------
        function : Callable[Concatenate[_Input, _Params], _Output]
            Arbitrary function with parameters.
        *args : _Params.args
            Positional arguments (not supported; must be empty).
        **kwargs : _Params.kwargs
            Keyword arguments for the function. Values are registered as
            parameters, buffers, or plain attributes depending on their type.

        Examples
        --------
        Suppose you have a function that describes a nonlinear response and has
        trainable parameters. See the example in
        [NonlinearElement][svetlanna.elements.NonlinearElement].
        """
        super().__init__()
        self.function = function

        if len(args) > 0:
            raise ValueError(
                "PartialWithParameters does not support positional arguments. "
                "Please use only keyword arguments."
            )

        self.__function_args = args  # for typing purposes
        self.__function_kwargs_keys = list(kwargs.keys())  # for typing purposes

        for name, value in dict(kwargs).items():
            if isinstance(value, Parameter):
                # SVETlANNa's Parameter is handled by pointing auxiliary attribute on
                # their inner_storage with a name plus _svtlnn_inner_storage suffix:
                setattr(self, name + "_svtlnn_inner_storage", value.inner_storage)
                setattr(self, name, value)

            elif isinstance(value, torch.nn.Parameter):
                self.register_parameter(name, value)
            elif isinstance(value, torch.Tensor):
                self.register_buffer(name, value)
            else:
                setattr(self, name, value)

    def forward(self, function_argument: _Input) -> _Output:
        # The function is called with the current values of the parameters.
        kwargs = {name: getattr(self, name) for name in self.__function_kwargs_keys}
        return self.function(function_argument, *self.__function_args, **kwargs)

    if TYPE_CHECKING:

        def __call__(self, function_argument: _Input) -> _Output: ...
