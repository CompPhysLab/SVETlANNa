import torch
from typing import Callable, Any, TypeAlias, TYPE_CHECKING, Mapping

# TODO: fix impropriate .to() method handling in parameters
# TODO: .data in Parameter and .inner_parameter.data are not the same (at least for constrained parameter),
# this can cause memory duplication, should be fixed somehow


class InnerParameterStorageModule(torch.nn.Module):
    def __init__(self, params_to_store: Mapping[str, Any]):
        super().__init__()
        self.expand(params_to_store)

    def expand(self, params_to_store: Mapping[str, Any]):
        """Add more parameters to the storage

        Parameters
        ----------
        params_to_store : Mapping[str, Any]
            parameters to store
        """
        for name, value in params_to_store.items():
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

    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...


class Parameter(torch.Tensor):
    """`torch.Parameter` replacement.
    Added for further feature enrichment.
    """

    @staticmethod
    def __new__(cls, *args, **kwargs):
        # see https://github.com/albanD/subclass_zoo/blob/ec47458346c2a1cfcd5e676926a4bbc6709ff62e/base_tensor.py   # noqa: E501
        return super(cls, Parameter).__new__(cls)

    def __init__(self, data: Any, requires_grad: bool = True):
        """
        Parameters
        ----------
        data : Any
            parameter tensor
        requires_grad : bool, optional
            if the parameter requires gradient, by default True
        """
        super().__init__()

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)

        # real parameter that should be optimized
        self.inner_storage = InnerParameterStorageModule(
            {
                "inner_parameter": torch.nn.Parameter(
                    data=data, requires_grad=requires_grad
                )
            }
        )

    @property
    def inner_parameter(self) -> torch.nn.Parameter:
        return self.inner_storage.inner_parameter

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # see https://pytorch.org/docs/stable/notes/extending.html#extending-torch-python-api   # noqa: E501

        # real parameter should be used for any calculations,
        # therefore the `instance` should be replaced to
        # `instance.inner_parameter` in `args` and `kwargs`
        if kwargs is None:
            kwargs = {}
        kwargs = {
            k: v.inner_parameter if isinstance(v, cls) else v for k, v in kwargs.items()
        }
        args = (a.inner_parameter if isinstance(a, cls) else a for a in args)
        return func(*args, **kwargs)

    def __repr__(self, **kwargs):
        return repr(self.inner_parameter)


class ConstrainedParameter(Parameter):
    """Constrained parameter"""

    @staticmethod
    def __new__(cls, *args, **kwargs):
        return super(torch.Tensor, ConstrainedParameter).__new__(cls)

    def __init__(
        self,
        data: Any,
        min_value: Any,
        max_value: Any,
        bound_func: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
        inv_bound_func: Callable[[torch.Tensor], torch.Tensor] = torch.logit,
        requires_grad: bool = True,
    ):
        r"""
        Parameters
        ----------
        data : Any
            parameter tensor
        min_value : Any
            minimum value tensor
        max_value : Any
            maximum value tensor
        bound_func : Callable[[torch.Tensor], torch.Tensor], optional
            function that map $\\mathbb{R}\to[0,1]$, by default `torch.sigmoid`
        inv_bound_func : Callable[[torch.Tensor], torch.Tensor], optional
            inverse function of `bound_func`, by default `torch.logit`
        requires_grad : bool, optional
            if the parameter requires gradient, by default True
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)

        if not isinstance(min_value, torch.Tensor):
            min_value = torch.tensor(min_value)

        if not isinstance(max_value, torch.Tensor):
            max_value = torch.tensor(max_value)

        # To find initial inner parameter value y0 one should calculate
        # y0 = inv_bound_func( (x0 - m) / (M - m) )
        # where x0 is data value
        a = max_value - min_value  # M - m
        b = min_value  # m
        initial_value = inv_bound_func((data - b) / a)

        super().__init__(data=initial_value, requires_grad=requires_grad)

        self.inner_storage.expand(
            {
                "a": a,
                "b": b,
                "min_value": min_value,
                "max_value": max_value,
                "bound_func": bound_func,
                "inv_bound_func": inv_bound_func,
            }
        )

    @property
    def min_value(self) -> torch.Tensor:
        return self.inner_storage.min_value

    @property
    def max_value(self) -> torch.Tensor:
        return self.inner_storage.max_value

    @property
    def bound_func(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.inner_storage.bound_func

    @property
    def inv_bound_func(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.inner_storage.inv_bound_func

    @property
    def value(self) -> torch.Tensor:
        """Parameter value

        Returns
        -------
        torch.Tensor
            Constrained parameter value computed with bound_func
        """
        # for inner parameter value y:
        # x = (M-m) * bound_function( y ) + m = a * bound_function( y ) + b
        return (
            self.inner_storage.a * self.bound_func(self.inner_parameter)
            + self.inner_storage.b
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # the same as for Parameter class, `instance.value` should be used
        if kwargs is None:
            kwargs = {}
        kwargs = {k: v.value if isinstance(v, cls) else v for k, v in kwargs.items()}
        args = (a.value if isinstance(a, cls) else a for a in args)
        return func(*args, **kwargs)

    def __repr__(self, **kwargs):
        return f"Constrained parameter containing:\n{repr(self.value)}"


OptimizableFloat: TypeAlias = float | torch.Tensor | torch.nn.Parameter | Parameter
OptimizableTensor: TypeAlias = torch.Tensor | torch.nn.Parameter | Parameter

BoundedParameter = ConstrainedParameter
