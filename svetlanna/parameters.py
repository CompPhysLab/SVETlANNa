import torch
from typing import Callable, Any


class Parameter(torch.Tensor):
    """`torch.Parameter` replacement.
    Added for further feature enrichment.
    """
    @staticmethod
    def __new__(cls, *args, **kwargs):
        # see https://github.com/albanD/subclass_zoo/blob/ec47458346c2a1cfcd5e676926a4bbc6709ff62e/base_tensor.py   # noqa: E501
        return super(cls, Parameter).__new__(cls)

    def __init__(
        self,
        data: Any,
        requires_grad: bool = True
    ):
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
        self.inner_parameter = torch.nn.Parameter(
            data=data,
            requires_grad=requires_grad
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # see https://pytorch.org/docs/stable/notes/extending.html#extending-torch-python-api   # noqa: E501

        # real parameter should be used for any calculations,
        # therefore the `instance` should be replaced to
        # `instance.inner_parameter` in `args` and `kwargs`
        if kwargs is None:
            kwargs = {}
        kwargs = {
            k: v.inner_parameter if isinstance(v, cls) else v for k, v in kwargs.items()    # noqa: E501
        }
        args = (a.inner_parameter if isinstance(a, cls) else a for a in args)
        return func(*args, **kwargs)

    def __repr__(self, *args, **kwargs) -> str:
        return repr(self.inner_parameter)


def sigmoid_inv(x: torch.Tensor) -> torch.Tensor:
    """Inverse sigmoid function

    Parameters
    ----------
    x : torch.Tensor
        the input tensor

    Returns
    -------
    torch.Tensor
        the output tensor
    """
    return torch.log(x/(1-x))


# TODO: rename to ConstrainedParameter
class BoundedParameter(Parameter):
    """Constrained parameter
    """
    @staticmethod
    def __new__(cls, *args, **kwargs):
        return super(torch.Tensor, BoundedParameter).__new__(cls)

    def __init__(
        self,
        data: Any,
        min_value: Any,
        max_value: Any,
        bound_func: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
        inv_bound_func: Callable[[torch.Tensor], torch.Tensor] = sigmoid_inv,
        requires_grad: bool = True
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
            function that map $\mathbb{R}\to[0,1]$, by default torch.sigmoid
        inv_bound_func : Callable[[torch.Tensor], torch.Tensor], optional
            inverse function of `bound_func`
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

        super().__init__(
            data=initial_value,
            requires_grad=requires_grad
        )

        self.min_value = min_value
        self.max_value = max_value

        self.__a = a
        self.__b = b

        self.bound_func = bound_func

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
        return self.__a * self.bound_func(self.inner_parameter) + self.__b

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # the same as for Parameter class, `instance.value` should be used
        if kwargs is None:
            kwargs = {}
        kwargs = {
            k: v.value if isinstance(v, cls) else v for k, v in kwargs.items()
        }
        args = (a.value if isinstance(a, cls) else a for a in args)
        return func(*args, **kwargs)

    def __repr__(self) -> str:
        return f'Bounded parameter containing:\n{repr(self.value)}'
