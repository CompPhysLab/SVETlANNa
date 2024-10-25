from svetlanna.parameters import Parameter, ConstrainedParameter
import torch
import pytest


@pytest.mark.parametrize(
    "parameter", [
        Parameter(data=123.),
        ConstrainedParameter(data=123., min_value=0, max_value=300)
    ]
)
def test_new(parameter: Parameter | ConstrainedParameter):
    # check if parameter is a tensor and not a torch parameter
    assert isinstance(parameter, torch.Tensor)
    assert not isinstance(parameter, torch.nn.Parameter)

    # check if parameter works as a tensor
    assert isinstance(parameter * 2, torch.Tensor)
    assert not isinstance(parameter * 2, Parameter)

    assert isinstance(parameter.inner_parameter, torch.nn.Parameter)


@pytest.mark.parametrize(
    "parameter", [
        Parameter(data=123.),
        ConstrainedParameter(data=123., min_value=0, max_value=300)
    ]
)
def test_behavior_as_a_tensor(parameter):
    a = 123.
    b = 10
    res_mul = torch.tensor(a * b)  # a * b
    res_pow = torch.tensor(a ** b)  # a + b

    # test __torch_function__ for args processing
    torch.testing.assert_close(parameter * b, res_mul)
    torch.testing.assert_close(parameter**b, res_pow)
    # test __torch_function__ for kwargs processing
    torch.testing.assert_close(torch.mul(input=parameter, other=b), res_mul)
    torch.testing.assert_close(torch.pow(parameter, b), res_pow)


def test_bounded_parameter_inner_value():
    data = 2.
    min_value = 0.
    max_value = 5.

    # === default bound_func ===
    parameter = ConstrainedParameter(
        data=data,
        min_value=min_value,
        max_value=max_value
    )

    # test inner parameter value
    torch.testing.assert_close(
        (max_value-min_value) * torch.sigmoid(parameter.inner_parameter) + min_value,
        torch.tensor(data)
    )

    # === custom bound_func ===
    def bound_func(x: torch.Tensor) -> torch.Tensor:
        if x < 0:
            return torch.tensor(0.)
        if x > 1:
            return torch.tensor(1.)
        return x

    def inv_bound_func(x: torch.Tensor) -> torch.Tensor:
        return x

    parameter = ConstrainedParameter(
        data=data,
        min_value=min_value,
        max_value=max_value,
        bound_func=bound_func,
        inv_bound_func=inv_bound_func
    )

    # test `value` property
    torch.testing.assert_close(parameter.value, torch.tensor(data))

    # test inner parameter value
    torch.testing.assert_close(
        (max_value-min_value) * bound_func(parameter.inner_parameter) + min_value,
        torch.tensor(data)
    )
