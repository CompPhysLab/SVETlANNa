import svetlanna as sv
import torch
import pytest


def test_partial_with_parameters_init():
    """Test initialization of PartialWithParameters."""

    def f(x, a):
        return x * a

    # Only keyword arguments are allowed (positional args should raise ValueError)
    a = torch.tensor(10.0)
    with pytest.raises(ValueError):
        sv.PartialWithParameters(f, a)

    # Test that the function works correctly with provided kwargs
    f_partial = sv.PartialWithParameters(f, a=a)

    x = torch.tensor(5.0)
    assert f_partial(x) == x * a

    # Test that parameters can be updated after initialization
    f_partial.a = torch.tensor(20.0)
    assert f_partial(x) == x * 20.0


def test_partial_with_parameters_registration():
    """Test that parameters are properly registered as parameters and buffers."""

    def f(x, a, b, c, d):
        return x + a + b + c + d

    partial = sv.PartialWithParameters(
        f,
        a=torch.nn.Parameter(torch.tensor(1.0)),
        b=torch.tensor(2.0),
        c=sv.Parameter(torch.tensor(3.0)),
        d=sv.ConstrainedParameter(torch.tensor(4.0), min_value=0.0, max_value=10.0),
    )

    # Check that torch.nn.Parameter is registered as a parameter
    assert hasattr(partial, "a")
    assert "a" in [name for name, _ in partial.named_parameters()]

    # Check that torch.Tensor is registered as a buffer
    assert hasattr(partial, "b")
    assert "b" in [name for name, _ in partial.named_buffers()]

    # Check that sv.Parameter and sv.ConstrainedParameter are accessible
    # and their inner storages are registered
    assert hasattr(partial, "c")
    assert hasattr(partial, "d")
    assert hasattr(partial, "c_svtlnn_inner_storage")
    assert hasattr(partial, "d_svtlnn_inner_storage")


def test_partial_with_parameters_device(device_simple: str):
    """Test device migration for PartialWithParameters."""

    def f(x, regular_number, tensor, torch_parameter, parameter, constrained_parameter):
        return (
            x
            + regular_number
            + tensor
            + torch_parameter
            + parameter
            + constrained_parameter
        )

    class A(torch.nn.Module):
        def __init__(self, function):
            super().__init__()
            self.function = function

        def forward(self, x):
            return self.function(x)

    a = A(
        sv.PartialWithParameters(
            f,
            regular_number=1.0,
            tensor=torch.tensor(0.0),
            torch_parameter=torch.nn.Parameter(torch.tensor(2.0)),
            parameter=sv.Parameter(torch.tensor(3.0)),
            constrained_parameter=sv.ConstrainedParameter(
                torch.tensor(4.0),
                min_value=0.0,
                max_value=10.0,
            ),
        )
    )

    x = torch.tensor(5.0)
    assert a(x) == f(x, 0, 1, 2, 3, 4)

    # All parameters should be moved to the target device without errors
    a.to(device_simple)
    assert a(x.to(device_simple)) == f(x, 0, 1, 2, 3, 4)
