from svetlanna.parameters import Parameter, ConstrainedParameter
from svetlanna.parameters import InnerParameterStorageModule
import torch
import pytest


def test_inner_parameter_storage():
    """
    Tests the inner parameter storage module."""
    torch_parameter = torch.nn.Parameter(torch.tensor(1.0))
    torch_tensor = torch.tensor(2.0)
    sv_parameter = Parameter(torch.tensor(3.0))
    sv_bounded_parameter = ConstrainedParameter(
        torch.tensor(4.0), min_value=0.0, max_value=2.0
    )

    storage = InnerParameterStorageModule(
        {
            "value1": torch_parameter,
            "value2": torch_tensor,
            "value3": sv_parameter,
            "value4": sv_bounded_parameter,
        }
    )

    # test if values are registered
    assert storage.value1 is torch_parameter
    assert storage.value2 is torch_tensor
    assert storage.value3 is sv_parameter
    assert storage.value4 is sv_bounded_parameter

    # torch parameter should be registered as parameter
    assert torch_parameter in list(storage.parameters())

    # tensors and svetlanna paramentes should be registered as buffers
    assert torch_tensor in list(storage.buffers())
    assert sv_parameter in list(storage.buffers())
    assert sv_bounded_parameter in list(storage.buffers())

    # test if non-tensors can not be used to create a storage
    with pytest.raises(TypeError):
        InnerParameterStorageModule(
            {
                "a": 123,  # type: ignore
            }
        )


@pytest.mark.parametrize(
    "parameter",
    [
        Parameter(data=123.0),
        ConstrainedParameter(data=123.0, min_value=0, max_value=300),
    ],
)
def test_new(parameter: Parameter | ConstrainedParameter):
    """
    Tests the properties of a new parameter object.

        This function verifies that the provided parameter is a PyTorch tensor,
        not a `torch.nn.Parameter`, and behaves correctly with basic tensor operations.
        It also checks the types of inner attributes.

        Args:
            parameter: The Parameter or ConstrainedParameter instance to test.

        Returns:
            None
    """
    # check if parameter is a tensor and not a torch parameter
    assert isinstance(parameter, torch.Tensor)
    assert not isinstance(parameter, torch.nn.Parameter)

    # check if parameter works as a tensor
    assert isinstance(parameter * 2, torch.Tensor)
    assert not isinstance(parameter * 2, Parameter)

    assert isinstance(parameter.inner_parameter, torch.nn.Parameter)
    assert isinstance(parameter.inner_storage, InnerParameterStorageModule)


@pytest.mark.parametrize(
    "parameter",
    [
        Parameter(data=123.0),
        ConstrainedParameter(data=123.0, min_value=0, max_value=300),
    ],
)
def test_behavior_as_a_tensor(parameter):
    """
    Tests the behavior of the parameter when used as a tensor.

        This tests multiplication and exponentiation operations with a scalar,
        both directly and using torch functions to ensure proper handling via
        __torch_function__.

        Args:
            parameter: The parameter object to test.

        Returns:
            None
    """
    a = 123.0
    b = 10
    res_mul = torch.tensor(a * b)  # a * b
    res_pow = torch.tensor(a**b)  # a + b

    # test __torch_function__ for args processing
    torch.testing.assert_close(parameter * b, res_mul)
    torch.testing.assert_close(parameter**b, res_pow)
    # test __torch_function__ for kwargs processing
    torch.testing.assert_close(torch.mul(input=parameter, other=b), res_mul)
    torch.testing.assert_close(torch.pow(parameter, b), res_pow)


def test_bounded_parameter_inner_value():
    """
    Tests the inner parameter value of ConstrainedParameter with and without custom bound functions.

        This test verifies that the inner parameter correctly maps to the constrained data
        value using both the default sigmoid function and a user-defined bound function.
        It also checks the `value` property when a custom bound function is provided.

        Args:
            None

        Returns:
            None
    """
    data = 2.0
    min_value = 0.0
    max_value = 5.0

    # === default bound_func ===
    parameter = ConstrainedParameter(
        data=data, min_value=min_value, max_value=max_value
    )

    # test inner parameter value
    torch.testing.assert_close(
        (max_value - min_value) * torch.sigmoid(parameter.inner_parameter) + min_value,
        torch.tensor(data),
    )

    # === custom bound_func ===
    def bound_func(x: torch.Tensor) -> torch.Tensor:
        if x < 0:
            return torch.tensor(0.0)
        if x > 1:
            return torch.tensor(1.0)
        return x

    def inv_bound_func(x: torch.Tensor) -> torch.Tensor:
        return x

    parameter = ConstrainedParameter(
        data=data,
        min_value=min_value,
        max_value=max_value,
        bound_func=bound_func,
        inv_bound_func=inv_bound_func,
    )

    # test `value` property
    torch.testing.assert_close(parameter.value, torch.tensor(data))

    # test inner parameter value
    torch.testing.assert_close(
        (max_value - min_value) * bound_func(parameter.inner_parameter) + min_value,
        torch.tensor(data),
    )


@pytest.mark.parametrize(
    "parameter",
    [
        Parameter(data=123.0),
        ConstrainedParameter(data=123.0, min_value=0, max_value=300),
    ],
)
def test_repr(parameter):
    """
    Tests the repr of a parameter.

        Args:
            parameter: The parameter to test.

        Returns:
            None: This function only asserts that `repr(parameter)` does not raise an exception.
    """
    assert repr(parameter)


@pytest.mark.parametrize(
    ("device",),
    [
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda is not available"
            ),
        ),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="mps is not available"
            ),
        ),
    ],
)
def test_storage_to_device(device):
    """
    Tests moving an InnerParameterStorageModule to a specified device and back to CPU.

        Args:
            device: The device to move the storage to (e.g., 'cuda', 'mps').

        Returns:
            None
    """
    torch_parameter = torch.nn.Parameter(torch.tensor(1.0))
    torch_tensor = torch.tensor(2.0)
    sv_parameter = Parameter(torch.tensor(3.0))
    sv_bounded_parameter = ConstrainedParameter(
        torch.tensor(4.0), min_value=0.0, max_value=2.0
    )

    storage = InnerParameterStorageModule(
        {
            "value1": torch_parameter,
            "value2": torch_tensor,
            "value3": sv_parameter,
            "value4": sv_bounded_parameter,
        }
    )

    storage.to(device=device)
    # test if all values has been transferred to the device
    assert storage.value1.device.type == device
    assert storage.value2.device.type == device
    assert storage.value3.device.type == device
    assert storage.value4.device.type == device

    storage.to(device="cpu")
    # test if all values has been transferred to the cpu
    assert storage.value1.device.type == "cpu"
    assert storage.value2.device.type == "cpu"
    assert storage.value3.device.type == "cpu"
    assert storage.value4.device.type == "cpu"


@pytest.mark.parametrize(
    ("device",),
    [
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda is not available"
            ),
        ),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="mps is not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "parameter",
    [
        Parameter(data=torch.tensor(123.0, dtype=torch.float32)),
        ConstrainedParameter(
            data=torch.tensor(123.0, dtype=torch.float32), min_value=0, max_value=300
        ),
    ],
)
def test_parameter_to_device(device, parameter):
    """
    Tests that a Parameter or ConstrainedParameter can be moved to the specified device.

        Args:
            device: The device to move the parameter to (e.g., 'cuda', 'mps').
            parameter: The Parameter or ConstrainedParameter instance to test.

        Returns:
            None
    """
    # transferred_parameter = parameter.to(device)
    # assert transferred_parameter.device.type == device
    # assert transferred_parameter.inner_storage.device.type == device

    parameter.inner_storage.to(device=device)
    assert parameter.inner_parameter.device.type == device
