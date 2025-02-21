from svetlanna.parameters import Parameter, ConstrainedParameter
from svetlanna.parameters import InnerParameterStorageModule
import torch
import pytest


def test_inner_parameter_storage():
    torch_parameter = torch.nn.Parameter(torch.tensor(1.))
    torch_tensor = torch.tensor(2.)
    sv_parameter = Parameter(torch.tensor(3.))
    sv_bounded_parameter = ConstrainedParameter(
        torch.tensor(4.), min_value=0., max_value=2.
    )

    storage = InnerParameterStorageModule(
        {
            'value1': torch_parameter,
            'value2': torch_tensor,
            'value3': sv_parameter,
            'value4': sv_bounded_parameter,
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
                'a': 123,  # type: ignore
            }
        )


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
    assert isinstance(parameter.inner_storage, InnerParameterStorageModule)


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
        (max_value-min_value) * torch.sigmoid(parameter.inner_parameter)
        + min_value,
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
        (max_value-min_value) * bound_func(parameter.inner_parameter)
        + min_value,
        torch.tensor(data)
    )


@pytest.mark.parametrize(
    "parameter", [
        Parameter(data=123.),
        ConstrainedParameter(data=123., min_value=0, max_value=300)
    ]
)
def test_repr(parameter):
    assert repr(parameter)


@pytest.mark.parametrize(
    ("device",),
    [
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="cuda is not available"
            )
        ),
        pytest.param(
            'mps',
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(),
                reason="mps is not available"
            )
        )
    ]
)
def test_storage_to_device(device):
    torch_parameter = torch.nn.Parameter(torch.tensor(1.))
    torch_tensor = torch.tensor(2.)
    sv_parameter = Parameter(torch.tensor(3.))
    sv_bounded_parameter = ConstrainedParameter(
        torch.tensor(4.), min_value=0., max_value=2.
    )

    storage = InnerParameterStorageModule(
        {
            'value1': torch_parameter,
            'value2': torch_tensor,
            'value3': sv_parameter,
            'value4': sv_bounded_parameter,
        }
    )

    storage.to(device=device)
    # test if all values has been transferred to the device
    assert storage.value1.device.type == device
    assert storage.value2.device.type == device
    assert storage.value3.device.type == device
    assert storage.value4.device.type == device

    storage.to(device='cpu')
    # test if all values has been transferred to the cpu
    assert storage.value1.device.type == 'cpu'
    assert storage.value2.device.type == 'cpu'
    assert storage.value3.device.type == 'cpu'
    assert storage.value4.device.type == 'cpu'


@pytest.mark.parametrize(
    ("device",),
    [
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="cuda is not available"
            )
        ),
        pytest.param(
            'mps',
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(),
                reason="mps is not available"
            )
        )
    ]
)
@pytest.mark.parametrize(
    "parameter", [
        Parameter(data=torch.tensor(123., dtype=torch.float32)),
        ConstrainedParameter(
            data=torch.tensor(123., dtype=torch.float32),
            min_value=0,
            max_value=300
        )
    ]
)
def test_parameter_to_device(device, parameter):
    # transferred_parameter = parameter.to(device)
    # assert transferred_parameter.device.type == device
    # assert transferred_parameter.inner_storage.device.type == device

    parameter.inner_storage.to(device=device)
    assert parameter.inner_parameter.device.type == device
