import svetlanna as sv
from svetlanna.parameters import Parameter, ConstrainedParameter
from svetlanna.parameters import InnerParameterStorageModule
import torch
import pytest


def test_inner_parameter_storage():
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

    # tensors should be registered as buffers
    assert torch_tensor in list(storage.buffers())

    # svetlanna parameters should be registered as buffers
    assert sv_parameter is storage.value3
    assert sv_parameter.inner_storage is getattr(storage, "value3_svtlnn_inner_storage")
    assert sv_bounded_parameter is storage.value4
    assert sv_bounded_parameter.inner_storage is getattr(
        storage, "value4_svtlnn_inner_storage"
    )


@pytest.mark.parametrize(
    "parameter",
    [
        Parameter(data=123.0),
        ConstrainedParameter(data=123.0, min_value=0, max_value=300),
    ],
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
    "parameter",
    [
        Parameter(data=123.0),
        ConstrainedParameter(data=123.0, min_value=0, max_value=300),
    ],
)
def test_behavior_as_a_tensor(parameter):
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
    assert repr(parameter)


def test_storage_to_device(device_simple: str):
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

    storage.to(device=device_simple)
    # test if all values has been transferred to the device
    assert storage.value1.device.type == device_simple
    assert storage.value2.device.type == device_simple
    assert storage.value3.device.type == device_simple
    assert storage.value4.device.type == device_simple

    storage.to(device="cpu")
    # test if all values has been transferred to the cpu
    assert storage.value1.device.type == "cpu"
    assert storage.value2.device.type == "cpu"
    assert storage.value3.device.type == "cpu"
    assert storage.value4.device.type == "cpu"


@pytest.mark.parametrize(
    "parameter",
    [
        Parameter(data=torch.tensor(123.0, dtype=torch.float32)),
        ConstrainedParameter(
            data=torch.tensor(123.0, dtype=torch.float32), min_value=0, max_value=300
        ),
    ],
)
def test_parameter_to_device(
    device_simple: str, parameter: Parameter | ConstrainedParameter
):

    class E(sv.elements.Element):
        def __init__(self, parameter: Parameter | ConstrainedParameter) -> None:
            super().__init__(
                simulation_parameters=sv.SimulationParameters(
                    x=torch.linspace(0, 1, 10),
                    y=torch.linspace(0, 1, 10),
                    wavelength=1.0,
                )
            )
            self.parameter = parameter

        def forward(self, incident_wavefront: sv.Wavefront) -> sv.Wavefront:
            return incident_wavefront

    el = E(parameter)
    el.to(device=device_simple)

    assert el.parameter.device.type == device_simple
    assert el.parameter.data.device.type == device_simple
    assert el.parameter.inner_parameter.device.type == device_simple


def test_constrained_parameter_attrs():
    min_value = torch.tensor(0.0)
    max_value = torch.tensor(10.0)

    def bound_func(x: torch.Tensor) -> torch.Tensor:
        return x

    def inv_bound_func(x: torch.Tensor) -> torch.Tensor:
        return x

    parameter = ConstrainedParameter(
        data=5.0,
        min_value=min_value,
        max_value=max_value,
        bound_func=bound_func,
        inv_bound_func=inv_bound_func,
    )

    assert parameter.min_value is min_value
    assert parameter.max_value is max_value
    assert parameter.bound_func is bound_func
    assert parameter.inv_bound_func is inv_bound_func
