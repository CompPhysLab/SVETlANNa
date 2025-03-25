import svetlanna
import svetlanna.elements
import torch
from svetlanna.elements.element import INNER_PARAMETER_SUFFIX
import pytest

import svetlanna.specs


class ElementToTest(svetlanna.elements.Element):
    def __init__(
        self,
        simulation_parameters: svetlanna.SimulationParameters,
        test_parameter,
        test_buffer,
    ) -> None:
        super().__init__(simulation_parameters)
        self.test_parameter = self.process_parameter(
            'test_parameter', test_parameter
        )
        self.test_buffer = self.make_buffer(
            'test_buffer', test_buffer
        )

    def forward(
        self,
        incident_wavefront: svetlanna.Wavefront
    ) -> svetlanna.Wavefront:
        return super().forward(incident_wavefront)


def test_setattr():
    sim_params = svetlanna.SimulationParameters(
        {
            'W': torch.linspace(-10, 10, 100),
            'H': torch.linspace(-10, 10, 100),
            'wavelength': 1.,
        }
    )
    test_parameter = svetlanna.Parameter(10.)
    element = ElementToTest(
        sim_params,
        test_parameter=test_parameter,
        test_buffer=None
    )

    # check if inner storage of the parameter has been saved
    parameter_name = 'test_parameter' + INNER_PARAMETER_SUFFIX
    assert getattr(element, parameter_name) is test_parameter.inner_storage
    assert element.test_parameter.inner_parameter in element.parameters()


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
def test_make_buffer(device):
    sim_params = svetlanna.SimulationParameters(
        {
            'W': torch.linspace(-10, 10, 100),
            'H': torch.linspace(-10, 10, 100),
            'wavelength': 1.,
        }
    )
    test_buffer = torch.tensor(123.)
    element = ElementToTest(
        sim_params,
        test_parameter=None,
        test_buffer=test_buffer
    )

    # check if buffer has been registered
    assert hasattr(element, 'test_buffer')
    assert getattr(element, 'test_buffer') in element.buffers()

    # check if buffer is automatically transferred to device
    element.to(device)
    assert getattr(element, 'test_buffer').device.type == device

    # test if a buffer cannot be registered with a tensor on a device
    # distinct from the simulation parameters' device
    with pytest.raises(ValueError):
        element = ElementToTest(
            sim_params,
            test_parameter=None,
            test_buffer=test_buffer.to(device)
        )


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
def test_process_parameter(device):
    sim_params = svetlanna.SimulationParameters(
        {
            'W': torch.linspace(-10, 10, 100),
            'H': torch.linspace(-10, 10, 100),
            'wavelength': 1.,
        }
    )
    test_parameter = torch.nn.Parameter(torch.tensor(123.))
    element = ElementToTest(
        sim_params,
        test_parameter=test_parameter,
        test_buffer=None
    )

    # check if parameter has been registered
    assert hasattr(element, 'test_parameter')
    assert getattr(element, 'test_parameter') in element.parameters()

    # check if parameter is automatically transferred to device
    element.to(device)
    assert getattr(element, 'test_parameter').device.type == device

    # test tensor as a parameter
    test_parameter = torch.tensor(123.)
    element = ElementToTest(
        sim_params,
        test_parameter=test_parameter,
        test_buffer=None
    )

    # check if test_parameter has been registered as a buffer
    assert hasattr(element, 'test_parameter')
    assert getattr(element, 'test_parameter') not in element.parameters()
    assert getattr(element, 'test_parameter') in element.buffers()

    # test if a parameter cannot be registered with a tensor on a device
    # distinct from the simulation parameters' device
    with pytest.raises(ValueError):
        element = ElementToTest(
            sim_params,
            test_parameter=test_parameter.to(device),
            test_buffer=None
        )


def test_to_specs():
    sim_params = svetlanna.SimulationParameters(
        {
            'W': torch.linspace(-10, 10, 100),
            'H': torch.linspace(-10, 10, 100),
            'wavelength': 1.,
        }
    )
    test_parameter = torch.nn.Parameter(torch.tensor(123.))
    element = ElementToTest(
        sim_params,
        test_parameter=test_parameter,
        test_buffer=None
    )

    specs = list(element.to_specs())
    assert len(specs) == 1
    assert specs[0].parameter_name == 'test_parameter'

    representations = list(specs[0].representations)
    assert len(representations) == 1
    assert isinstance(representations[0], svetlanna.specs.ReprRepr)
