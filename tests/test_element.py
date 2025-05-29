import svetlanna
import svetlanna.elements
import torch
from svetlanna.elements.element import INNER_PARAMETER_SUFFIX, _BufferedValueContainer
import pytest

import svetlanna.specs


class ElementToTest(svetlanna.elements.Element):
    """
    Represents an element for testing within a simulation.

        This class serves as a basic building block for testing purposes,
        primarily passing data through without modification.
    """

    def __init__(
        self,
        simulation_parameters: svetlanna.SimulationParameters,
        test_parameter,
        test_buffer,
    ) -> None:
        """
        Initializes the class with simulation parameters and test data.

            Args:
                simulation_parameters: The simulation parameters object.
                test_parameter: The parameter to be processed.
                test_buffer: The buffer to be created.

            Returns:
                None
        """
        super().__init__(simulation_parameters)
        self.test_parameter = self.process_parameter("test_parameter", test_parameter)
        self.test_buffer = self.make_buffer("test_buffer", test_buffer)

    def forward(self, incident_wavefront: svetlanna.Wavefront) -> svetlanna.Wavefront:
        """
        Passes the incident wavefront through the layer.

          This method simply calls the `forward` method of the parent class,
          effectively passing the input wavefront unchanged.

          Args:
            incident_wavefront: The incoming wavefront.

          Returns:
            svetlanna.Wavefront: The transmitted wavefront (identical to the input).
        """
        return super().forward(incident_wavefront)


def test_setattr():
    """
    Tests that setattr correctly saves inner storage of parameters.

        This test creates a simulation and an element with a parameter, then
        asserts that the inner storage of the parameter is saved as expected
        and accessible through a specific attribute name. It also verifies that
        the inner parameter is present in the element's parameters dictionary.

        Args:
            None

        Returns:
            None
    """
    sim_params = svetlanna.SimulationParameters(
        {
            "W": torch.linspace(-10, 10, 100),
            "H": torch.linspace(-10, 10, 100),
            "wavelength": 1.0,
        }
    )
    test_parameter = svetlanna.Parameter(10.0)
    element = ElementToTest(sim_params, test_parameter=test_parameter, test_buffer=None)

    # check if inner storage of the parameter has been saved
    parameter_name = "test_parameter" + INNER_PARAMETER_SUFFIX
    assert getattr(element, parameter_name) is test_parameter.inner_storage
    assert element.test_parameter.inner_parameter in element.parameters()


@pytest.mark.parametrize(
    ("device",),
    [
        pytest.param("cpu"),
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
def test_make_buffer(device):
    """
    Tests the registration and device placement of a buffer.

        Args:
            device: The device to move the element to ('cpu', 'cuda', or 'mps').

        Returns:
            None
    """
    sim_params = svetlanna.SimulationParameters(
        {
            "W": torch.linspace(-10, 10, 100),
            "H": torch.linspace(-10, 10, 100),
            "wavelength": 1.0,
        }
    )
    test_buffer = torch.tensor(123.0)
    element = ElementToTest(sim_params, test_parameter=None, test_buffer=test_buffer)

    # check if buffer has been registered
    assert hasattr(element, "test_buffer")
    assert getattr(element, "test_buffer") in element.buffers()

    # check if buffer is automatically transferred to device
    element.to(device)
    assert getattr(element, "test_buffer").device.type == device

    # test if a buffer cannot be registered with a tensor on a device
    # distinct from the simulation parameters' device
    if device != "cpu":
        with pytest.raises(ValueError):
            element = ElementToTest(
                sim_params, test_parameter=None, test_buffer=test_buffer.to(device)
            )


@pytest.mark.parametrize(
    ("device",),
    [
        pytest.param("cpu"),
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
def test_process_parameter(device):
    """
    Tests the processing of parameters within the ElementToTest class.

        This test verifies that parameters are correctly registered, transferred to the specified device,
        and handled appropriately when provided as tensors or directly as nn.Parameters. It also checks
        for ValueErrors when attempting to register a parameter tensor on a different device than the simulation.

        Args:
            device: The device (e.g., 'cpu', 'cuda', 'mps') to test with.

        Returns:
            None
    """
    sim_params = svetlanna.SimulationParameters(
        {
            "W": torch.linspace(-10, 10, 100),
            "H": torch.linspace(-10, 10, 100),
            "wavelength": 1.0,
        }
    )
    test_parameter = torch.nn.Parameter(torch.tensor(123.0))
    element = ElementToTest(sim_params, test_parameter=test_parameter, test_buffer=None)

    # check if parameter has been registered
    assert hasattr(element, "test_parameter")
    assert getattr(element, "test_parameter") in element.parameters()

    # check if parameter is automatically transferred to device
    element.to(device)
    assert getattr(element, "test_parameter").device.type == device

    # test tensor as a parameter
    test_parameter = torch.tensor(123.0)
    element = ElementToTest(sim_params, test_parameter=test_parameter, test_buffer=None)

    # check if test_parameter has been registered as a buffer
    assert hasattr(element, "test_parameter")
    assert getattr(element, "test_parameter") not in element.parameters()
    assert getattr(element, "test_parameter") in element.buffers()

    # test if a parameter cannot be registered with a tensor on a device
    # distinct from the simulation parameters' device
    if device != "cpu":
        with pytest.raises(ValueError):
            element = ElementToTest(
                sim_params, test_parameter=test_parameter.to(device), test_buffer=None
            )


def test_to_specs():
    """
    Tests the conversion of an element to specifications.

        This test creates a simulation parameter set and an element with a
        test parameter, then asserts that the `to_specs` method generates a list
        containing a single specification for the test parameter, and that this
        specification contains a representation of type ReprRepr.

        Args:
            None

        Returns:
            None
    """
    sim_params = svetlanna.SimulationParameters(
        {
            "W": torch.linspace(-10, 10, 100),
            "H": torch.linspace(-10, 10, 100),
            "wavelength": 1.0,
        }
    )
    test_parameter = torch.nn.Parameter(torch.tensor(123.0))
    element = ElementToTest(sim_params, test_parameter=test_parameter, test_buffer=None)

    specs = list(element.to_specs())
    assert len(specs) == 1
    assert specs[0].parameter_name == "test_parameter"

    representations = list(specs[0].representations)
    assert len(representations) == 1
    assert isinstance(representations[0], svetlanna.specs.ReprRepr)


def test_make_buffer_pattern():
    """
    Tests the creation of a buffer pattern using make_buffer.

        This test instantiates an ElementToTest object with simulation parameters and
        asserts that calling make_buffer returns an instance of _BufferedValueContainer.
        It also checks for expected warnings when assigning a buffered value to another attribute.

        Args:
            None

        Returns:
            None
    """
    sim_params = svetlanna.SimulationParameters(
        {
            "W": torch.linspace(-10, 10, 100),
            "H": torch.linspace(-10, 10, 100),
            "wavelength": 1.0,
        }
    )
    element = ElementToTest(sim_params, test_parameter=None, test_buffer=None)

    assert isinstance(element.make_buffer("x", None), _BufferedValueContainer)

    with pytest.warns(
        match="You set the attribute y with an object of internal type _BufferedValueContainer. Make sure this is the intended behavior."
    ):
        element.y = element.make_buffer("x", None)


def test_repr_html():
    """
    Tests the HTML representation of an element.

        This test instantiates a simulation and an ElementToTest object,
        then asserts that the _repr_html_() method returns a string.

        Parameters:
            None

        Returns:
            None
    """
    sim_params = svetlanna.SimulationParameters(
        {
            "W": torch.linspace(-10, 10, 100),
            "H": torch.linspace(-10, 10, 100),
            "wavelength": 1.0,
        }
    )
    element = ElementToTest(sim_params, test_parameter=None, test_buffer=None)

    assert isinstance(element._repr_html_(), str)
