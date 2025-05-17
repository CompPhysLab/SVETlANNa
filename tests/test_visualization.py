import svetlanna
from svetlanna.visualization.widgets import default_widget_html_method
from svetlanna.visualization.widgets import generate_structure_html
from svetlanna.visualization import show_structure, show_specs
from svetlanna.visualization.widgets import draw_wavefront
from svetlanna.visualization import show_stepwise_forward
from svetlanna.specs.specs_writer import _ElementInTree
from svetlanna.visualization.widgets import SpecsWidget, StepwiseForwardWidget
import torch
import builtins
import pytest


def test_html_element():
    """
    Tests that the HTML representation of a FreeSpace element is generated."""
    sim_params = svetlanna.SimulationParameters(
        {"W": torch.tensor([0]), "H": torch.tensor([0]), "wavelength": 1}
    )
    element = svetlanna.elements.FreeSpace(sim_params, distance=1, method="AS")

    assert element._repr_html_()


def test_default_widget_html_method():
    """
    Tests the default widget HTML method.

        Args:
            None

        Returns:
            None
    """
    assert default_widget_html_method(123, "test", "element_type", [])


def test_generate_structure_html():
    """
    Tests the generation of HTML structure for a simple simulation element.

        This test creates a basic simulation setup with a FreeSpace element and
        a nested NoWidgetHTMLElement, then asserts that generate_structure_html
        returns without errors when given this structure.

        Args:
            None

        Returns:
            None
    """
    sim_params = svetlanna.SimulationParameters(
        {"W": torch.tensor([0]), "H": torch.tensor([0]), "wavelength": 1}
    )
    element = svetlanna.elements.FreeSpace(sim_params, distance=1, method="AS")

    class NoWidgetHTMLElement:
        def to_specs(self):
            return []

    assert generate_structure_html(
        [_ElementInTree(element, 0, [_ElementInTree(NoWidgetHTMLElement(), 0, [])])]
    )


def test_show_structure(monkeypatch):
    """
    Tests the show_structure function's behavior with and without IPython."""
    import IPython.display

    # monkeypatch IPython.display.display
    displayed = False

    def set_displayed():
        nonlocal displayed
        displayed = True

    monkeypatch.setattr(IPython.display, "display", lambda _: set_displayed())

    # Test if the HTML has been displayed
    displayed = False
    show_structure()
    assert displayed

    # Test if the warning displayed in the case of IPython absence
    # monkeypatching import statement
    original_import = builtins.__import__

    def import_with_no_ipython(name, *args, **kwargs):
        if name == "IPython.display":
            raise ImportError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_no_ipython)

    warning_msg = "Currently only display via ipython is supported."
    with pytest.warns(UserWarning, match=warning_msg):
        displayed = False
        show_structure()
        assert not displayed


def test_show_specs():
    """
    Tests the show_specs function.

        This test creates a simple simulation setup with a FreeSpace element and
        verifies that the show_specs function returns a SpecsWidget containing
        information about the element.

        Args:
            None

        Returns:
            None
    """
    sim_params = svetlanna.SimulationParameters(
        {"W": torch.tensor([0]), "H": torch.tensor([0]), "wavelength": 1}
    )
    element = svetlanna.elements.FreeSpace(sim_params, distance=1, method="AS")

    widget = show_specs(element)
    assert isinstance(widget, SpecsWidget)
    assert len(widget.elements) == 1
    assert widget.elements[0]["name"] == "FreeSpace"


def test_draw_wavefront():
    """
    Tests the draw_wavefront function with different type combinations.

        This test creates a simple plane wavefront and then calls draw_wavefront
        with single types and all available types to ensure it functions correctly
        for various plotting configurations.

        Args:
            None

        Returns:
            bool: True if all assertions pass, indicating the function works as expected.
    """
    sim_params = svetlanna.SimulationParameters(
        {
            "W": torch.linspace(-1, 1, 10),
            "H": torch.linspace(-1, 1, 10),
            "wavelength": 1,
        }
    )
    wavefront = svetlanna.Wavefront.plane_wave(sim_params)

    # Single type
    types = ("A", "I", "phase", "Re", "Im")
    for t in types:
        assert draw_wavefront(wavefront, sim_params, types_to_plot=(t,))

    # All types
    assert draw_wavefront(wavefront, sim_params, types_to_plot=types)


def test_show_stepwise_forward():
    """
    Tests the show_stepwise_forward function with various elements.

        This test creates a simulation setup with different optical elements,
        including a valid FreeSpace element, an element that returns None, and
        an element that returns a tensor instead of an image. It then asserts
        that the resulting widget is a StepwiseForwardWidget, contains all three
        elements, and correctly represents their outputs in JSON format.

        Args:
            None

        Returns:
            None
    """
    sim_params = svetlanna.SimulationParameters(
        {
            "W": torch.linspace(-1, 1, 10),
            "H": torch.linspace(-1, 1, 10),
            "wavelength": 1,
        }
    )

    class NoneForwardElement(torch.nn.Module):
        def forward(self, x):
            return None

        def to_specs(self):
            return []

    class WrongTensorForwardElement(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([1, 2, 3.0])

        def to_specs(self):
            return []

    element1 = svetlanna.elements.FreeSpace(sim_params, distance=1, method="AS")
    element2 = NoneForwardElement()
    element3 = WrongTensorForwardElement()

    wavefront = svetlanna.Wavefront.plane_wave(sim_params)

    widget = show_stepwise_forward(
        element1, element2, element3, input=wavefront, simulation_parameters=sim_params
    )

    assert isinstance(widget, StepwiseForwardWidget)
    assert len(widget.elements) == 3

    element1_json = widget.elements[0]
    assert element1_json["name"] == "FreeSpace"
    assert element1_json["output_image"]

    element2_json = widget.elements[1]
    assert element2_json["name"] == "NoneForwardElement"
    assert element2_json["output_image"] is None

    element3_json = widget.elements[2]
    assert element3_json["name"] == "WrongTensorForwardElement"
    assert element3_json["output_image"][:1] == "\n"
