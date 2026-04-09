import svetlanna
from svetlanna.visualization.widgets import default_widget_html_method
from svetlanna.visualization.widgets import generate_structure_html
from svetlanna.visualization import show_structure, show_specs
from svetlanna.visualization.widgets import draw_wavefront
from svetlanna.visualization import show_stepwise_forward
from svetlanna.specs.specs_writer import _ElementInTree
from svetlanna.visualization.widgets import SpecsWidget, StepwiseForwardWidget
from svetlanna.visualization.widgets import _apply_slices
import torch
import builtins
import pytest
from unittest.mock import patch
import matplotlib.figure
import matplotlib.pyplot


BASE64_JPEG_HEADER = "/9j/"


def test_html_element():
    sim_params = svetlanna.SimulationParameters(
        {
            "x": torch.tensor([0]),
            "y": torch.tensor([0]),
            "wavelength": 1,
        }
    )
    element = svetlanna.elements.FreeSpace(sim_params, distance=1, method="AS")

    assert element._repr_html_()


def test_default_widget_html_method():
    assert default_widget_html_method(123, "test", "element_type", [])


def test_generate_structure_html():
    sim_params = svetlanna.SimulationParameters(
        {
            "x": torch.tensor([0]),
            "y": torch.tensor([0]),
            "wavelength": 1,
        }
    )
    element = svetlanna.elements.FreeSpace(sim_params, distance=1, method="AS")

    class NoWidgetHTMLElement:
        def to_specs(self):
            return []

    assert generate_structure_html(
        [_ElementInTree(element, 0, [_ElementInTree(NoWidgetHTMLElement(), 0, [])])]
    )


def test_show_structure(monkeypatch):
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
    sim_params = svetlanna.SimulationParameters(
        {
            "x": torch.tensor([0]),
            "y": torch.tensor([0]),
            "wavelength": 1,
        }
    )
    element = svetlanna.elements.FreeSpace(sim_params, distance=1, method="AS")

    widget = show_specs(element)
    assert isinstance(widget, SpecsWidget)
    assert len(widget.elements) == 1
    assert widget.elements[0]["name"] == "FreeSpace"


def test_draw_wavefront():
    sim_params = svetlanna.SimulationParameters(
        {
            "x": torch.linspace(-1, 1, 10),
            "y": torch.linspace(-1, 1, 10),
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

    sim_params = svetlanna.SimulationParameters(
        {
            "x": torch.linspace(-1, 1, 5),
            "y": torch.linspace(-1, 1, 10),
            "wavelength": 1,
        }
    )

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
    assert element1_json["output_image"].startswith(BASE64_JPEG_HEADER)

    element2_json = widget.elements[1]
    assert element2_json["name"] == "NoneForwardElement"
    assert element2_json["output_image"] is None

    element3_json = widget.elements[2]
    assert element3_json["name"] == "WrongTensorForwardElement"
    assert element3_json["output_image"][:1] == "\n"

    # ==== test with x and y axes swapped ====
    sim_params = svetlanna.SimulationParameters(
        {
            "y": torch.linspace(-1, 1, 10),
            "x": torch.linspace(-1, 1, 5),
            "wavelength": 1,
        }
    )

    element1 = svetlanna.elements.FreeSpace(sim_params, distance=1, method="AS")
    element2 = NoneForwardElement()
    element3 = WrongTensorForwardElement()

    wavefront = svetlanna.Wavefront.plane_wave(sim_params)

    widget_yx = show_stepwise_forward(
        element1, element2, element3, input=wavefront, simulation_parameters=sim_params
    )
    # the image always will be plotted with x as the horizontal axis and y as the vertical axis, so the output image should be the same for both widgets
    assert widget.elements[0]["output_image"] == widget_yx.elements[0]["output_image"]

    # ==== test with more axes ====
    sim_params = svetlanna.SimulationParameters(
        {
            "x": torch.linspace(-1, 1, 10),
            "y": torch.linspace(-1, 1, 10),
            "wavelength": torch.linspace(1, 2, 10),
        }
    )

    element1 = svetlanna.elements.FreeSpace(sim_params, distance=1, method="AS")

    wavefront = svetlanna.Wavefront.plane_wave(sim_params)

    widget = show_stepwise_forward(
        element1, input=wavefront, simulation_parameters=sim_params
    )
    assert (
        widget.elements[0]["output_image"]
        == "\nCurrently only wavefronts of shape (Ny, Nx) are supported for plotting, but got (10, 10, 10)."
    )

    # ==== test if x and y length of 1 works ====
    for Nx in [7, 1]:
        for Ny in [4, 1]:
            sim_params = svetlanna.SimulationParameters(
                {
                    "x": torch.linspace(-1, 1, Nx),
                    "y": torch.linspace(-1, 1, Ny),
                    "wavelength": 1,
                }
            )

            element1 = svetlanna.elements.FreeSpace(sim_params, distance=1, method="AS")

            wavefront = svetlanna.Wavefront.plane_wave(sim_params)

            widget = show_stepwise_forward(
                element1, input=wavefront, simulation_parameters=sim_params
            )
            assert widget.elements[0]["output_image"].startswith(BASE64_JPEG_HEADER)

    # ==== test if figure closes properly on error ====
    with patch.object(
        matplotlib.figure.Figure, "savefig", side_effect=RuntimeError("error message")
    ):
        with patch.object(matplotlib.pyplot, "close") as mock_close:
            sim_params = svetlanna.SimulationParameters(
                {
                    "x": torch.linspace(-1, 1, 10),
                    "y": torch.linspace(-1, 1, 10),
                    "wavelength": 1,
                }
            )

            element1 = svetlanna.elements.FreeSpace(sim_params, distance=1, method="AS")

            wavefront = svetlanna.Wavefront.plane_wave(sim_params)
            widget = show_stepwise_forward(
                element1, input=wavefront, simulation_parameters=sim_params
            )

            assert widget.elements[0]["output_image"] == "\nerror message"
            mock_close.assert_called_once()


def test_apply_slices():
    # test slices
    sim_params = svetlanna.SimulationParameters(
        {
            "x": torch.linspace(-1, 1, 4),
            "y": torch.linspace(-1, 1, 5),
            "wavelength": 1,
        }
    )
    wavefront = svetlanna.Wavefront.plane_wave(sim_params)

    # test default slicing
    wavefront_sliced, slices_list = _apply_slices(
        wavefront,
        {
            "x": slice(2, 5),
            "y": slice(3, 7),
        },
        sim_params,
    )
    torch.testing.assert_close(wavefront_sliced, wavefront[3:5, 2:4])
    assert slices_list == [slice(3, 7), slice(2, 5)]

    # Check if "_" works
    wavefront_sliced, slices_list = _apply_slices(
        wavefront,
        {
            "_": (slice(None), slice(None)),
        },
        sim_params,
    )
    torch.testing.assert_close(wavefront_sliced, wavefront)
    assert slices_list == [slice(None), slice(None)]

    # Check if "_" works works when there are more axes provided
    wavefront_sliced, slices_list = _apply_slices(
        wavefront,
        {
            "_": (0, 1, 2, 3, 4),
        },
        sim_params,
    )
    torch.testing.assert_close(wavefront_sliced, wavefront[0, 1])
    assert slices_list == [0, 1]

    # Check if there is an error when the slice is not a tuple for unnamed axes
    with pytest.raises(ValueError):
        _apply_slices(
            wavefront,
            {
                "_": 0,
            },
            sim_params,
        )
