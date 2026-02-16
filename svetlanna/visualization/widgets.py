import anywidget
import traitlets
import pathlib
from dataclasses import dataclass
from jinja2 import Environment, FileSystemLoader, select_autoescape
from ..specs import Specsable
from ..specs.specs_writer import _ElementInTree, _ElementsIterator
from ..specs.specs_writer import write_specs_to_html
from io import StringIO, BytesIO
from warnings import warn
from typing import cast, Literal, Union, Callable
import torch
from torch.utils.hooks import RemovableHandle
from ..simulation_parameters import SimulationParameters
import base64


STATIC_FOLDER = pathlib.Path(__file__).parent / "static"
TEMPLATES_FOLDER = pathlib.Path(__file__).parent / "templates"

jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATES_FOLDER), autoescape=select_autoescape()
)


StepwisePlotTypes = Union[
    Literal["A"], Literal["I"], Literal["phase"], Literal["Re"], Literal["Im"]
]


class StepwiseForwardWidget(anywidget.AnyWidget):
    _esm = STATIC_FOLDER / "stepwise_forward_widget.js"
    _css = STATIC_FOLDER / "setup_widget.css"

    elements: traitlets.List[dict] = traitlets.List([]).tag(sync=True)
    structure_html = traitlets.Unicode("").tag(sync=True)


class SpecsWidget(anywidget.AnyWidget):
    _esm = STATIC_FOLDER / "specs_widget.js"
    _css = STATIC_FOLDER / "setup_widget.css"

    elements: traitlets.List[dict] = traitlets.List([]).tag(sync=True)
    structure_html = traitlets.Unicode("").tag(sync=True)


@dataclass(frozen=True, slots=True)
class ElementHTML:
    """Representation of an element in HTML format."""

    element_type: str | None
    html: str


def default_widget_html_method(
    index: int, name: str, element_type: str | None, subelements: list[ElementHTML]
) -> str:
    """Default `_widget_html_` method used for rendering `Specsable` elements.

    Parameters
    ----------
    index : int
        The unique index of the element.
        Should be used as an id of HTML element containing the element.
    name : str
        Human readable name of the element
    element_type : str | None
        Human readable type of the element as a subelement, if any
    subelements : list[ElementHTML]
        Subelements of the element.

    Returns
    -------
    str
        rendered HTML
    """
    return jinja_env.get_template("widget_default.html.jinja").render(
        index=index, name=name, subelements=subelements
    )


def _get_widget_html_method(element: Specsable) -> Callable[..., str]:
    """Returns `_widget_html_` method based on type of element.

    Parameters
    ----------
    element : Specsable
        The element

    Returns
    -------
    Any
        `_widget_html_` method
    """
    if hasattr(element, "_widget_html_"):
        return getattr(element, "_widget_html_")

    return default_widget_html_method


def _subelements_html(subelements: list[_ElementInTree]) -> list[ElementHTML]:
    """Generate rendered HTML for all elements of provided list.

    Parameters
    ----------
    subelements : list[_ElementInTree]
        Elements in the elements tree

    Returns
    -------
    list[ElementHTML]
        List of rendered HTML
    """
    res = []

    for subelement in subelements:
        widget_html_method = _get_widget_html_method(subelement.element)

        raw_subelement_html = widget_html_method(
            index=subelement.element_index,
            name=subelement.element.__class__.__name__,
            element_type=subelement.subelement_type,
            subelements=_subelements_html(subelement.children),
        )

        res.append(ElementHTML(subelement.subelement_type, html=raw_subelement_html))

    return res


def generate_structure_html(subelements: list[_ElementInTree]) -> str:
    """Generate HTML for a setup structure.

    Parameters
    ----------
    subelements : list[_ElementInTree]
        Elements tree

    Returns
    -------
    str
        Rendered HTML
    """

    elements_html = _subelements_html(subelements)

    return jinja_env.get_template("widget_structure_container.html.jinja").render(
        elements_html=elements_html
    )


def show_structure(*specsable: Specsable):
    """Display a setup structure using IPython's HTML display.
    Useful for previewing specs hierarchies in notebooks.

    Parameters
    ----------
    *specsable : Specsable
        One or more specsable elements to display

    Examples
    --------
    ```python
    import svetlanna as sv
    import torch
    from svetlanna.visualization import show_structure

    Nx = Ny = 128
    sim_params = sv.SimulationParameters(
        x=torch.linspace(-1, 1, Nx),
        y=torch.linspace(-1, 1, Ny),
        wavelength=0.1,
    )

    setup = sv.LinearOpticalSetup(
        [
            sv.elements.RectangularAperture(sim_params, width=0.5, height=0.5),
            sv.elements.FreeSpace(sim_params, distance=0.2, method="AS"),
            sv.elements.DiffractiveLayer(sim_params, mask=torch.rand(Ny, Nx), mask_norm=1),
            sv.elements.FreeSpace(sim_params, distance=0.2, method="AS"),
        ]
    )

    show_structure(setup)
    ```
    Output (in IPython environment):
    <iframe
    src="/reference/visualization/show_structure.html"
    style="width:100%; height:150px; border: 0; color-scheme: inherit;" allowtransparency="true"></iframe>
    """
    try:
        from IPython.display import HTML, display

        # Generate HTML
        elements = _ElementsIterator(*specsable, directory="")
        structure_html = generate_structure_html(elements.tree)

        # Display HTML
        display(HTML(structure_html))

    except ImportError:
        warn("Currently only display via ipython is supported.")


def show_specs(*specsable: Specsable) -> SpecsWidget:
    """Display a setup structure with interactive specs preview

    Returns
    -------
    SpecsWidget
        The widget

    Examples
    --------
    ```python
    import svetlanna as sv
    import torch
    from svetlanna.visualization import show_specs

    Nx = Ny = 128
    sim_params = sv.SimulationParameters(
        x=torch.linspace(-1, 1, Nx),
        y=torch.linspace(-1, 1, Ny),
        wavelength=0.1,
    )

    setup = sv.LinearOpticalSetup(
        [
            sv.elements.RectangularAperture(sim_params, width=0.5, height=0.5),
            sv.elements.FreeSpace(sim_params, distance=0.2, method="AS"),
            sv.elements.DiffractiveLayer(sim_params, mask=torch.rand(Ny, Nx), mask_norm=1),
            sv.elements.FreeSpace(sim_params, distance=0.2, method="AS"),
        ]
    )

    show_specs(setup)
    ```
    Output (in IPython environment):
    <iframe
    src="/reference/visualization/show_specs.html"
    style="width:100%; height:500px; border: 0; color-scheme: inherit;" allowtransparency="true"></iframe>

    """

    elements = _ElementsIterator(*specsable, directory="")

    # Prepare elements data for widget
    elements_json = []
    for element_index, element, writer_context_generator in elements:
        stream = StringIO("")
        # Write element's parameter specs to the stream
        write_specs_to_html(element, element_index, writer_context_generator, stream)

        elements_json.append(
            {
                "index": element_index,
                "name": element.__class__.__name__,
                "specs_html": stream.getvalue(),
            }
        )

    # Generate structure HTML
    structure_html = generate_structure_html(elements.tree)

    # Create a widget
    widget = SpecsWidget(structure_html=structure_html, elements=elements_json)

    return widget


def draw_wavefront(
    wavefront: torch.Tensor,
    simulation_parameters: SimulationParameters,
    types_to_plot: tuple[StepwisePlotTypes, ...] = ("I", "phase"),
) -> bytes:
    """Show field propagation in the setup via widget.
    Currently only wavefronts of shape `(x, y)` are supported.

    Parameters
    ----------
    wavefront : Tensor
        The Input wavefront
    simulation_parameters : SimulationParameters
        Simulation parameters
    types_to_plot : tuple[StepwisePlotTypes, ...], optional
        Field properties to plot, by default ('I', 'phase')

    Returns
    -------
    bytes
        byte-coded image
    """
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    import numpy as np

    stream = BytesIO()

    x = simulation_parameters.x.numpy(force=True)
    y = simulation_parameters.y.numpy(force=True)

    X = np.empty(len(x) + 1, dtype=x.dtype)
    dx = np.diff(x)

    X[1:-1] = x[:-1] + dx / 2
    X[0] = x[0] - dx[0] / 2
    X[-1] = x[-1] + dx[-1] / 2

    Y = np.empty(len(y) + 1, dtype=y.dtype)
    dy = np.diff(y)

    Y[1:-1] = y[:-1] + dy / 2
    Y[0] = y[0] - dy[0] / 2
    Y[-1] = y[-1] + dy[-1] / 2

    n_plots = len(types_to_plot)

    width_to_height = (X.max() - X.min()) / (Y.max() - Y.min())
    vmax = wavefront.abs().max().item()
    vmin = -vmax

    figure, ax = plt.subplots(
        1, n_plots, figsize=(2 + 3 * n_plots * width_to_height, 3), dpi=120
    )

    for i, plot_type in enumerate(types_to_plot):
        if isinstance(ax, Axes):
            axes = ax
        else:
            axes = ax[i]
            axes = cast(Axes, axes)

        if plot_type == "A":
            # Plot the wavefront amplitude
            axes.pcolorfast(
                X,
                Y,
                wavefront.abs().numpy(force=True),
                cmap="viridis",
            )
            axes.set_title("Amplitude")

        elif plot_type == "I":
            # Plot the wavefront intensity
            axes.pcolorfast(
                X,
                Y,
                (wavefront.abs() ** 2).numpy(force=True),
                cmap="magma",
            )
            axes.set_title("Intensity")

        elif plot_type == "phase":
            # Plot the wavefront phase
            axes.pcolorfast(
                X,
                Y,
                wavefront.angle().numpy(force=True),
                vmin=-torch.pi,
                vmax=torch.pi,
                cmap="twilight",
            )
            axes.set_title("Phase")

        elif plot_type == "Re":
            # Plot the wavefront real part
            axes.pcolorfast(
                X,
                Y,
                wavefront.real.numpy(force=True),
                vmax=vmax,
                vmin=vmin,
                cmap="seismic",
            )
            axes.set_title("Real part")

        elif plot_type == "Im":
            # Plot the wavefront imaginary part
            axes.pcolorfast(
                X,
                Y,
                wavefront.imag.numpy(force=True),
                vmax=vmax,
                vmin=vmin,
                cmap="seismic",
            )
            axes.set_title("Imaginary part")

        axes.set_aspect("equal")

    plt.tight_layout()
    figure.savefig(stream)
    plt.close(figure)

    return stream.getvalue()


def show_stepwise_forward(
    *specsable: Specsable,
    input: torch.Tensor,
    simulation_parameters: SimulationParameters,
    types_to_plot: tuple[StepwisePlotTypes, ...] = ("I", "phase"),
) -> StepwiseForwardWidget:
    """Display the wavefront propagation through a setup structure
    using a widget interface. Currently only wavefronts
    of shape `(x, y)` are supported.

    Parameters
    ----------
    input : torch.Tensor
        The Input wavefront
    simulation_parameters : SimulationParameters
        Simulation parameters
    types_to_plot : tuple[StepwisePlotTypes, ...], optional
        Field properties to plot, by default ('I', 'phase')

    Returns
    -------
    StepwiseForwardWidget
        The widget

    Examples
    --------
    ```python
    import svetlanna as sv
    import torch
    from svetlanna.visualization import show_stepwise_forward

    Nx = Ny = 128
    sim_params = sv.SimulationParameters(
        x=torch.linspace(-1, 1, Nx),
        y=torch.linspace(-1, 1, Ny),
        wavelength=0.1,
    )

    setup = sv.LinearOpticalSetup(
        [
            sv.elements.RectangularAperture(sim_params, width=0.5, height=0.5),
            sv.elements.FreeSpace(sim_params, distance=0.2, method="AS"),
            sv.elements.DiffractiveLayer(sim_params, mask=torch.rand(Ny, Nx), mask_norm=1),
            sv.elements.FreeSpace(sim_params, distance=0.2, method="AS"),
        ]
    )

    input_wavefront = sv.Wavefront.plane_wave(sim_params)
    show_stepwise_forward(
        setup,
        input=input_wavefront,
        simulation_parameters=sim_params,
        types_to_plot=("I", "phase", "Re"),
    )
    ```
    Output (in IPython environment):
    <iframe
    src="/reference/visualization/show_stepwise_forward.html"
    style="width:100%; height:500px; border: 0; color-scheme: inherit;" allowtransparency="true"></iframe>

    """

    elements_to_call = tuple(s for s in specsable)
    elements = _ElementsIterator(*elements_to_call, directory="")

    outputs = {}

    def capture_output_hook(module, args, output):
        # The hook that captures the output
        if isinstance(output, torch.Tensor):
            outputs[module] = output.clone()

    registered_hooks: list[RemovableHandle] = []

    try:
        # Iterate over all elements and register forward hooks for all modules
        for _, element, context_generator in elements:
            for _ in context_generator:
                pass

            if isinstance(element, torch.nn.Module):
                registered_hooks.append(
                    element.register_forward_hook(
                        capture_output_hook, with_kwargs=False
                    )
                )

        # Call forward methods in all specsables
        with torch.no_grad():
            for element in elements_to_call:
                if isinstance(element, torch.nn.Module):
                    element(input)

        # Prepare elements data for widget
        elements_json: list[dict] = []
        for element_index, element, context_generator in elements:
            for _ in context_generator:
                pass

            # Draw the wavefront if any
            if element in outputs:
                try:
                    output_image = base64.b64encode(
                        draw_wavefront(
                            wavefront=outputs[element],
                            simulation_parameters=simulation_parameters,
                            types_to_plot=types_to_plot,
                        )
                    ).decode()
                except Exception as e:
                    output_image = f"\n{e}"
            else:
                output_image = None

            elements_json.append(
                {
                    "index": element_index,
                    "name": element.__class__.__name__,
                    "output_image": output_image,
                }
            )

        # Generate structure HTML
        structure_html = generate_structure_html(elements.tree)

        # Create a widget
        widget = StepwiseForwardWidget(
            structure_html=structure_html, elements=elements_json
        )

        return widget

    finally:
        # Remove forward hooks
        for hook in registered_hooks:
            hook.remove()
