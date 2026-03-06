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
from typing import Mapping, Sequence, cast, Literal, Union, Protocol, SupportsIndex
from types import EllipsisType
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

    elements: traitlets.List[dict] = traitlets.List([]).tag(sync=True)  # type: ignore
    structure_html = traitlets.Unicode("").tag(sync=True)  # type: ignore


class SpecsWidget(anywidget.AnyWidget):
    _esm = STATIC_FOLDER / "specs_widget.js"
    _css = STATIC_FOLDER / "setup_widget.css"

    elements: traitlets.List[dict] = traitlets.List([]).tag(sync=True)  # type: ignore
    structure_html = traitlets.Unicode("").tag(sync=True)  # type: ignore


@dataclass(frozen=True, slots=True)
class ElementHTML:
    """Representation of an element in HTML format."""

    element_type: str | None
    html: str


class WidngetHTMLMethod(Protocol):
    def __call__(
        self,
        index: int,
        name: str,
        element_type: str | None,
        subelements: list[ElementHTML],
    ) -> str: ...

    """Render an element as HTML for widget display.

    Parameters
    ----------
    index : int
        Unique element index.
        Should be used as the id of an HTML element containing the element.
    name : str
        Human-readable element name.
    element_type : str | None
        Human-readable subelement type, if any.
    subelements : list[ElementHTML]
        Already rendered child elements.

    Returns
    -------
    str
        Rendered HTML representation of the element.
    """


def default_widget_html_method(
    index: int, name: str, element_type: str | None, subelements: list[ElementHTML]
) -> str:
    """Render a `Specsable` element with the default widget template.

    See `WidngetHTMLMethod` for parameter details.
    """
    return jinja_env.get_template("widget_default.html.jinja").render(
        index=index, name=name, subelements=subelements
    )


def _get_widget_html_method(element: Specsable) -> WidngetHTMLMethod:
    """Return a widget HTML renderer for an element.

    If the element defines `_widget_html_`, that method is returned.
    Otherwise, `default_widget_html_method` is used.

    Parameters
    ----------
    element : Specsable
        The element

    Returns
    -------
    WidngetHTMLMethod
        `_widget_html_` method
    """
    if hasattr(element, "_widget_html_"):
        return getattr(element, "_widget_html_")

    return default_widget_html_method


def _subelements_html(subelements: list[_ElementInTree]) -> list[ElementHTML]:
    """Generate rendered HTML for all provided tree nodes.

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
    """Generate HTML for the setup structure tree.

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
    """Display setup structure in an IPython environment.

    This helper renders only the hierarchy of elements (without parameter specs)
    and is useful for quick notebook previews.

    Parameters
    ----------
    *specsable : Specsable
        One or more `Specsable` objects to display.

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
    src="show_structure.html"
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
    """Display setup structure with interactive specs preview.

    Returns
    -------
    SpecsWidget
        Widget with element tree and per-element specs HTML.

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
    src="show_specs.html"
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


Index = Union[SupportsIndex | slice | EllipsisType | int | Sequence[int]]


def draw_wavefront(
    wavefront: torch.Tensor,
    simulation_parameters: SimulationParameters,
    types_to_plot: tuple[StepwisePlotTypes, ...] = ("I", "phase"),
    slices_to_plot: Mapping[str, Index | tuple[Index, ...]] | None = None,
) -> bytes:
    """Render wavefront slices into a JPEG image.

    The function applies optional axis slicing and draws one or more field
    representations (`"A"`, `"I"`, `"phase"`, `"Re"`, `"Im"`).
    Only a 2D result (`x`, `y`) can be plotted after slicing.

    Parameters
    ----------
    wavefront : Tensor
        Input wavefront tensor.
    simulation_parameters : SimulationParameters
        Simulation parameters
    types_to_plot : tuple[StepwisePlotTypes, ...], optional
        Field properties to plot, by default (`"I"`, `"phase"`).
    slices_to_plot : Mapping[str, Index | tuple[Index, ...]] | None, optional
        Axis slices to apply before plotting. Use `"_"` for unnamed axes.
        Default is `None` (plot the full wavefront).

    Returns
    -------
    bytes
        JPEG bytes of the rendered figure.

    """
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    import numpy as np

    slices: list[Index] = [slice(None)] * wavefront.ndim

    if axes_slices := (slices_to_plot or {}).get("_", None):
        if not isinstance(axes_slices, tuple):
            raise ValueError(
                f"For axes with no name ('_'), the value should be a tuple of slices, but got {type(axes_slices)}."
            )
        for i, axis_slice in enumerate(axes_slices):
            slices[i] = axis_slice

    for axis_name, axis_slice in (slices_to_plot or {}).items():
        if axis_name == "_":
            continue

        axis_slice = cast(Index, axis_slice)
        idx = simulation_parameters.index(axis_name)
        slices[idx] = axis_slice

    for i, slice_ in enumerate(reversed(slices)):
        additional_slice = (slice(None),) * i
        wavefront = wavefront[..., slice_, *additional_slice]

    wf = wavefront.numpy(force=True)

    x_slice = slices[simulation_parameters.index("x")]
    y_slice = slices[simulation_parameters.index("y")]
    x = simulation_parameters.x[x_slice].numpy(force=True)
    y = simulation_parameters.y[y_slice].numpy(force=True)

    if simulation_parameters.index("x") < simulation_parameters.index("y"):
        wf = wf.T

    if not len(wf.shape) == 2:
        raise ValueError(
            f"Currently only wavefronts of shape (Ny, Nx) are supported for plotting, but got {wf.shape}."
        )

    if len(x) > 1:
        X = np.empty(len(x) + 1, dtype=x.dtype)
        dx = np.diff(x)

        X[1:-1] = x[:-1] + dx / 2
        X[0] = x[0] - dx[0] / 2
        X[-1] = x[-1] + dx[-1] / 2
    else:
        X = np.empty(len(x) + 1, dtype=x.dtype)

        dy = y[-1] - y[0]
        if dy == 0:
            dy = 1
        X[0] = x[0] - dy / 2
        X[1] = x[0] + dy / 2

    if len(y) > 1:
        Y = np.empty(len(y) + 1, dtype=y.dtype)
        dy = np.diff(y)

        Y[1:-1] = y[:-1] + dy / 2
        Y[0] = y[0] - dy[0] / 2
        Y[-1] = y[-1] + dy[-1] / 2
    else:
        Y = np.empty(len(y) + 1, dtype=y.dtype)

        dy = y[-1] - y[0]
        if dy == 0:
            dy = 1
        Y[0] = y[0] - dy / 2
        Y[1] = y[0] + dy / 2

    n_plots = len(types_to_plot)

    vmax = np.abs(wf).max()
    vmin = -vmax

    figure, ax = plt.subplots(1, n_plots, figsize=(2 + 3 * n_plots, 3), dpi=120)

    try:
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
                    np.abs(wf),
                    cmap="viridis",
                )
                axes.set_title("Amplitude")

            elif plot_type == "I":
                # Plot the wavefront intensity
                axes.pcolorfast(
                    X,
                    Y,
                    (np.abs(wf) ** 2),
                    cmap="magma",
                )
                axes.set_title("Intensity")

            elif plot_type == "phase":
                # Plot the wavefront phase
                axes.pcolorfast(
                    X,
                    Y,
                    np.angle(wf),
                    vmin=-np.pi,
                    vmax=np.pi,
                    cmap="twilight",
                )
                axes.set_title("Phase")

            elif plot_type == "Re":
                # Plot the wavefront real part
                axes.pcolorfast(
                    X,
                    Y,
                    wf.real,
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
                    wf.imag,
                    vmax=vmax,
                    vmin=vmin,
                    cmap="seismic",
                )
                axes.set_title("Imaginary part")

            axes.set_box_aspect(1)
            axes.set_aspect("equal")
            if len(x) == 1:
                axes.set_xticks([x[0]])
            if len(y) == 1:
                axes.set_yticks([y[0]])

        stream = BytesIO()

        plt.tight_layout()
        figure.savefig(stream, format="jpeg")
        plt.close(figure)

        return stream.getvalue()

    except Exception as e:
        plt.close(figure)
        raise e


def show_stepwise_forward(
    *specsable: Specsable,
    input: torch.Tensor,
    simulation_parameters: SimulationParameters,
    types_to_plot: tuple[StepwisePlotTypes, ...] = ("I", "phase"),
    slices_to_plot: Mapping[str, Index | tuple[Index, ...]] | None = None,
) -> StepwiseForwardWidget:
    """Display stepwise wavefront propagation for setup elements.

    The function registers forward hooks on `torch.nn.Module` elements,
    runs a forward pass for each provided root element, captures intermediate
    outputs, renders them as images, and returns an interactive widget.

    Parameters
    ----------
    input : torch.Tensor
        Input wavefront.
    simulation_parameters : SimulationParameters
        Simulation parameters
    types_to_plot : tuple[StepwisePlotTypes, ...], optional
        Field properties to plot, by default (`"I"`, `"phase"`).
    slices_to_plot : Mapping[str, Index | tuple[Index, ...]] | None, optional
        Axis slices to apply before plotting each captured output.
        Default is `None`.

    Returns
    -------
    StepwiseForwardWidget
        Widget containing setup structure and captured per-element outputs.

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
    src="show_stepwise_forward.html"
    style="width:100%; height:500px; border: 0; color-scheme: inherit;" allowtransparency="true"></iframe>

    ---

    Plot only the central area:
    ```python
    show_stepwise_forward(
        setup,
        input=input_wavefront,
        simulation_parameters=sim_params,
        types_to_plot=("I", "phase", "Re"),
        slices_to_plot={
            "x": (sim_params.x > -0.5) & (sim_params.x < 0.5),
            "y": (sim_params.y > -0.5) & (sim_params.y < 0.5),
        },
    )
    ```

    Equivalent tensor indexing:
    ```python linenums="0"
    wavefront[(sim_params.y > -0.5) & (sim_params.y < 0.5), (sim_params.x > -0.5) & (sim_params.x < 0.5)]
    ```

    ---

    Slice additional named axes:
    ```python
    show_stepwise_forward(
        setup,
        input=input_wavefront,
        simulation_parameters=sim_params,
        types_to_plot=("I", "phase", "Re"),
        slices_to_plot={"wavelength": 0},
    )
    ```

    Equivalent tensor indexing:
    ```python linenums="0"
    wavefront[0, :, :]
    ```

    ---

    Slice unnamed axes (for example, batch):
    ```python
    show_stepwise_forward(
        setup,
        input=input_wavefront,
        simulation_parameters=sim_params,
        types_to_plot=("I", "phase", "Re"),
        slices_to_plot={
            "_": (0, 0),
        },
    )
    ```

    The `"_"` value must be a tuple in the order of unnamed axes.
    Equivalent tensor indexing:
    ```python linenums="0"
    wavefront[0, 0, :, :]
    ```

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
                            slices_to_plot=slices_to_plot,
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
