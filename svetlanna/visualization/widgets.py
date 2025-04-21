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


STATIC_FOLDER = pathlib.Path(__file__).parent / 'static'
TEMPLATES_FOLDER = pathlib.Path(__file__).parent / 'templates'

jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATES_FOLDER),
    autoescape=select_autoescape()
)


StepwisePlotTypes = Union[
    Literal['A'],
    Literal['I'],
    Literal['phase'],
    Literal['Re'],
    Literal['Im']
]


class StepwiseForwardWidget(anywidget.AnyWidget):
    _esm = STATIC_FOLDER / 'stepwise_forward_widget.js'
    _css = STATIC_FOLDER / 'setup_widget.css'

    elements = traitlets.List([]).tag(sync=True)
    structure_html = traitlets.Unicode('').tag(sync=True)


class SpecsWidget(anywidget.AnyWidget):
    _esm = STATIC_FOLDER / 'specs_widget.js'
    _css = STATIC_FOLDER / 'setup_widget.css'

    elements = traitlets.List([]).tag(sync=True)
    structure_html = traitlets.Unicode('').tag(sync=True)


@dataclass(frozen=True, slots=True)
class ElementHTML:
    """Representation of an element in HTML format."""
    element_type: str | None
    html: str


def default_widget_html_method(
    index: int,
    name: str,
    element_type: str | None,
    subelements: list[ElementHTML]
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
    return jinja_env.get_template('widget_default.html.jinja').render(
        index=index, name=name, subelements=subelements
    )


def _get_widget_html_method(
    element: Specsable
) -> Callable[..., str]:
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
    if hasattr(element, '_widget_html_'):
        return getattr(element, '_widget_html_')

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
            subelements=_subelements_html(subelement.children)
        )

        res.append(
            ElementHTML(
                subelement.subelement_type,
                html=raw_subelement_html
            )
        )

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

    return jinja_env.get_template(
        'widget_structure_container.html.jinja'
    ).render(elements_html=elements_html)


def show_structure(*specsable: Specsable):
    """Display a setup structure using IPython's HTML display.
    Useful for previewing specs hierarchies in notebooks.
    """
    try:
        from IPython.display import HTML, display

        # Generate HTML
        elements = _ElementsIterator(*specsable, directory='')
        structure_html = generate_structure_html(elements.tree)

        # Display HTML
        display(HTML(structure_html))

    except ModuleNotFoundError:
        warn("Currently only display via ipython is supported.")


def show_specs(*specsable: Specsable) -> SpecsWidget:
    """Display a setup structure with interactive specs preview

    Returns
    -------
    SpecsWidget
        The widget
    """

    elements = _ElementsIterator(*specsable, directory='')

    # Prepare elements data for widget
    elements_json = []
    for element_index, element, writer_context_generator in elements:
        stream = StringIO('')
        # Write element's parameter specs to the stream
        write_specs_to_html(
            element, element_index, writer_context_generator, stream
        )

        elements_json.append(
            {
                'index': element_index,
                'name': element.__class__.__name__,
                'specs_html': stream.getvalue()
            }
        )

    # Generate structure HTML
    structure_html = generate_structure_html(elements.tree)

    # Create a widget
    widget = SpecsWidget(
        structure_html=structure_html,
        elements=elements_json
    )

    return widget


def draw_wavefront(
    wavefront: torch.Tensor,
    simulation_parameters: SimulationParameters,
    types_to_plot: tuple[StepwisePlotTypes, ...] = ('I', 'phase')
) -> bytes:
    """Show field propagation in the setup via widget.
    Currently only wavefronts of shape `(W, H)` are supported.

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

    stream = BytesIO()

    width = simulation_parameters.axes.W.cpu()
    height = simulation_parameters.axes.H.cpu()

    n_plots = len(types_to_plot)

    width_to_height = (
        width.max() - width.min()
    ) / (
        height.max() - height.min()
    )

    figure, ax = plt.subplots(
        1, n_plots,
        figsize=(2+3*n_plots*width_to_height, 3),
        dpi=120
    )

    for i, plot_type in enumerate(types_to_plot):
        if isinstance(ax, Axes):
            axes = ax
        else:
            axes = ax[i]
            axes = cast(Axes, axes)

        if plot_type == 'A':
            # Plot the wavefront amplitude
            axes.pcolorfast(
                width,
                height,
                wavefront.abs().numpy(force=True)
            )
            axes.set_title('Amplitude')

        elif plot_type == 'I':
            # Plot the wavefront intensity
            axes.pcolorfast(
                width,
                height,
                (wavefront.abs()**2).numpy(force=True)
            )
            axes.set_title('Intensity')

        elif plot_type == 'phase':
            # Plot the wavefront phase
            axes.pcolorfast(
                width,
                height,
                wavefront.angle().numpy(force=True),
                vmin=-torch.pi,
                vmax=torch.pi,
            )
            axes.set_title('Phase')

        elif plot_type == 'Re':
            # Plot the wavefront real part
            axes.pcolorfast(
                width,
                height,
                wavefront.real.numpy(force=True),
            )
            axes.set_title('Real part')

        elif plot_type == 'Im':
            # Plot the wavefront imaginary part
            axes.pcolorfast(
                width,
                height,
                wavefront.imag.numpy(force=True),
            )
            axes.set_title('Imaginary part')

        axes.set_aspect('equal')

    plt.tight_layout()
    figure.savefig(stream)
    plt.close(figure)

    return stream.getvalue()


def show_stepwise_forward(
    *specsable: Specsable,
    input: torch.Tensor,
    simulation_parameters: SimulationParameters,
    types_to_plot: tuple[StepwisePlotTypes, ...] = ('I', 'phase')
) -> StepwiseForwardWidget:
    """Display the wavefront propagation through a setup structure
    using a widget interface. Currently only wavefronts
    of shape `(W, H)` are supported.

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
    """

    elements_to_call = tuple(s for s in specsable)
    elements = _ElementsIterator(*elements_to_call, directory='')

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
        elements = _ElementsIterator(*elements_to_call, directory='')

        elements_json = []
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
                            types_to_plot=types_to_plot
                        )
                    ).decode()
                except Exception as e:
                    output_image = f'\n{e}'
            else:
                output_image = None

            elements_json.append(
                {
                    'index': element_index,
                    'name': element.__class__.__name__,
                    'output_image': output_image
                }
            )

        # Generate structure HTML
        structure_html = generate_structure_html(elements.tree)

        # Create a widget
        widget = StepwiseForwardWidget(
            structure_html=structure_html,
            elements=elements_json
        )

        return widget

    except Exception as e:
        raise e

    finally:
        # Remove forward hooks
        for hook in registered_hooks:
            hook.remove()
