import anywidget
import traitlets
import pathlib
from dataclasses import dataclass
from jinja2 import Environment, FileSystemLoader, select_autoescape
from ..specs import Specsable
from ..specs.specs_writer import _ElementInTree, _ElementsIterator
from warnings import warn


STATIC_FOLDER = pathlib.Path(__file__).parent / 'static'
TEMPLATES_FOLDER = pathlib.Path(__file__).parent / 'templates'

jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATES_FOLDER),
    autoescape=select_autoescape()
)


class LinearOpticalSetupWidget(anywidget.AnyWidget):
    _esm = STATIC_FOLDER / 'setup_widget.js'
    _css = STATIC_FOLDER / 'setup_widget.css'

    elements = traitlets.List([]).tag(sync=True)
    settings = traitlets.Dict({
        'open': True,
        'show_all': False,
    }).tag(sync=True)


class LinearOpticalSetupStepwiseForwardWidget(LinearOpticalSetupWidget):
    _esm = STATIC_FOLDER / 'setup_stepwise_forward_widget.js'
    _css = STATIC_FOLDER / 'setup_widget.css'

    wavefront_images = traitlets.List([]).tag(sync=True)


@dataclass(frozen=True, slots=True)
class ElementHTML:
    """Representation of element in the HTML format"""
    element_type: str | None
    html: str


def default_widget_html_method(
    index: int,
    name: str,
    element_type: str | None,
    subelements: list[ElementHTML]
) -> str:
    """This function represents default `_widget_html_` method used for Specsable.

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


def _get_widget_html_method(element: Specsable):
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


def generate_structure_html(*specsable: Specsable) -> str:
    """Generate HTML for a setup structure.

    Returns
    -------
    str
        Rendered HTML
    """
    elements = _ElementsIterator(*specsable, directory='')

    elements_html = _subelements_html(elements.tree)

    return jinja_env.get_template(
        'widget_structure_container.html.jinja'
    ).render(elements_html=elements_html)


def show_structure(*specsable: Specsable):
    """Display a setup structure.
    """
    try:
        from IPython.display import HTML, display

        raw_html = generate_structure_html(*specsable)
        display(HTML(raw_html))

    except ModuleNotFoundError:
        warn("Currently only display via ipython is supported.")
