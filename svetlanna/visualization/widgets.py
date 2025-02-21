import anywidget
import traitlets
import pathlib


STATIC_FOLDER = pathlib.Path(__file__).parent / 'static'


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
