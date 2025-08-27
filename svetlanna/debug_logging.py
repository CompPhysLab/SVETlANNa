from torch.nn.modules.module import register_module_forward_hook
from torch.nn.modules.module import register_module_buffer_registration_hook
from torch.nn.modules.module import register_module_parameter_registration_hook
from torch.nn.modules.module import register_module_module_registration_hook
from torch.utils.hooks import RemovableHandle
from .elements import Element
from torch import Tensor
import logging
from typing import Any, Literal
from functools import partial


logger = logging.getLogger(__name__)

__handles: None | tuple[RemovableHandle, ...] = None
__logging_type: Literal['logging', 'print'] = 'print'


def agr_short_description(arg: Any) -> str:
    """Create short description string based on arg type

    Parameters
    ----------
    arg : Any
        argument value

    Returns
    -------
    str
        description
    """
    if isinstance(arg, Tensor):
        return f'{type(arg)} shape={arg.shape}, dtype={arg.dtype}, device={arg.device}'
    else:
        return f'{type(arg)}'


def log_message(message: str):
    if __logging_type == 'logging':
        logger.debug(message)
    elif __logging_type == 'print':
        print(message)


def forward_logging_hook(module, input, output) -> None:
    """Global debug forward hook for all elements"""
    if not isinstance(module, Element):
        return

    args_info = ''

    # cast inputs and outputs to tuples
    input = (input,) if not isinstance(input, tuple) else input
    output = (output,) if not isinstance(output, tuple) else output

    for i, _input in enumerate(input):
        args_info += f'\n   input {i}: {agr_short_description(_input)}'

    for i, _output in enumerate(output):
        args_info += f'\n   output {i}: {agr_short_description(_output)}'

    log_message(
        f'The forward method of {module._get_name()} was computed{args_info}'
    )


def register_logging_hook(
    module, name, value,
    type: Literal['Parameter', 'Buffer', 'Module']
) -> None:
    if not isinstance(module, Element):
        return

    value_info = f'\n   {agr_short_description(value)}'

    log_message(
        f'{type} of {module._get_name()} was registered with name {name}:{value_info}'
    )


def set_debug_logging(
    mode: bool,
    type: Literal['logging', 'print'] = 'print'
):
    """Enables and disables debug logging.
    If type is `'print'`, then messages are printed using `print`,
    if type is `'logging'` the messages are written in the logger
    named `svetlanna.logging`.

    Parameters
    ----------
    mode : bool
        flag whether to enable debug logging
    type: Literal['logging', 'print']
        type of logging messages output
    """
    global __handles
    global __logging_type

    if type not in ('logging', 'print'):
        raise ValueError(
            f"Logging type should be 'logging' or 'print, not {type}"
        )
    __logging_type = type

    if mode:
        if __handles is None:
            __handles = (
                register_module_forward_hook(
                    forward_logging_hook
                ),
                register_module_parameter_registration_hook(
                    partial(register_logging_hook, type='Parameter')
                ),
                register_module_buffer_registration_hook(
                    partial(register_logging_hook, type='Buffer')
                ),
                register_module_module_registration_hook(
                    partial(register_logging_hook, type='Module')
                )
            )
    else:
        if __handles is not None:
            for handle in __handles:
                handle.remove()
            __handles = None
