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


logger = logging.getLogger("svetlanna.logging")

__handles: None | tuple[RemovableHandle, ...] = None
__logging_type: Literal["logging", "print"] = "print"


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
        return f"{type(arg)} shape={arg.shape}, dtype={arg.dtype}, device={arg.device}"
    else:
        return f"{type(arg)}"


def log_message(message: str):
    if __logging_type == "logging":
        logger.debug(message)
    elif __logging_type == "print":
        print(message)


def forward_logging_hook(module, input, output) -> None:
    """Global debug forward hook for all elements"""
    if not isinstance(module, Element):
        return

    args_info = ""

    # cast inputs and outputs to tuples
    input = (input,) if not isinstance(input, tuple) else input
    output = (output,) if not isinstance(output, tuple) else output

    for i, _input in enumerate(input):
        args_info += f"\n   input {i}: {agr_short_description(_input)}"

    for i, _output in enumerate(output):
        args_info += f"\n   output {i}: {agr_short_description(_output)}"

    log_message(f"The forward method of {module._get_name()} was computed{args_info}")


def register_logging_hook(
    module, name, value, type: Literal["Parameter", "Buffer", "Module"]
) -> None:
    if not isinstance(module, Element):
        return

    value_info = f"\n   {agr_short_description(value)}"

    log_message(
        f"{type} of {module._get_name()} was registered with name {name}:{value_info}"
    )


def set_debug_logging(mode: bool, type: Literal["logging", "print"] = "print"):
    """Enable or disable debug logging for elements.

    Logs information about element registration (parameters, buffers, submodules)
    and forward pass execution.
    This helps debug and trace data flow through the optical setup.

    Parameters
    ----------
    mode : bool
        Whether to enable debug logging.
    type : Literal['logging', 'print'], optional
        Output method: `'print'` uses `print()`, `'logging'` writes to the
        `svetlanna.logging` logger at DEBUG level, by default `'print'`.

    Raises
    ------
    ValueError
        If `type` is not `'logging'` or `'print'`.

    Examples
    --------
    ```python
    import svetlanna as sv
    import torch
    from svetlanna.logging import set_debug_logging

    set_debug_logging(True)

    sim_params = sv.SimulationParameters(...)

    diffractive_layer = sv.elements.DiffractiveLayer(
        simulation_parameters=sim_params,
        mask=torch.rand(Ny, Nx),
    )
    input_wavefront = sv.Wavefront.plane_wave(sim_params)
    diffractive_layer(input_wavefront)
    ```

    Output:
    ```linenums="0"
    Buffer of DiffractiveLayer was registered with name mask:
       <class 'torch.Tensor'> shape=torch.Size([512, 512]), dtype=torch.float32, device=cpu
    The forward method of DiffractiveLayer was computed
       input 0: <class 'svetlanna.wavefront.Wavefront'> shape=torch.Size([512, 512]), dtype=torch.complex64, device=cpu
       output 0: <class 'svetlanna.wavefront.Wavefront'> shape=torch.Size([512, 512]), dtype=torch.complex64, device=cpu
    ```
    """
    global __handles
    global __logging_type

    if type not in ("logging", "print"):
        raise ValueError(f"Logging type should be 'logging' or 'print, not {type}")
    __logging_type = type

    if mode:
        if __handles is None:
            __handles = (
                register_module_forward_hook(forward_logging_hook),
                register_module_parameter_registration_hook(
                    partial(register_logging_hook, type="Parameter")
                ),
                register_module_buffer_registration_hook(
                    partial(register_logging_hook, type="Buffer")
                ),
                register_module_module_registration_hook(
                    partial(register_logging_hook, type="Module")
                ),
            )
    else:
        if __handles is not None:
            for handle in __handles:
                handle.remove()
            __handles = None
