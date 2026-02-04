import torch
import warnings
from .element import Element
from ..simulation_parameters import SimulationParameters
from ..parameters import OptimizableTensor
from ..wavefront import Wavefront
from typing import Callable, Tuple, Literal, TypeVar, Generic
from typing import ParamSpec, Concatenate
from torch.nn.functional import interpolate


def one_step_tanh(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    "function f: [0,1] -> [0,1] with steeper transition controlled by alpha"
    "f(0) = 0, f(1) = 1"
    "f'(0) = f'(1)"
    if alpha == 0:
        return x
    return torch.tanh(alpha * (x - 0.5)) / torch.tanh(alpha / 2) / 2 + 0.5


def one_step_cos(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    "function f: [0,1] -> [0,1] with steeper transition controlled by alpha"
    "f(0) = 0, f(1) = 1"
    "f'(0) = f'(1)"
    return (1 - torch.cos(torch.pi * x ** (alpha + 1))) / 2


_P = ParamSpec("_P")


def QuantizerFromStepFunction(
    N: int,
    max_value: float,
    one_step_function: Callable[
        Concatenate[torch.Tensor, _P],
        torch.Tensor,
    ],
) -> Callable[Concatenate[torch.Tensor, _P], torch.Tensor]:

    def f(x: torch.Tensor, *args: _P.args, **kwargs: _P.kwargs) -> torch.Tensor:
        y = x / max_value * N

        y = torch.remainder(y, N)
        frac = torch.frac(y)
        base = torch.floor(y)

        return max_value * (one_step_function(frac, *args, **kwargs) + base) / N

    return f


def _mesh_intersection_indices(
    x1_left: float,
    x1_right: float,
    x2_left: float,
    x2_right: float,
    dx: float,
) -> tuple[slice, int, slice, int]:
    mesh1_left_idx = 0
    mesh1_right_idx = round((x1_right - x1_left) / dx)
    mesh1_N = mesh1_right_idx - mesh1_left_idx + 1

    mesh2_left_idx = round((x2_left - x1_left) / dx)
    mesh2_right_idx = round((x2_right - x1_left) / dx)
    mesh2_N = mesh2_right_idx - mesh2_left_idx + 1

    intersection_start_idx = max(mesh1_left_idx, mesh2_left_idx)
    intersection_end_idx = min(mesh1_right_idx, mesh2_right_idx)

    if intersection_start_idx > intersection_end_idx:
        return slice(0), mesh1_N, slice(0), mesh2_N

    mesh1_slice = slice(intersection_start_idx, intersection_end_idx + 1)
    mesh2_slice = slice(
        intersection_start_idx - mesh2_left_idx,
        intersection_end_idx - mesh2_left_idx + 1,
    )
    return mesh1_slice, mesh1_N, mesh2_slice, mesh2_N


def identity(phase: torch.Tensor) -> torch.Tensor:
    return phase


_F = TypeVar("_F", bound=Callable[[torch.Tensor], torch.Tensor])


class SpatialLightModulator(Element, Generic[_F]):
    """A class that described the field after propagating through the
    Spatial Light Modulator with a given phase mask
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        mask: OptimizableTensor,
        height: float,
        width: float,
        lut_function: _F = identity,  # type: ignore
        center: Tuple[float, float] = (0.0, 0.0),
        mode: Literal[
            "nearest", "bilinear", "bicubic", "area", "nearest-exact"
        ] = "nearest",
    ):

        super().__init__(simulation_parameters)

        self.mask = self.process_parameter("mask", mask)
        self.height = self.process_parameter("height", height)
        self.width = self.process_parameter("width", width)
        self.center = self.process_parameter("center", center)
        self.mode = self.process_parameter("mode", mode)
        self.lut_function = self.process_parameter("lut_function", lut_function)

        x = self.simulation_parameters.x
        y = self.simulation_parameters.y

        x_nodes = x.shape[0]
        y_nodes = y.shape[0]
        # Compute spatial grid spacing
        dx = (x[1] - x[0]).cpu().item() if x_nodes > 1 else 1.0
        dy = (y[1] - y[0]).cpu().item() if y_nodes > 1 else 1.0

        x_slice, _, slm_x_slice, slm_x_N = _mesh_intersection_indices(
            x[0].cpu().item(),
            x[-1].cpu().item(),
            self.center[0] - self.width / 2,
            self.center[0] + self.width / 2,
            dx,
        )

        y_slice, _, slm_y_slice, slm_y_N = _mesh_intersection_indices(
            y[0].cpu().item(),
            y[-1].cpu().item(),
            self.center[1] - self.height / 2,
            self.center[1] + self.height / 2,
            dy,
        )

        if slm_x_N < self.mask.size(1) or slm_y_N < self.mask.size(0):
            warnings.warn(
                f"SLM mask after interpolation will be of smaller size {(slm_y_N, slm_x_N)} than the original one {self.mask.size()}!"
            )

        x_index = self.simulation_parameters.index("x")
        y_index = self.simulation_parameters.index("y")
        _mesh_slice = [slice(None) for _ in range(max(-x_index, -y_index))]
        _mesh_slice[x_index] = x_slice
        _mesh_slice[y_index] = y_slice
        self._mesh_slice = tuple(_mesh_slice)

        self._slm_slice = (slm_y_slice, slm_x_slice)
        self._slm_size = (slm_y_N, slm_x_N)

        _aperture = torch.zeros((y_nodes, x_nodes))
        _aperture = self.simulation_parameters.cast(_aperture, "y", "x")
        _aperture[self._mesh_slice] = 1
        self._aperture = self.make_buffer("_aperture", _aperture)

    def _phase_mask(self) -> torch.Tensor:

        # interpolate to dimensions from simulation_parameters
        mask = self.mask.unsqueeze(0).unsqueeze(0)
        resized_mask = interpolate(mask, size=self._slm_size, mode=self.mode)
        resized_mask = resized_mask.squeeze(0).squeeze(0)

        resized_mask = resized_mask[self._slm_slice]

        resized_mask = self.lut_function(resized_mask)

        return resized_mask

    @property
    def transmission_function(self) -> torch.Tensor:

        phase = self._phase_mask()

        transmission_function = self._aperture.clone() + 0j
        transmission_function[self._mesh_slice] = transmission_function[
            self._mesh_slice
        ] * torch.exp(1j * phase)

        return transmission_function

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:

        wavefront = incident_wavefront * self._aperture + 0j

        phase = self._phase_mask()
        wavefront[self._mesh_slice] = wavefront[self._mesh_slice] * torch.exp(
            1j * phase
        )

        return wavefront

    def reverse(self, transmission_wavefront: Wavefront) -> Wavefront:

        wavefront = transmission_wavefront * self._aperture + 0j

        phase = self._phase_mask()
        wavefront[self._mesh_slice] = wavefront[self._mesh_slice] * torch.exp(
            -1j * phase
        )

        return wavefront
