import torch
import warnings
from .element import Element
from ..simulation_parameters import SimulationParameters
from ..parameters import OptimizableTensor
from ..wavefront import Wavefront, mul
from typing import Callable, Tuple, Literal
from torch.nn.functional import interpolate


def mesh_intersection_indices(
    x1_left: float,
    x1_right: float,
    x2_left: float,
    x2_right: float,
    dx: float,
) -> tuple[slice, slice]:
    mesh1_left_idx = 0
    mesh1_right_idx = round((x1_right - x1_left) / dx)

    mesh2_left_idx = round((x2_left - x1_left) / dx)
    mesh2_right_idx = round((x2_right - x1_left) / dx)

    intersection_start_idx = max(mesh1_left_idx, mesh2_left_idx)
    intersection_end_idx = min(mesh1_right_idx, mesh2_right_idx)

    if intersection_start_idx > intersection_end_idx:
        return slice(0), slice(0)

    mesh1_slice = slice(intersection_start_idx, intersection_end_idx + 1)
    mesh2_slice = slice(
        intersection_start_idx - mesh2_left_idx,
        intersection_end_idx - mesh2_left_idx + 1,
    )
    return mesh1_slice, mesh2_slice


class SpatialLightModulator(Element):
    """A class that described the field after propagating through the
    Spatial Light Modulator with a given phase mask
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        mask: OptimizableTensor,
        height: float,
        width: float,
        center: Tuple[float, float] = (0.0, 0.0),
        mask_function: Callable[[torch.Tensor], torch.Tensor] | None = None,
        mask_norm: float = 2 * torch.pi,
        mode: Literal[
            "nearest", "bilinear", "bicubic", "area", "nearest-exact"
        ] = "nearest",
    ):

        super().__init__(simulation_parameters)

        self.mask = self.process_parameter("mask", mask)
        self.height = height
        self.width = width
        self.center = center
        self.mask_function = mask_function
        self.mask_norm = mask_norm
        self.mode = mode

        x = self.simulation_parameters.x
        y = self.simulation_parameters.y

        x_nodes = x.shape[0]
        y_nodes = y.shape[0]
        # Compute spatial grid spacing
        dx = (x[1] - x[0]).cpu().item() if x_nodes > 1 else 1.0
        dy = (y[1] - y[0]).cpu().item() if y_nodes > 1 else 1.0

        x_intersection_slice, slm_x_intersection_slice = mesh_intersection_indices(
            x[0].cpu().item(),
            x[-1].cpu().item(),
            self.center[0] - self.width / 2,
            self.center[0] + self.width / 2,
            dx,
        )

        y_intersection_slice, slm_y_intersection_slice = mesh_intersection_indices(
            y[0].cpu().item(),
            y[-1].cpu().item(),
            self.center[1] - self.height / 2,
            self.center[1] + self.height / 2,
            dy,
        )

        # TODO: create slices for xy and mask meshgrids, continue rewritung from here

    def _resized_mask(self) -> torch.Tensor:

        _y_indices, _x_indices = torch.where(self._aperture == 1)
        _y_indices, _x_indices = torch.unique(_y_indices), torch.unique(
            _x_indices
        )  # noqa: E501

        left_boundary = _x_indices[
            torch.argmin(
                torch.abs(
                    self._x_linear[_x_indices] - (self.center[0] - self.width / 2)
                )  # noqa: E501
            ).item()
        ]
        right_boundary = _x_indices[
            torch.argmin(
                torch.abs(
                    self._x_linear[_x_indices] - (self.center[0] + self.width / 2)
                )  # noqa: E501
            ).item()
        ]
        top_boundary = _y_indices[
            torch.argmin(
                torch.abs(
                    self._y_linear[_y_indices] - (self.center[1] + self.height / 2)
                )  # noqa: E501
            ).item()
        ]
        bottom_boundary = _y_indices[
            torch.argmin(
                torch.abs(
                    self._y_linear[_y_indices] - (self.center[1] - self.height / 2)
                )  # noqa: E501
            ).item()
        ]

        x_nodes_interpolate = right_boundary - left_boundary + 1
        y_nodes_interpolate = top_boundary - bottom_boundary + 1

        _resized_mask = self.mask.unsqueeze(0).unsqueeze(0)

        # interpolate to dimensions from simulation_parameters
        _resized_mask = interpolate(
            _resized_mask,
            size=(y_nodes_interpolate, x_nodes_interpolate),
            mode=self.mode,
        )

        # delete added dimensions
        resized_mask = _resized_mask.squeeze(0).squeeze(0)

        if resized_mask.size() < self.mask.size():
            warnings.warn(
                f"New mask size {resized_mask.size()} is smaller than the original one {self.mask.size()}! "
            )

        return resized_mask

    @property
    def transmission_function(self) -> torch.Tensor:

        _resized_mask = self._resized_mask()

        indices = (
            slice(self.bottom_boundary, self.top_boundary + 1),
            slice(self.left_boundary, self.right_boundary + 1),
        )

        _phase_mask = _aperture.clone()

        # NON-DIFFERENTIABLE!
        # quantized_mask = 2 * torch.pi * (
        #     self.step_function(
        #         (
        #             (self.number_of_levels * _resized_mask) % (2 * torch.pi)
        #         ) / (2 * torch.pi)
        #     ) + (self.number_of_levels * _resized_mask) // (2 * torch.pi)
        # ) / self.number_of_levels

        # FIXED
        quantized_mask = self.step_function(_resized_mask)

        _phase_mask[indices] = quantized_mask

        transmission_function = torch.exp(1j * _phase_mask)

        return transmission_function

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:

        return mul(
            incident_wavefront,
            self.transmission_function,
            ("y", "x"),
            self.simulation_parameters,
        )

    def reverse(self, transmission_wavefront: Wavefront) -> Wavefront:

        return mul(
            transmission_wavefront,
            torch.conj(self.transmission_function),
            ("y", "x"),
            self.simulation_parameters,
        )
