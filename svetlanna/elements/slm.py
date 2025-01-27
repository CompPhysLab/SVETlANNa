import torch

from .element import Element
from ..simulation_parameters import SimulationParameters
from ..parameters import OptimizableFloat
from ..wavefront import Wavefront, mul
from typing import Callable, Tuple, Literal
from torch.nn.functional import interpolate, relu


class SpatialLightModulator(Element):
    """A class that described the field after propagating through the
    Spatial Light Modulator with a given phase mask

    Parameters
    ----------
    Element : _type_
        _description_
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        mask: torch.Tensor,
        height: OptimizableFloat,
        width: OptimizableFloat,
        location: Tuple = (0., 0.),
        number_of_levels: int = 256,
        step_function: Callable[[torch.Tensor], torch.Tensor] = relu,
        mode: Literal[
            'nearest',
            'bilinear',
            'bicubic',
            'area',
            'nearest-exact'
        ] = 'nearest'
    ):

        super().__init__(simulation_parameters)

        self.mask = mask
        self.height = height
        self.width = width
        self.step_function = step_function
        self.x, self.y = location
        self.mode = mode
        self.number_of_levels = number_of_levels

        self.height_resolution, self.width_resolution = self.mask.shape

        self._device = self.simulation_parameters.device

        self._w_index = self.simulation_parameters.axes.index('W')
        self._h_index = self.simulation_parameters.axes.index('H')

        self._x_linear = self.simulation_parameters.axes.W
        self._y_linear = self.simulation_parameters.axes.H

        self._x_grid = self._x_linear[None, :]
        self._y_grid = self._y_linear[:, None]

    @property
    def get_aperture(self) -> torch.Tensor:

        self.aperture = ((torch.abs(
            self._x_grid - self.x) <= self.width/2) * (torch.abs(
                self._y_grid - self.y) <= self.height/2)).to(
                    dtype=torch.get_default_dtype(),
                    device=self._device
                )
        return self.aperture

    @property
    def resize_mask(self) -> torch.Tensor:

        _y_indices, _x_indices = torch.where(self.aperture == 1)
        _y_indices, _x_indices = torch.unique(_y_indices), torch.unique(_x_indices)  # noqa: E501

        self.left_boundary = _x_indices[
            torch.argmin(
                torch.abs(self._x_linear[_x_indices] - (self.x - self.width / 2))   # noqa: E501
            ).item()
        ]
        self.right_boundary = _x_indices[
            torch.argmin(
                torch.abs(self._x_linear[_x_indices] - (self.x + self.width / 2))   # noqa: E501
            ).item()
        ]
        self.top_boundary = _y_indices[
            torch.argmin(
                torch.abs(self._y_linear[_y_indices] - (self.y + self.height / 2))  # noqa: E501
            ).item()
        ]
        self.bottom_boundary = _y_indices[
            torch.argmin(
                torch.abs(self._y_linear[_y_indices] - (self.y - self.height / 2))  # noqa: E501
            ).item()
        ]

        x_nodes_interpolate = self.right_boundary - self.left_boundary + 1
        y_nodes_interpolate = self.top_boundary - self.bottom_boundary + 1

        _resized_mask = self.mask.unsqueeze(0).unsqueeze(0)

        # interpolate to dimensions from simulation_parameters
        _resized_mask = interpolate(
            _resized_mask,
            size=(y_nodes_interpolate, x_nodes_interpolate),
            mode=self.mode
        )

        # delete added dimensions
        self.resized_mask = _resized_mask.squeeze(0).squeeze(0)
        return self.resized_mask

    @property
    def get_transmission_function(self) -> torch.Tensor:

        _aperture = self.get_aperture
        _resized_mask = self.resize_mask

        indices = (
            slice(self.bottom_boundary, self.top_boundary + 1),
            slice(self.left_boundary, self.right_boundary + 1)
        )

        _phase_mask = _aperture.clone()

        quantized_mask = 2 * torch.pi * (
            self.step_function(
                (
                    (self.number_of_levels * _resized_mask) % (2 * torch.pi)
                ) / (2 * torch.pi)
            ) + (self.number_of_levels * _resized_mask) // (2 * torch.pi)
        ) / self.number_of_levels

        _phase_mask[indices] = quantized_mask

        self.transmission_function = torch.exp(
            1j * _phase_mask
        )

        return self.transmission_function

    def forward(self, input_field: Wavefront) -> Wavefront:

        return mul(
                    input_field,
                    self.get_transmission_function,
                    ('H', 'W'),
                    self.simulation_parameters
                )

    def reverse(self, transmission_field: Wavefront) -> Wavefront:

        return mul(
                    transmission_field,
                    torch.conj(self.get_transmission_function),
                    ('H', 'W'),
                    self.simulation_parameters
                )
