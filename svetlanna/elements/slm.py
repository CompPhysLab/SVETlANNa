import torch

from .element import Element
from ..simulation_parameters import SimulationParameters
from ..parameters import OptimizableTensor, OptimizableFloat
from ..wavefront import Wavefront, mul
from typing import Callable
from torch.nn.functional import interpolate


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
        mask: torch.Tensor,
        height: OptimizableFloat,
        height_resolution: int,
        width: OptimizableFloat,
        width_resolution: int,
        step_function: Callable
    ):
        self.mask = mask
        self.height = height
        self.height_resolution = height_resolution
        self.width = width
        self.width_resolution = width_resolution
        self.step_function = step_function

    def to_simulation_parameters(
        self,
        simulation_parameters: SimulationParameters
    ) -> torch.Tensor:

        _x_linear = simulation_parameters.axes.W
        _y_linear = simulation_parameters.axes.H

        x_nodes = _x_linear.shape[0]
        y_nodes = _y_linear.shape[0]

        upscaled_mask = self.mask.unsqueeze(0).unsqueeze(0)

        # interpolate to dimensions from simulation_parameters
        upscaled_mask = interpolate(
            upscaled_mask,
            size=(y_nodes, x_nodes),
            mode='nearest-exact'
        )

        # delete added dimensions
        self.upscaled_mask = upscaled_mask.squeeze(0).squeeze(0)
        return self.upscaled_mask

    def forward(self):
        return None


# TODO: check docstrings
# class SpatialLightModulator(Element):
#     """A class that described the field after propagating through the
#     Spatial Light Modulator with a given phase mask

#     Parameters
#     ----------
#     Element : _type_
#         _description_
#     """

#     def __init__(
#         self,
#         simulation_parameters: SimulationParameters,
#         mask: OptimizableTensor,
#         number_of_levels: int = 256
#     ):
#         """Constructor method

#         Parameters
#         ----------
#         simulation_parameters : SimulationParameters
#             Class exemplar, that describes optical system
#         mask : torch.Tensor
#             Phase mask in grey format for the SLM, every element must be int
#         number_of_levels : int, optional
#             Number of phase quantization levels for the SLM, by default 256
#         """

#         super().__init__(simulation_parameters)

#         self.mask = mask
#         self.number_of_levels = number_of_levels

#         self.transmission_function = torch.exp(
#             1j * 2 * torch.pi / self.number_of_levels * self.mask
#         )

#     def get_transmission_function(self) -> torch.Tensor:
#         """Method which returns the transmission function of
#         the SLM

#         Returns
#         -------
#         torch.Tensor
#             transmission function of the SLM
#         """

#         return self.transmission_function

#     def forward(self, input_field: Wavefront) -> Wavefront:
#         """Method that calculates the field after propagating through the SLM

#         Parameters
#         ----------
#         input_field : Wavefront
#             Field incident on the SLM

#         Returns
#         -------
#         Wavefront
#             The field after propagating through the SLM
#         """

#         return mul(
#             input_field,
#             self.transmission_function,
#             ('H', 'W'),
#             self.simulation_parameters
#         )

#     def reverse(self, transmission_field: torch.Tensor) -> Wavefront:
#         """Method that calculates the field after passing the SLM in back
#         propagation

#         Parameters
#         ----------
#         transmission_field : torch.Tensor
#             Field incident on the SLM in back propagation
#             (transmitted field in forward propagation)

#         Returns
#         -------
#         torch.tensor
#             Field transmitted on the SLM in back propagation
#             (incident field in forward propagation)
#         """

#         return mul(
#             transmission_field,
#             torch.conj(self.transmission_function),
#             ('H', 'W'),
#             self.simulation_parameters
#         )
