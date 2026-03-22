from typing import TYPE_CHECKING, Literal, Iterable
import torch
from torch import nn
from svetlanna import Wavefront, SimulationParameters
from svetlanna import elements
from svetlanna import LinearOpticalSetup
from svetlanna.specs import ParameterSpecs, SubelementSpecs
from svetlanna.parameters import OptimizableFloat, OptimizableTensor


class ConvLayer4F(nn.Module):
    """
    Diffractive convolutional layer based on a 4f system.
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        focal_length: OptimizableFloat,
        conv_diffractive_mask: OptimizableTensor,
        conv_mask_norm: float = 2 * torch.pi,
        fs_method: Literal["fresnel", "AS"] = "AS",
    ):
        """
        Parameters
        ----------
        simulation_parameters: SimulationParameters
            Simulation parameters.
        focal_length: OptimizableFloat
            A focal length for ThinLense's in a 4f system.
        conv_diffractive_mask: OptimizableTensor
            An initial mask for a DiffractiveLayer placed between two lenses in the system.
        conv_mask_norm: float
            A normalization factor for the convolutional mask.
        fs_method: Literal['fresnel', 'AS']
            A method for FreeSpace's in the system.
        """
        super().__init__()

        self.simulation_parameters = simulation_parameters
        # 4f-system
        self.focal_length = focal_length
        # for DiffractiveLayer
        self.conv_diffractive_mask = conv_diffractive_mask

        self.conv_mask_norm = conv_mask_norm
        # for FreeSpace
        self.fs_method = fs_method

        # compose a 4f system
        self.conv_layer_4f = LinearOpticalSetup(
            [
                elements.FreeSpace(
                    simulation_parameters=self.simulation_parameters,
                    distance=self.focal_length,
                    method=self.fs_method,
                ),
                elements.ThinLens(
                    simulation_parameters=self.simulation_parameters,
                    focal_length=self.focal_length,
                ),
                elements.FreeSpace(
                    simulation_parameters=self.simulation_parameters,
                    distance=self.focal_length,
                    method=self.fs_method,
                ),
                elements.DiffractiveLayer(
                    simulation_parameters=self.simulation_parameters,
                    mask=self.conv_diffractive_mask,
                    mask_norm=self.conv_mask_norm,
                ),
                elements.FreeSpace(
                    simulation_parameters=self.simulation_parameters,
                    distance=self.focal_length,
                    method=self.fs_method,
                ),
                elements.ThinLens(
                    simulation_parameters=self.simulation_parameters,
                    focal_length=self.focal_length,
                ),
                elements.FreeSpace(
                    simulation_parameters=self.simulation_parameters,
                    distance=self.focal_length,
                    method=self.fs_method,
                ),
            ]
        )

    def forward(self, input_wavefront: Wavefront):
        return self.conv_layer_4f(input_wavefront)

    def to_specs(self) -> Iterable[ParameterSpecs | SubelementSpecs]:
        return (
            SubelementSpecs(str(i), element)
            for i, element in enumerate(self.conv_layer_4f.elements)
        )

    if TYPE_CHECKING:

        def __call__(self, input_wavefront: Wavefront) -> Wavefront: ...


class ConvDiffNetwork4F(nn.Module):
    """
    A simple convolutional network with a 4f system as an optical convolutional layer.
        Comment: -> [4f system (convolution)] -> [some system of elements] ->
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        network_elements: Iterable[elements.Element],
        focal_length: OptimizableFloat,
        conv_diffractive_mask: OptimizableTensor,
        conv_mask_norm: float = 2 * torch.pi,
        fs_method: Literal["fresnel", "AS"] = "AS",
    ):
        """
        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Simulation parameters.
        network_elements : Iterable[elements.Element]
            List of Elements for a Network after a convolutional layer (4f system).
        focal_length: OptimizableFloat
            A focal length for ThinLense's in a 4f system.
        conv_diffractive_mask: OptimizableTensor
            An initial mask for a DiffractiveLayer placed between two lenses in the system.
        conv_mask_norm: float
            A normalization factor for the convolutional mask.
        fs_method: Literal['fresnel', 'AS']
            A method for FreeSpace's in the system.
        """
        super().__init__()

        self.simulation_parameters = simulation_parameters

        # CONVOLUTIONAL LAYER
        self.focal_length = focal_length
        self.conv_diffractive_mask = conv_diffractive_mask
        self.conv_mask_norm = conv_mask_norm
        self.fs_method = fs_method

        self.conv_layer = ConvLayer4F(
            simulation_parameters=self.simulation_parameters,
            focal_length=self.focal_length,
            conv_diffractive_mask=self.conv_diffractive_mask,
            conv_mask_norm=self.conv_mask_norm,
            fs_method=self.fs_method,
        )

        # PART OF THE NETWORK AFTER A 4F CONVOLUTION
        self.net_after_conv = LinearOpticalSetup(network_elements)

    def forward(self, input_wavefront: Wavefront) -> Wavefront:

        # propagate through a convolutional layer
        wavefront_after_convolution = self.conv_layer(input_wavefront)
        # propagate through other layers
        result = self.net_after_conv(wavefront_after_convolution)

        return result

    def to_specs(self) -> Iterable[ParameterSpecs | SubelementSpecs]:
        return (
            SubelementSpecs(
                "4F Convolution System",
                self.conv_layer,
            ),
            SubelementSpecs(
                "Linear Setup",
                self.net_after_conv,
            ),
        )

    if TYPE_CHECKING:

        def __call__(self, input_wavefront: Wavefront) -> Wavefront: ...
