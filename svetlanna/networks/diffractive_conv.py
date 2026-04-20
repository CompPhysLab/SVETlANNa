from typing import TYPE_CHECKING, Literal, Iterable
import torch
from torch import nn
from svetlanna import Wavefront, SimulationParameters
from svetlanna import elements
from svetlanna import LinearOpticalSetup
from svetlanna.specs import ParameterSpecs, SubelementSpecs
from svetlanna.parameters import OptimizableFloat, OptimizableTensor
from svetlanna.visualization import ElementHTML, jinja_env


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
            A focal length for [ThinLens][svetlanna.elements.ThinLens]'s in a 4f system.
        conv_diffractive_mask: OptimizableTensor
            An initial mask for a [DiffractiveLayer][svetlanna.elements.DiffractiveLayer] placed between two lenses in the system.
        conv_mask_norm: float
            A normalization factor for the convolutional mask.
        fs_method: Literal['fresnel', 'AS']
            A method for FreeSpace's in the system.

        Examples
        --------
        ```python
        import svetlanna as sv
        from svetlanna.visualization import show_structure

        sim_params = ...

        conv_layer_4f = ConvLayer4F(
            simulation_parameters=sim_params,
            focal_length=0.1,
            conv_diffractive_mask=torch.rand(sim_params.axis_sizes(("y", "x"))),
        )

        show_structure(conv_layer_4f)
        ```
        Output (in IPython environment):
        <iframe
        src="show_structure_ConvLayer4F.html"
        style="width:100%; height:250px; border: 0; color-scheme: inherit;" allowtransparency="true"></iframe>
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

        # TODO: should the free spaces be singletons (one instance instead of 4)? Does the same apply to lenses?
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

    @staticmethod
    def _widget_html_(
        index: int, name: str, element_type: str | None, subelements: list[ElementHTML]
    ) -> str:
        return jinja_env.get_template("widget_convlayer4f.html.jinja").render(
            index=index, name=name, subelements=subelements
        )

    if TYPE_CHECKING:

        def __call__(self, input_wavefront: Wavefront) -> Wavefront: ...


class ConvDiffNetwork4F(nn.Module):
    """
    A simple convolutional network with a 4f system as an optical convolutional layer.
    It consists of a [ConvLayer4F][svetlanna.networks.ConvLayer4F] and a [LinearOpticalSetup][svetlanna.LinearOpticalSetup] after it.
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
            A focal length for [ThinLens][svetlanna.elements.ThinLens]'s in a 4f system.
        conv_diffractive_mask: OptimizableTensor
            An initial mask for a [DiffractiveLayer][svetlanna.elements.DiffractiveLayer] placed between two lenses in the system.
        conv_mask_norm: float
            A normalization factor for the convolutional mask.
        fs_method: Literal['fresnel', 'AS']
            A method for FreeSpace's in the system.

        Examples
        --------
        ```python
        import svetlanna as sv
        from svetlanna.visualization import show_structure

        sim_params = ...

        conv_diff_network_4f = ConvDiffNetwork4F(
            simulation_parameters=sim_params,
            network_elements=(
                sv.elements.FreeSpace(
                    simulation_parameters=sim_params, distance=0.1, method="AS"
                ),
                sv.elements.ThinLens(simulation_parameters=sim_params, focal_length=0.1),
                sv.elements.FreeSpace(
                    simulation_parameters=sim_params, distance=0.1, method="AS"
                ),
            ),
            focal_length=0.1,
            conv_diffractive_mask=torch.rand(sim_params.axis_sizes(("y", "x"))),
        )

        show_structure(conv_diff_network_4f)
        ```
        Output (in IPython environment):
        <iframe
        src="show_structure_ConvDiffNetwork4F.html"
        style="width:100%; height:25rem; border: 0; color-scheme: inherit;" allowtransparency="true"></iframe>
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
