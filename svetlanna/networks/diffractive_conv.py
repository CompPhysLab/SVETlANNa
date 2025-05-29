from typing import Literal, Iterable

import torch
from torch import nn
from svetlanna import Wavefront, SimulationParameters, ConstrainedParameter
from svetlanna import elements

# for visualisation:
from svetlanna import LinearOpticalSetup
from svetlanna.specs import ParameterSpecs, SubelementSpecs


class ConvLayer4F(nn.Module):
    """
    Diffractive convolutional layer based on a 4f system.
    """

    # TODO: Add a custom aperture (defined by a mask) before a DiffractiveLayer?

    def __init__(
        self,
        sim_params: SimulationParameters,
        focal_length: float,
        conv_diffractive_mask: torch.Tensor,
        learnable_mask: bool = False,
        max_phase: float = 2 * torch.pi,
        fs_method: Literal["fresnel", "AS"] = "AS",
    ):
        """
        Parameters
        ----------
        sim_params: SimulationParameters
            Simulation parameters for the task.
        focal_length: float
            A focal length for ThinLense's in a 4f system.
        conv_diffractive_mask: torch.Tensor
            An initial mask for a DiffractiveLayer placed between two lenses in the system.
        learnable_mask: bool
            If True – a mask for a DiffractiveLayer in the 4f system will be learnable.
            Else – not learnable (const).
        max_phase: float
            A maximal phase for a Diffractive layer in the system.
        fs_method: Literal['fresnel', 'AS']
            A method for FreeSpace's in the system.
        """
        super().__init__()

        self.sim_params = sim_params
        # 4f-system
        self.focal_length = focal_length
        # for DiffractiveLayer
        self.conv_diffractive_mask = conv_diffractive_mask
        self.learnable_mask = learnable_mask
        self.max_phase = max_phase
        # for FreeSpace
        self.fs_method = fs_method

        # compose a 4f system
        self.conv_layer_4f = self.get_conv_layer_4f()

    def get_free_space(self):
        """
        Returns a FreeSpace of a focal length for a 4f system.
        """
        return elements.FreeSpace(
            simulation_parameters=self.sim_params,
            distance=self.focal_length,  # distance is not learnable!
            method=self.fs_method,
        )

    def get_thin_lens(self):
        """
        Returns a ThinLens with a pre-defined focal length.
        """
        return elements.ThinLens(
            simulation_parameters=self.sim_params,
            focal_length=self.focal_length,
        )

    def get_diffractive_layer(self):
        """
        Returns a DiffractiveLayer according to pre-defined settings from init.
        It can be trainable or not according to `self.learnable_mask` flag.
        """
        if self.learnable_mask:  # if a DiffractiveLayer mask must be learnable
            diff_layer = elements.DiffractiveLayer(
                simulation_parameters=self.sim_params,
                mask=ConstrainedParameter(
                    self.conv_diffractive_mask, min_value=0, max_value=self.max_phase
                ),
            )
        else:
            diff_layer = elements.DiffractiveLayer(
                simulation_parameters=self.sim_params,
                mask=self.conv_diffractive_mask,  # mask is not changing during the training!
            )

        return diff_layer

    def get_conv_layer_4f(self):
        """
        Constructs a 4f system using sequential layers.

          This method creates a sequence of optical elements – free spaces and thin lenses –
          arranged in a 4f configuration, including a diffractive layer for convolution
          in the Fourier plane.

          Returns:
            nn.Sequential: A PyTorch Sequential model representing the 4f system.
        """
        system_elements = [
            self.get_free_space(),  # <-- F
            self.get_thin_lens(),  # <-- ThinLens
            self.get_free_space(),  # <-- F
            self.get_diffractive_layer(),  # <-- convolution in a Fourier plane
            self.get_free_space(),  # <-- F
            self.get_thin_lens(),  # <-- ThinLens
            self.get_free_space(),  # <-- F
        ]
        return nn.Sequential(*system_elements)

    def forward(self, input_wf: Wavefront):
        """
        Forward propagation through a convolutional diffractive system based on a 4f system.

        Parameters
        ----------
        input_wf: Wavefront('batch_size', 'H', 'W')
            An input wavefront(s).

        Returns
        -------
        : Wavefront
            A wavefront after a propagation through a system.
        """
        return self.conv_layer_4f(input_wf)


class ConvDiffNetwork4F(nn.Module):
    """
    A simple convolutional network with a 4f system as an optical convolutional layer.
        Comment: -> [4f system (convolution)] -> [some system of elements] ->
    """

    def __init__(
        self,
        sim_params: SimulationParameters,
        network_elements_list: list,
        focal_length: float,
        conv_phase_mask: torch.Tensor,
        learnable_mask: bool = False,
        max_phase: float = 2 * torch.pi,
        fs_method: Literal["fresnel", "AS"] = "AS",
        device: str | torch.device = torch.get_default_device(),
    ):
        """
        Parameters
        ----------
        sim_params : SimulationParameters
            Simulation parameters for the task.
        network_elements_list : list
            List of Elements for a Network after a convolutional layer (4f system).

        focal_length: float
            A focal length for ThinLense's in a 4f system.
        conv_phase_mask: torch.Tensor
            An initial mask for a DiffractiveLayer placed between two lenses in the system.
        learnable_mask: bool
            If True – a mask for a DiffractiveLayer will be learnable. Else – not learnable (const).
        max_phase: float
            A maximal phase for a Diffractive layer in the 4f system.
        fs_method: Literal['fresnel', 'AS']
            A method for FreeSpace's in the system.

        device: str | torch.device
            Device.
        """
        super().__init__()

        self.sim_params = sim_params

        self.__device = torch.device(device)

        # CONVOLUTIONAL LAYER
        self.focal_length = focal_length
        self.conv_phase_mask = conv_phase_mask
        self.learnable_mask = learnable_mask
        self.max_phase = max_phase
        self.fs_method = fs_method

        self.conv_layer = ConvLayer4F(
            sim_params=self.sim_params,
            focal_length=self.focal_length,
            conv_diffractive_mask=self.conv_phase_mask,
            learnable_mask=self.learnable_mask,
            max_phase=self.max_phase,
            fs_method=self.fs_method,
        ).to(self.__device)

        # PART OF THE NETWORK AFTER A 4F CONVOLUTION
        self.network_elements_list = network_elements_list
        self.net_after_conv = nn.Sequential(*self.network_elements_list).to(
            self.__device
        )

    def forward(self, wavefront_in):
        """
        Parameters
        ----------
        wavefront_in: Wavefront('bs', 'H', 'W')
            Input wavefront or a batch of Wavefronts.

        Returns
        -------
        : torch.Tensor | Wavefront
            Output after a Wavefront propagation through a Convolutional layer and the Other Part of all Network.
        """
        # propagate through a convolutional layer
        wavefront_after_convolution = self.conv_layer(wavefront_in)
        # propagate through other layers
        result = self.net_after_conv(wavefront_after_convolution)

        return result

    def to_specs(self) -> Iterable[ParameterSpecs | SubelementSpecs]:
        """
        Returns the specs for the 4F convolution system and linear setup.

            Args:
                None

            Returns:
                Iterable[ParameterSpecs | SubelementSpecs]: An iterable containing
                SubelementSpecs objects representing the 4F Convolution System and
                Linear Setup, each with its corresponding LinearOpticalSetup.
        """
        return (
            SubelementSpecs(
                "4F Convolution System",
                LinearOpticalSetup(list(self.conv_layer.conv_layer_4f)),
            ),
            SubelementSpecs(
                "Linear Setup", LinearOpticalSetup(list(self.net_after_conv))
            ),
        )

    def to(self, device: str | torch.device | int) -> "ConvDiffNetwork4F":
        if self.__device == torch.device(device):
            return self

        return ConvDiffNetwork4F(
            sim_params=self.sim_params,
            network_elements_list=self.network_elements_list,
            focal_length=self.focal_length,
            conv_phase_mask=self.conv_phase_mask,
            learnable_mask=self.learnable_mask,
            max_phase=self.max_phase,
            fs_method=self.fs_method,
            device=device,
        )

    @property
    def device(self) -> str | torch.device | int:
        """
        Returns the device on which the model is located.

          Args:
            None

          Returns:
            The device as a string, torch.device object, or integer.
        """
        return self.__device
