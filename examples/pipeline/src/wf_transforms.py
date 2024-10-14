import torch
from torch import nn

from svetlanna.wavefront import Wavefront
from svetlanna.simulation_parameters import SimulationParameters


class ToWavefront(nn.Module):
    """
    Transformation of a Tensor to a Wavefront. Three types of transform:
        (1) modulation_type='amp'
            tensor values transforms to amplitude, phase = 0
        (2) modulation_type='phase'
            tensor values transforms to phases (from 0 to 2pi - eps), amp = const
        (3) modulation_type='amp&phase' (any other str)
            tensor values transforms to amplitude and phase simultaneously
    """
    def __init__(self, modulation_type=None):
        """
        Method that returns the image obtained from the incident field by a detector
        (in the simplest case the image on a detector is an intensities image)
        ...

        Parameters
        ----------
        modulation_type : str
            A type of modulation to obtain a wavefront.
        """
        super().__init__()
        self.eps = 1e-6  # necessary for phase modulation
        self.modulation_type = modulation_type

    def forward(self, img_tensor: torch.Tensor) -> Wavefront:
        """
        Function that transforms Tensor to Wavefront.
        ...

        Parameters
        ----------
        img_tensor : torch.Tensor
            A Tensor to be transformed to a Wavefront.

        Returns
        -------
        img_wavefront : Wavefront
            A resulted Wavefront obtained via one of modulation types (self.modulation_type).
        """
        # creation of a wavefront based on an image
        max_val = img_tensor.max()
        min_val = img_tensor.min()
        normalized_tensor = (img_tensor - min_val) / (max_val - min_val)  # values from 0 to 1

        if self.modulation_type == 'amp':  # amplitude modulation
            amplitudes = normalized_tensor
            phases = torch.zeros(size=img_tensor.size())
        else:
            # image -> phases from 0 to 2pi - eps
            phases = normalized_tensor * (2 * torch.pi - self.eps)
            if self.modulation_type == 'phase':  # phase modulation
                # TODO: what is with an amplitude? Can it be zero?
                amplitudes = torch.ones(size=img_tensor.size())  # constant amplitude
            else:  # phase AND amplitude modulation 'amp&phase'
                amplitudes = normalized_tensor

        # construct wavefront
        img_wavefront = Wavefront(amplitudes * torch.exp(1j * phases))

        return img_wavefront
