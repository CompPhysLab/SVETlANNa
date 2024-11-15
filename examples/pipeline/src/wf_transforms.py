import torch
from torch import nn

from svetlanna.wavefront import Wavefront


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
        Parameters
        ----------
        modulation_type : str
            A type of modulation to obtain a wavefront.
        """
        super().__init__()
        # since images are usually in the range [0, 255]
        self.eps = 2 * torch.pi / 255  # necessary for phase modulation
        self.modulation_type = modulation_type

    def forward(self, img_tensor: torch.Tensor) -> Wavefront:
        """
        Function that transforms Tensor to Wavefront.
        ...

        Parameters
        ----------
        img_tensor : torch.Tensor
            A Tensor (of shape [C, H, W] in the range [0, 1]) to be transformed to a Wavefront.

        Returns
        -------
        img_wavefront : Wavefront
            A resulted Wavefront obtained via one of modulation types (self.modulation_type).
        """
        # creation of a wavefront based on an image
        normalized_tensor = img_tensor  # values from 0 to 1, shape=[C, H, W]

        if self.modulation_type == 'amp':  # amplitude modulation
            amplitudes = normalized_tensor
            phases = torch.zeros(size=img_tensor.size())
        else:
            # image -> phases from 0 to 2pi - eps
            phases = normalized_tensor * (2 * torch.pi - self.eps)
            if self.modulation_type == 'phase':  # phase modulation
                # TODO: What is with an amplitude? Can it be zero?
                amplitudes = torch.ones(size=img_tensor.size())  # constant amplitude
            else:  # phase AND amplitude modulation 'amp&phase'
                amplitudes = normalized_tensor

        # construct wavefront
        img_wavefront = Wavefront(amplitudes * torch.exp(1j * phases))

        return img_wavefront
