import warnings

import torch
from torch import nn
import numpy as np
from svetlanna.wavefront import Wavefront
from svetlanna import SimulationParameters


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
        self.eps = 1e-6  # necessary for phase modulation
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
        if img_tensor.size()[0] == 1:  # only one channel
            # squeeze 0th channel dimension of image tensor
            normalized_tensor = torch.squeeze(img_tensor, 0)  # values from 0 to 1, shape=[H, W]
        else:  # more than 1 color channels
            normalized_tensor = img_tensor  # values from 0 to 1, shape=[C, H, W]
            # TODO: check that in simulation parameters we have the same number of wavelengths?

        if self.modulation_type == 'amp':  # amplitude modulation
            amplitudes = normalized_tensor
            phases = torch.zeros(size=normalized_tensor.size())
        else:
            # image -> phases from -pi + eps to pi - eps
            normalized_tensor_fix = normalized_tensor
            normalized_tensor_fix[normalized_tensor_fix == 1.] -= self.eps  # maximal values - eps
            normalized_tensor_fix[normalized_tensor_fix == 0.] += self.eps  # 0 + eps

            # [0, 1] --> [-pi + eps, pi - eps]
            phases = normalized_tensor_fix * 2 * torch.pi - torch.pi

            if self.modulation_type == 'phase':  # phase modulation
                # TODO: What is with an amplitude?
                amplitudes = torch.ones(size=normalized_tensor.size())  # constant amplitude
            else:  # phase AND amplitude modulation 'amp&phase'
                amplitudes = normalized_tensor

        # construct wavefront
        img_wavefront = Wavefront(amplitudes * torch.exp(1j * phases))

        return img_wavefront


class GaussModulation(nn.Module):
    """
    Multiplies an amplitude of a Wavefront on a gaussian.
    """
    def __init__(
            self,
            sim_params: SimulationParameters,
            fwhm_x, fwhm_y,
            peak_x=0., peak_y=0.
    ):
        """
        Parameters
        ----------
        fwhm_x, fwhm_y : float
            The full width at half maximum along axes (SI units).
        peak_x, peak_y : float
            Peak position in a plane (SI units).
        """
        super().__init__()
        self.sim_params = sim_params
        self.sigma_x = fwhm_x / 2 / np.sqrt(2 * np.log(2))  # sigmas
        self.sigma_y = fwhm_y / 2 / np.sqrt(2 * np.log(2))
        self.peak_x = peak_x  # peak coordinates
        self.peak_y = peak_y
        # generate a gaussian
        self.gauss = self.get_gauss()

    def get_gauss(self):
        """
        Generates a gaussian according to simulation parameters!
        ...

        Returns
        -------
        gauss_2d : torch.Tensor
            A gaussian distribution in a 2D plane.
        """
        x_grid, y_grid = self.sim_params.meshgrid(x_axis='W', y_axis='H')

        gauss_2d = 1 * torch.exp(
            -1 * (
                    (x_grid - self.peak_x) ** 2 / 2 / self.sigma_x ** 2 +
                    (y_grid - self.peak_y) ** 2 / 2 / self.sigma_y ** 2
            )
        )
        return gauss_2d

    def forward(self, wf: Wavefront) -> Wavefront:
        """
        Multiplies an input wavefront on a gauss.
        ...

        Parameters
        ----------
        wf : Wavefront
            An input wavefront of a shape corresponding to simulation parameters.

        Returns
        -------
        wf_gauss : Wavefront
            A gaussian distribution in a 2D plane.
        """
        sim_nodes_shape = self.sim_params.axes_size(axs=('H', 'W'))  # [H, W]

        if not wf.size()[-2:] == sim_nodes_shape:
            warnings.warn(
                message='A shape of an input Wavefront does not match with SimulationParameters! Gauss was not applied!'
            )
            wf_gauss = wf
        else:
            wf_gauss = wf * self.gauss

        return wf_gauss
