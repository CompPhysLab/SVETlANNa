import torch
from torch import nn
from svetlanna import Wavefront, SimulationParameters


class DiffractiveRNN(nn.Module):
    """
    A simple recurrent diffractive network of an architecture proposed in the article:
        https://www.nature.com/articles/s41566-021-00796-w
    """

    def __init__(
            self,
            sim_params: SimulationParameters,
            sequence_len: int,
            fusing_coeff: float,
            read_in_layer: nn.Sequential,
            memory_layer: nn.Sequential,
            hidden_forward_layer: nn.Sequential,
            read_out_layer: nn.Sequential,
            detector_layer: nn.Sequential,
            device: str | torch.device = torch.get_default_device(),
    ):
        """
        sim_params: SimulationParameters
            Simulation parameters for the task.
        sequence_len: int
            A size (number of frames) of sequences (of Wavefronts) for prediction.
        fusing_coeff: float
            A coefficient in a function for a hidden state (lambda in methods of the article).
        read_in_layer, memory_layer: nn.Sequential
            Systems of elements for a D-RNN parts (see the article).
        hidden_forward_layer: nn.Sequential
            System for a hidden state after each frame input.
            Comment:
                mix_i = (1 - fusing_coeff) * read_in_layer(input_i) + fusing_coeff * hidden_i
                hidden_i = hidden_forward_layer(mix_i)
        read_out_layer, detector_layer: nn.Sequential
            System of elements for a D-RNN output. `detector_layer` ends with a Detector!
        device: torch.device
            Specified device.
        """
        super().__init__()

        self.sequence_len = sequence_len
        self.fusing_coeff = fusing_coeff
        self.sim_params = sim_params

        self.h, self.w = self.sim_params.axes_size(
            axs=('H', 'W')
        )  # height and width for a wavefronts

        self.__device = torch.device(device)

        # -------------------------------------------------- ALL LAYERS
        self.read_in_layer = read_in_layer.to(self.__device)
        self.memory_layer = memory_layer.to(self.__device)
        self.hidden_forward_layer = hidden_forward_layer.to(self.__device)
        self.read_out_layer = read_out_layer.to(self.__device)
        # ----------------------------------------- LAYER WITH DETECTOR
        self.detector_layer = detector_layer.to(self.__device)
        # -------------------------------------------------------------

    def forward(self, subsequence_wf: Wavefront):
        """
        Parameters
        ----------
        subsequence_wf: Wavefront('batch_size', 'sequence_len', 'H', 'W')
            Wavefronts for a sequence.
            Comment: works for a single wavelength in SimulationParameters!
        """
        # Comment: see all designations (from comments here) in the article!
        if len(subsequence_wf.shape) > 3:  # if a batch is an input
            batch_flag = True
            bs = subsequence_wf.shape[0]
            h_prev = Wavefront(
                torch.zeros(
                    size=(bs, self.h, self.w)
                )
            ).to(self.__device)  # h_{t - 1} - reset hidden for the first input
        else:
            batch_flag = False
            h_prev = Wavefront(
                torch.zeros(
                    size=(self.h, self.w)
                )
            ).to(self.__device)  # h_{t - 1} - reset hidden for the first input

        for frame_ind in range(self.sequence_len):
            if batch_flag:
                x_t = subsequence_wf[:, frame_ind, :, :]
            else:  # not a batch
                x_t = subsequence_wf[frame_ind, :, :]

            i_t = self.read_in_layer(x_t)  # f_2(x_t)

            m_t = self.memory_layer(h_prev)  # f_1(h_{t - 1})

            h_prev = self.fusing_coeff * m_t + (1 - self.fusing_coeff) * i_t
            h_prev = self.hidden_forward_layer(h_prev)

        out = self.detector_layer(self.read_out_layer(h_prev))

        return out

    def to(self, device: str | torch.device | int) -> 'DiffractiveRNN':
        if self.__device == torch.device(device):
            return self

        return DiffractiveRNN(
            sim_params=self.sim_params,
            sequence_len=self.sequence_len, fusing_coeff=self.fusing_coeff,
            read_in_layer=self.read_in_layer, memory_layer=self.memory_layer,
            hidden_forward_layer=self.hidden_forward_layer,
            read_out_layer=self.read_out_layer, detector_layer=self.detector_layer,
            device=device,
        )

    @property
    def device(self) -> str | torch.device | int:
        return self.__device
