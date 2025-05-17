from typing import Iterable, Literal
import torch
from torch import nn
from svetlanna import Wavefront, SimulationParameters
from svetlanna.elements import Element

# for visualisation:
from svetlanna import LinearOpticalSetup
from svetlanna.specs import ParameterSpecs, SubelementSpecs


class LinearAutoencoder(nn.Module):
    """
    A simple autoencoder network consisting of consistent encoder and decoder
    for a simultaneous training.
        .encode() - forward propagation through encoder elements
        .decode() - forward propagation through decoder elements

    Comment: Works only for a single wavelength, input wavefront ['h', 'w']
    """

    def __init__(
        self,
        sim_params: SimulationParameters,
        encoder_elements_list: list[Element] | Iterable[Element],
        decoder_elements_list: list[Element] | Iterable[Element],
        to_return: Literal["wf", "amps"] = "wf",
        device: str | torch.device = torch.get_default_device(),
    ):
        """
        Parameters
        ----------
        sim_params : SimulationParameters
            Simulation parameters for the task.
        encoder_elements_list : list[Element] | Iterable[Element]
            List of elements to compose an encoder.
        decoder_elements_list : list[Element] | Iterable[Element]
            List of elements to compose a decoder.
        to_return : Literal['wf', 'amps']
            Specifies what to return in .encode() and .decode().
            (1) 'wf' – just a wavefront as it is
            (2) 'amps' – a wavefront without phases
        device : torch.device
            Device.
        """
        super().__init__()

        self.sim_params = sim_params
        self.h, self.w = self.sim_params.axes_size(
            axs=("H", "W")
        )  # height and width for a Wavefronts

        self.__device = torch.device(device)
        self.to_return = to_return

        # ENCODER
        self.encoder_elements = encoder_elements_list
        self.encoder = nn.Sequential(*encoder_elements_list).to(self.__device)
        # DECODER
        self.decoder_elements = decoder_elements_list
        self.decoder = nn.Sequential(*decoder_elements_list).to(self.__device)

    def encode(self, wavefront_in):
        """
        Propagation through the encoder part – encode an image wavefront (input).

        Returns
        -------
        wavefront_encoded : Wavefront
            An encoded input wavefront.
        """
        if self.to_return == "wf":
            return self.encoder(wavefront_in)
        if self.to_return == "amps":
            return self.encoder(wavefront_in).abs() + 0j

    def decode(self, wavefront_encoded):
        """
        Propagation through the decoder part – decode an encoded image.

        Returns
        -------
        wavefront_decoded : Wavefront
            A decoded wavefront.
        """
        if self.to_return == "wf":
            return self.decoder(wavefront_encoded)
        if self.to_return == "amps":
            return self.decoder(wavefront_encoded).abs() + 0j

    def forward(self, wavefront_in):
        """
        Parameters
        ----------
        wavefront_in: Wavefront('bs', 'H', 'W')

        Returns
        -------
        wavefront_encoded, wavefront_decoded : torch.Wavefront
            Encoded and decoded wavefronts.
        """
        # encode
        wavefront_encoded = self.encode(wavefront_in)
        # decode from encoded
        wavefront_decoded = self.decode(wavefront_encoded)
        # results to calculate loss
        return wavefront_encoded, wavefront_decoded

    def to_specs(self) -> Iterable[ParameterSpecs | SubelementSpecs]:
        """
        Returns the encoder and decoder specifications.

            Args:
                None

            Returns:
                Iterable[ParameterSpecs | SubelementSpecs]: An iterable containing
                SubelementSpecs for the encoder and decoder, each holding a
                LinearOpticalSetup representing their respective elements.
        """
        return (
            SubelementSpecs("Encoder", LinearOpticalSetup(self.encoder_elements)),
            SubelementSpecs("Decoder", LinearOpticalSetup(self.decoder_elements)),
        )

    def to(self, device: str | torch.device | int) -> "LinearAutoencoder":
        if self.__device == torch.device(device):
            return self

        return LinearAutoencoder(
            sim_params=self.sim_params,
            encoder_elements_list=self.encoder_elements,
            decoder_elements_list=self.decoder_elements,
            to_return=self.to_return,
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
