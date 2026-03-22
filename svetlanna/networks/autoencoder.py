from typing import TYPE_CHECKING, Iterable
from torch import nn
from svetlanna.specs import ParameterSpecs, SubelementSpecs
from svetlanna import Wavefront
from svetlanna import LinearOpticalSetup
from svetlanna.elements import Element


class LinearAutoencoder(nn.Module):
    """
    A simple autoencoder network consisting of consistent encoder and decoder
    for a simultaneous training.
    """

    def __init__(
        self,
        encoder_elements: Iterable[Element],
        decoder_elements: Iterable[Element],
    ):
        """
        Parameters
        ----------
        encoder_elements : Iterable[Element]
            The encoder elements.
        decoder_elements : Iterable[Element]
            The decoder elements.
        """
        super().__init__()

        self.encoder = LinearOpticalSetup(encoder_elements)
        self.decoder = LinearOpticalSetup(decoder_elements)

    def encode(self, input_wavefront: Wavefront) -> Wavefront:
        """
        Propagation through the encoder part - encode a wavefront (input).

        Returns
        -------
        Wavefront
            An encoded input wavefront.
        """
        return self.encoder(input_wavefront)

    def decode(self, wavefront_encoded: Wavefront) -> Wavefront:
        """
        Propagation through the decoder part - decode an encoded wavefront.

        Returns
        -------
        Wavefront
            A decoded wavefront.
        """
        return self.decoder(wavefront_encoded)

    def forward(self, input_wavefront: Wavefront) -> Wavefront:
        wavefront_encoded = self.encode(input_wavefront)
        wavefront_decoded = self.decode(wavefront_encoded)
        return wavefront_decoded

    def to_specs(self) -> Iterable[ParameterSpecs | SubelementSpecs]:

        return [
            SubelementSpecs("Encoder", self.encoder),
            SubelementSpecs("Decoder", self.decoder),
        ]

    if TYPE_CHECKING:

        def __call__(self, input_wavefront: Wavefront) -> Wavefront: ...
