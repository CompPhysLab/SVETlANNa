from typing import TYPE_CHECKING, Iterable, cast
from torch import nn
from svetlanna.specs import ParameterSpecs, SubelementSpecs, Specsable
from .base import LinearOpticalSetupLike
from svetlanna import Wavefront


class LinearAutoencoder(nn.Module):
    """
    A simple autoencoder network consisting of consistent encoder and decoder
    for a simultaneous training.
    """

    def __init__(
        self,
        encoder_element: LinearOpticalSetupLike,
        decoder_element: LinearOpticalSetupLike,
    ):
        """
        Parameters
        ----------
        encoder_element : LinearOpticalSetupLike
            The encoder element.
        decoder_element : LinearOpticalSetupLike
            The decoder element.
        """
        super().__init__()

        self.encoder_element = encoder_element
        self.decoder_element = decoder_element

    def encode(self, input_wavefront: Wavefront) -> Wavefront:
        """
        Propagation through the encoder part - encode a wavefront (input).

        Returns
        -------
        Wavefront
            An encoded input wavefront.
        """
        return self.encoder_element(input_wavefront)

    def decode(self, wavefront_encoded: Wavefront) -> Wavefront:
        """
        Propagation through the decoder part - decode an encoded wavefront.

        Returns
        -------
        Wavefront
            A decoded wavefront.
        """
        return self.decoder_element(wavefront_encoded)

    def forward(self, input_wavefront: Wavefront) -> Wavefront:
        wavefront_encoded = self.encode(input_wavefront)
        wavefront_decoded = self.decode(wavefront_encoded)
        return wavefront_decoded

    def to_specs(self) -> Iterable[ParameterSpecs | SubelementSpecs]:
        specs = []

        if hasattr(self.encoder_element, "to_specs"):
            encoder_element = cast(Specsable, self.encoder_element)
            specs.append(SubelementSpecs("Encoder", encoder_element))

        if hasattr(self.decoder_element, "to_specs"):
            decoder_element = cast(Specsable, self.decoder_element)
            specs.append(SubelementSpecs("Decoder", decoder_element))

        return specs

    if TYPE_CHECKING:

        def __call__(self, input_wavefront: Wavefront) -> Wavefront: ...
