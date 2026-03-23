from typing import TYPE_CHECKING, Iterable
from torch import nn
from svetlanna.specs import ParameterSpecs, SubelementSpecs
from svetlanna import Wavefront
from svetlanna import LinearOpticalSetup
from svetlanna.elements import Element
from svetlanna.visualization import ElementHTML, jinja_env


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

        Examples
        --------
        ```python
        import svetlanna as sv
        from svetlanna.visualization import show_structure

        sim_params = ...

        linear_autoencoder = sv.networks.LinearAutoencoder(
            encoder_elements=(
                sv.elements.FreeSpace(
                    simulation_parameters=sim_params, distance=0.1, method="AS"
                ),
                sv.elements.ThinLens(simulation_parameters=sim_params, focal_length=0.1),
                sv.elements.FreeSpace(
                    simulation_parameters=sim_params, distance=0.1, method="AS"
                ),
            ),
            decoder_elements=(
                sv.elements.FreeSpace(
                    simulation_parameters=sim_params, distance=0.1, method="AS"
                ),
            )
        )

        show_structure(linear_autoencoder)
        ```
        Output (in IPython environment):
        <iframe
        src="show_structure_LinearAutoencoder.html"
        style="width:100%; height:400px; border: 0; color-scheme: inherit;" allowtransparency="true"></iframe>
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

    @staticmethod
    def _widget_html_(
        index: int, name: str, element_type: str | None, subelements: list[ElementHTML]
    ) -> str:
        return jinja_env.get_template("widget_autoencoder.html.jinja").render(
            index=index, name=name, subelements=subelements
        )

    if TYPE_CHECKING:

        def __call__(self, input_wavefront: Wavefront) -> Wavefront: ...
