from .element import Element
from ..simulation_parameters import SimulationParameters
import torch


# TODO: check docstrings
class SpatialLightModulator(Element):
    """A class that described the field after propagating through the
    Spatial Light Modulator with a given phase mask

    Parameters
    ----------
    Element : _type_
        _description_
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        mask: torch.Tensor,
        number_of_levels: int = 256
    ):
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Class exemplar, that describes optical system
        mask : torch.Tensor
            Phase mask in grey format for the SLM, every element must be int
        number_of_levels : int, optional
            Number of phase quantization levels for the SLM, by default 256
        """

        super().__init__(simulation_parameters)

        self.mask = mask
        self.number_of_levels = number_of_levels

        self.transmission_function = torch.exp(
            1j * 2 * torch.pi / self.number_of_levels * self.mask
        )

    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Method that calculates the field after propagating through the SLM

        Parameters
        ----------
        input_field : torch.Tensor
            Field incident on the SLM

        Returns
        -------
        torch.Tensor
            The field after propagating through the SLM
        """

        return input_field * self.transmission_function

    def reverse(self, transmission_field: torch.Tensor) -> torch.Tensor:
        """Method that calculates the field after passing the SLM in back
        propagation

        Parameters
        ----------
        transmission_field : torch.Tensor
            Field incident on the SLM in back propagation
            (transmitted field in forward propagation)

        Returns
        -------
        torch.tensor
            Field transmitted on the SLM in back propagation
            (incident field in forward propagation)
        """

        return transmission_field * torch.conj(self.transmission_function)

    def get_transmission_function(self):
        """Method which returns the transmission function of
        the SLM

        Returns
        -------
        torch.Tensor
            transmission function of the SLM
        """

        return self.transmission_function
