import torch
from .element import Element
from ..simulation_parameters import SimulationParameters
from ..wavefront import Wavefront, mul
from ..parameters import OptimizableTensor


class DiffractiveLayer(Element):
    """A class that described the field after propagating through the
    passive diffractive layer with a given phase mask
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        mask: OptimizableTensor,
        mask_norm: float = 2 * torch.pi
    ):
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Simulation parameters
        mask : OptimizableTensor
            Phase mask
        mask_norm : float, optional
            This value will be used as following:
            the phase addition is equal to `2*torch.pi * mask / mask_norm`.
            By default, `2*torch.pi`
        """

        super().__init__(simulation_parameters)

        self.mask = mask
        self.mask_norm = mask_norm

    @property
    def transmission_function(self):
        return torch.exp(
            (2j * torch.pi / self.mask_norm) * self.mask
        )

    def forward(self, input_field: Wavefront) -> Wavefront:
        """Method that calculates the field after propagating through the SLM

        Parameters
        ----------
        input_field : Wavefront
            Field incident on the SLM

        Returns
        -------
        Wavefront
            The field after propagating through the SLM
        """
        return mul(
            input_field,
            self.transmission_function,
            ('H', 'W'),
            self.simulation_parameters
        )

    def reverse(self, transmitted_field: Wavefront) -> Wavefront:
        """Method that calculates the field after passing the SLM in back
        propagation

        Parameters
        ----------
        transmitted_field : Wavefront
            Field incident on the SLM in back propagation
            (transmitted field in forward propagation)

        Returns
        -------
        Wavefront
            Field transmitted on the SLM in back propagation
            (incident field in forward propagation)
        """
        return mul(
            transmitted_field,
            torch.conj(self.transmission_function),
            ('H', 'W'),
            self.simulation_parameters
        )
