import torch
from typing import Callable
from .element import Element
from ..simulation_parameters import SimulationParameters
from ..wavefront import Wavefront


class NonlinearElement(Element):
    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        response_function: Callable[[torch.Tensor], torch.Tensor],
    ):
        """Nonlinear optical element with a given response function.
        The response function takes incident wavefront.

        Example:
        > response_function = lambda x: torch.polar(torch.sqrt(x.abs()), x.angle())


        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Simulation parameters
        response_function : Callable[[torch.Tensor], torch.Tensor]
            Function that describes the nonlinear response of the element
        """

        super().__init__(simulation_parameters)

        self.response_function = self.process_parameter(
            "response_function", response_function
        )

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        wavefront = self.response_function(incident_wavefront)
        return Wavefront(wavefront)
