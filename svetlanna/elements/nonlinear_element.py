from typing import Callable
from .element import Element
from ..simulation_parameters import SimulationParameters
from ..wavefront import Wavefront


class NonlinearElement(Element):
    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        response_function: Callable[[Wavefront], Wavefront],
    ):
        r"""
        Nonlinear optical element with a given response function.
        The response function takes an incident wavefront and returns the modified wavefront.

        Examples
        --------
        Suppose the response function is defined as
        $E^\text{out} = \sqrt{|E^\text{in}|}e^{i \arg(E^\text{in})}$:
        ```python hl_lines="8"
        import svetlanna as sv
        import torch

        sim_params = sv.SimulationParameters(...)

        sv.elements.NonlinearElement(
            simulation_parameters=sim_params,
            response_function = lambda E: torch.polar(torch.sqrt(E.abs()), E.angle())
        )
        ```

        If you want to optimize the parameters of the response function, you can use
        `svetlanna.PartialWithParameters` to wrap the response function with trainable parameters.
        For example, if the response function is defined as
        $E^\text{out} = |E^\text{in}|^a e^{i b \arg(E^\text{in})}$, where $0<a<1$ and
        $b$ are trainable, you can define the nonlinear element as follows:
        ```python hl_lines="1 2 6-10"
        def response_function(E, a, b):
            return torch.polar(E.abs()**a, b * E.angle())

        sv.elements.NonlinearElement(
            simulation_parameters=sim_params,
            response_function = sv.PartialWithParameters(
                response_function,
                a=sv.ConstrainedParameter(0.5, min_value=0.0, max_value=1.0),
                b=sv.Parameter(1.0),
            ),
        )
        ```

        You can also train a neural network inside the nonlinear element!
        ```python hl_lines="3"
        sv.elements.NonlinearElement(
            simulation_parameters=sim_params,
            response_function=my_neural_network,
        )
        ```


        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Simulation parameters.
        response_function : Callable[[Wavefront], Wavefront]
            Function that describes the nonlinear response of the element.
        """

        super().__init__(simulation_parameters)

        self.response_function = self.process_parameter(
            "response_function", response_function
        )

    def forward(self, incident_wavefront: Wavefront) -> Wavefront:
        wavefront = self.response_function(incident_wavefront)
        return Wavefront(wavefront)
