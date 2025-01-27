import torch
from typing import Callable, Dict
from .element import Element
from ..simulation_parameters import SimulationParameters
from ..wavefront import Wavefront


# TODO: add docstrings
class FunctionModule(torch.nn.Module):
    """A class for transforming an arbitrary function with multiple parameters.
    Allows training function parameters

    Parameters
    ----------
    torch : _type_
        _description_
    """
    def __init__(self, function: Callable, function_parameters: Dict) -> None:
        """Constructor method

        Parameters
        ----------
        function : Callable
            Arbitrary function with several parameters
        function_parameters : Dict
            Parameters of the function
        """
        super(FunctionModule, self).__init__()
        self.function = function
        self.function_parameters = function_parameters

        # convert the parameters into an object capable of learning
        if self.function_parameters:
            for name, value in self.function_parameters.items():
                if isinstance(value, torch.nn.Parameter):
                    self.register_parameter(name, value)
                elif isinstance(value, torch.Tensor):
                    self.register_buffer(name, value)

    def forward(
        self,
        function_argument: float | torch.Tensor
    ) -> Callable:
        """forward method for a class inherited from torch.nn.Module

        Parameters
        ----------
        function_argument : float | torch.Tensor
            Argument of the function

        Returns
        -------
        Callable
            Function with trainable parameters
        """

        if self.function_parameters:
            return self.function(function_argument, **self.function_parameters)
        else:
            return self.function(function_argument)


class NonlinearElement(Element):
    """A class representing a nonlinear optical element with a given intensity
    response function. Preserves the phase distribution of the incident
    wavefront

    Parameters
    ----------
    Element : _type_
        _description_
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        response_function: Callable[[torch.Tensor], torch.Tensor],
        response_parameters: Dict = None
    ):
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Class exemplar, that describes optical system
        response_function : Callable[[torch.Tensor], torch.Tensor]
            Intensity response function
        response_parameters : Dict, optional
            Parameters of the response_function. Parameters converted to
            svetlanna.Parameter(value) will be trained, by default None
        """

        super().__init__(simulation_parameters)

        self.response_function = FunctionModule(
            response_function,
            response_parameters
        )

    def forward(self, input_field: Wavefront) -> Wavefront:
        """Method calculating the wavefront after passing a nonlinear optical
        element

        Parameters
        ----------
        input_field : Wavefront
            Wavefront before the nonlinear optical element

        Returns
        -------
        Wavefront
            Wavefront passing through a nonlinear optical element
        """
        transformed_intensity = self.response_function.forward(
            input_field.intensity
        )
        # preserve the phase of the incident wavefront
        phase = input_field.phase

        return Wavefront(
            torch.sqrt(transformed_intensity) * torch.exp(1j * phase)
        )
