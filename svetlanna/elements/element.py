from abc import ABCMeta, abstractmethod
from torch import nn
from torch import Tensor
from ..simulation_parameters import SimulationParameters
from ..specs import ReprRepr, ParameterSpecs
from typing import Iterable, Literal
from ..parameters import BoundedParameter, Parameter
import torch

INNER_PARAMETER_SUFFIX = '_svtlnn_inner_parameter'


# TODO: check docstring
class Element(nn.Module, metaclass=ABCMeta):
    """A class that describes each element of the system

    Parameters
    ----------
    nn : _type_
        _description_
    metaclass : _type_, optional
        _description_, by default ABCMeta
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters
    ) -> None:
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Class exemplar that describes the optical system
        """

        super().__init__()

        self.simulation_parameters = simulation_parameters

        self._x_size = self.simulation_parameters.x_size
        self._y_size = self.simulation_parameters.y_size
        self._x_nodes = self.simulation_parameters.x_nodes
        self._y_nodes = self.simulation_parameters.y_nodes
        self._wavelength = self.simulation_parameters.wavelength

        self._x_linspace = torch.linspace(
            -self._x_size/2, self._x_size/2, self._x_nodes
        )
        self._y_linspace = torch.linspace(
            -self._y_size/2, self._y_size/2, self._y_nodes
        )
        self._x_grid, self._y_grid = torch.meshgrid(
            self._x_linspace, self._y_linspace, indexing='xy'
        )

    # TODO: check doctrings
    @abstractmethod
    def forward(self, Ein: Tensor) -> Tensor:

        """Forward propagation through the optical element"""

    def to_specs(self) -> Iterable[ParameterSpecs]:

        """Create specs"""

        for (name, parameter) in self.named_parameters():

            # BoundedParameter and Parameter support
            if name.endswith(INNER_PARAMETER_SUFFIX):
                name = name.removesuffix(INNER_PARAMETER_SUFFIX)
                parameter = self.__getattribute__(name)

            yield ParameterSpecs(
                name=name,
                representations=(ReprRepr(value=parameter),)
            )

    # TODO: create docstrings
    def __setattr__(self, name: str, value: Tensor | nn.Module) -> None:

        # BoundedParameter and Parameter are handled by pointing
        # auxiliary attribute on them with a name plus INNER_PARAMETER_SUFFIX
        if isinstance(value, (BoundedParameter, Parameter)):
            super().__setattr__(
                name + INNER_PARAMETER_SUFFIX, value.inner_parameter
            )

        return super().__setattr__(name, value)
