from typing import Iterable
from svetlanna.elements import Element
from svetlanna import Wavefront, SimulationParameters

import torch
from torch import nn


class RecurrentReservoir(nn.Module):
    """
    A recurrent reservoir. Supposed to by an ineducable element of a setup,
    which means that all elements mustn't have parameters tu update.
    All parameters are bounded or defined as a buffer!
    """
    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        forward_elements: Iterable[Element],
        backward_elements: Iterable[Element],
        attenuation_forward: float = 0,
        attenuation_backward: float = 0,
        time_masking: torch.Tensor = torch.tensor([1.]),
        init_hidden: Wavefront | None = None,
        device: str | torch.device = torch.get_default_device(),
    ) -> None:
        """
        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Simulation parameters for a task.
        forward_elements : Iterable[Element]
            A set of optical elements a sum of input and hidden wavefront propagates through.
        backward_elements : Iterable[Element]
            A set of optical elements for a feedback (to get a hidden state).
            An output wavefront (after forward elements) goes through backward elements and becomes a new hidden state.
        attenuation_forward, attenuation_backward: float
            Coefficients which characterize a part of energy that dissipates while forward and backward propagation.
            Values lie between 0 and 1.
        time_masking : torch.Tensor
            Turning one input Wavefront in several Wavefronts (by sequentially multiplying on values from a mask).
        init_hidden : Wavefront | None
            A n initial hidden wavefront ("a cold state").
        device : torch.device
            A device for calculations.
        """
        super().__init__()

        self.simulation_parameters = simulation_parameters

        self.__device = device
        if self.simulation_parameters.device is not self.__device:
            # TODO: right way to compare devices? check how it works?
            self.simulation_parameters.__device = self.simulation_parameters.to(self.__device)

        # TODO: check if all parameters are bounded oa a buffer?
        self.forward_elements = [el.to(self.__device) for el in forward_elements]
        self.backward_elements = [el.to(self.__device) for el in backward_elements]

        # attenuation constants
        self.attenuation_forward = attenuation_forward
        self.attenuation_backward = attenuation_backward

        # time masking for each input wavefront
        self.time_masking = time_masking.to(self.__device)

        # a hidden state initialization
        if init_hidden is None:
            self.hidden_state = None
            self.reset_hidden()
        else:
            self.hidden_state = init_hidden.to(self.__device)

    def reset_hidden(self):
        """
        Resets a "hidden" wavefront to zeros.

        Returns
        -------
        Wavefront
            A zero wavefront as an initial "hidden" wavefront.
        """
        self.hidden_state = Wavefront(
            torch.zeros(
                # TODO: right way to get a complete wavefront size?
                size=self.simulation_parameters.axes_size(
                    axs=('H', 'W')
                )
            )
        ).to(self.__device)

    def forward(self, input_wavefront: Wavefront) -> Wavefront:
        """
        A forward function for a network assembled from elements.

        Parameters
        ----------
        input_wavefront : torch.Tensor
            A wavefront that enters the optical network.

        Returns
        -------
        torch.Tensor
            An output wavefront after propagation of an input wavefront (+ hidden wavefront)
            through a forward net (output of the network).
        """
        for mask_value in self.time_masking:  # turning one time step in several time steps

            output_wavefront = mask_value * input_wavefront + self.hidden_state
            for el in self.forward_elements:
                output_wavefront = el.forward(output_wavefront)
            output_wavefront = output_wavefront * (1 - self.attenuation_forward) ** (1 / 2)

            # get a next hidden by propagation of the output
            hidden_state = output_wavefront
            for el in self.backward_elements:
                hidden_state = el.forward(hidden_state)
            # update hidden state
            self.hidden_state = hidden_state * (1 - self.attenuation_backward) ** (1 / 2)

        return output_wavefront

    def to(self, device: str | torch.device | int) -> 'RecurrentReservoir':
        if self.__device == torch.device(device):
            return self

        return RecurrentReservoir(
            simulation_parameters=self.simulation_parameters,
            forward_elements=self.forward_elements,
            backward_elements=self.backward_elements,
            attenuation_forward=self.attenuation_forward,
            attenuation_backward=self.attenuation_backward,
            time_masking=self.time_masking,
            init_hidden=self.hidden_state,
            device=device
        )

    @property
    def device(self) -> str | torch.device | int:
        return self.__device
