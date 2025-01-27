import torch

from .element import Element
from ..simulation_parameters import SimulationParameters
from ..parameters import OptimizableFloat
from ..wavefront import Wavefront, mul
from ..axes_math import tensor_dot


class ThinLens(Element):
    """A class that described the field after propagating through the
    thin lens.
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        focal_length: OptimizableFloat,
        radius: float = torch.inf
    ):
        """Thin lens element.

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            An instance describing the optical system's simulation parameters.
        focal_length : OptimizableFloat
            The focal length of the lens.
            Must be greater than 0 for a converging lens.
        radius : float
            The radius of the thin lens.
        """

        super().__init__(simulation_parameters)

        self.focal_length = self.process_parameter(
            'focal_length', focal_length
        )
        self.radius = self.process_parameter(
            'radius', radius
        )

        # Compute wave_number as a tensor
        wave_number, axes = tensor_dot(
            2 * torch.pi / self.simulation_parameters.axes.wavelength,
            torch.tensor([[1]]),
            'wavelength',
            ('H', 'W')
        )  # shape: ('wavelength', 1, 1) or (1, 1)

        # Registering Buffer for _wave_number
        self._wave_number = self.make_buffer(
            '_wave_number',
            wave_number
        )

        self._calc_axes = axes  # axes tuple used during calculations

        x_linear = self.simulation_parameters.axes.W
        y_linear = self.simulation_parameters.axes.H

        x_grid = x_linear[None, :]  # shape: (1, 'W')
        y_grid = y_linear[:, None]  # shape: ('H', 1)

        # Registering Buffer for _radius_squared
        self._radius_squared = self.make_buffer(
            '_radius_squared',
            x_grid**2 + y_grid**2
        )

        # Create a mask that acts as an aperture:
        # Regions of the field where x^2 + y^2 > radius^2
        # will propagate with no change in phase.
        if self.radius == torch.inf:
            self._radius_mask = 1.0
        else:
            self._radius_mask = self.make_buffer(
                '_radius_mask',
                (self._radius_squared <= self.radius**2).to(
                    dtype=torch.get_default_dtype()  # cast bool to float
                )
            )

    @property
    def transmission_function(self) -> torch.Tensor:
        return torch.exp(
            - 1j * self._radius_mask * self._radius_squared * (
                self._wave_number / (2 * self.focal_length)
            )
        )

    def get_transmission_function(self) -> torch.Tensor:
        """Returns the transmission function of the thin lens.

        Returns
        -------
        torch.Tensor
            The transmission function of the thin lens.
        """

        return self.transmission_function

    def forward(self, input_field: Wavefront) -> Wavefront:
        """Calculates the field after propagation through the thin lens.

        Parameters
        ----------
        input_field : Wavefront
            The field incident on the thin lens.

        Returns
        -------
        Wavefront
            The field after propagation through the thin lens.
        """

        return mul(
            input_field,
            self.transmission_function,
            self._calc_axes,
            self.simulation_parameters
        )

    def reverse(self, transmission_field: Wavefront) -> Wavefront:
        """Calculates the field after passing through the lens during
        back propagation.

        Parameters
        ----------
        transmission_field : Wavefront
            The field incident on the lens during back propagation.
            This corresponds to the transmitted field in forward propagation.

        Returns
        -------
        Wavefront
            The field transmitted through the lens during back propagation.
            This corresponds to the incident field in forward propagation.
        """
        return mul(
            transmission_field,
            torch.conj(self.transmission_function),
            self._calc_axes,
            self.simulation_parameters
        )
