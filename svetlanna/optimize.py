import torch

from svetlanna import SimulationParameters


class Optimizer:
    """Class for solving the phase retrieval problem
    """

    def __init__(
        self,
        simulation_parameters: SimulationParameters,
        target_intensity: torch.Tensor,
        source_intensity: torch.Tensor,
        number_of_levels: int = 256
    ):
        """Constructor method

        Parameters
        ----------
        simulation_parameters : SimulationParameters
            Class exemplar that describes the optical system
        target_intensity : torch.Tensor
            The intensity distribution to be obtained
        source_intensity : torch.Tensor
            Intensity distribution in front of the optimized diffractive
            optical element
        number_of_levels : int, optional
            Number of phase quantization levels for the SLM, by default 256
        """

        self.simulation_parameters = simulation_parameters

        self._x_size = self.simulation_parameters.x_size
        self._y_size = self.simulation_parameters.y_size
        self._x_nodes = self.simulation_parameters.x_nodes
        self._y_nodes = self.simulation_parameters.y_nodes
        self._wavelength = self.simulation_parameters.wavelength

        self._target_intensity = target_intensity
        self._source_intensity = source_intensity
        self._number_of_levels = number_of_levels

        self._target_amplitude = torch.sqrt(self._target_intensity)
        self._source_amplitude = torch.sqrt(self._source_intensity)

        self._x_linspace = torch.linspace(
            -self._x_size/2, self._x_size/2, self._x_nodes
        )
        self._y_linspace = torch.linspace(
            -self._y_size/2, self._y_size/2, self._y_nodes
        )
        self._x_grid, self._y_grid = torch.meshgrid(
            self._x_linspace, self._y_linspace, indexing='xy'
        )

    def gerchberg_saxton_algorithm(
            self,
            forward,
            reverse,
            initial_approximation: torch.Tensor,
            tol: float = 1e-3,
            maxiter: int = 500
    ):
        """Gerchberg Saxton's optimization algorithm

        Parameters
        ----------
        forward : _type_
            Function returning field at direct passage of the optical system
        reverse : _type_
            Function returning field at return passage of the optical system
        initial_approximation : torch.Tensor
            Initial approximation for the phase mask
        tol : float, optional
            Exit criterion of the optimization algorithm, by default 1e-3
        maxiter : int, optional
            maximum number of iterations for the algorithm, by default 500

        Returns
        -------
        torch.Tensor, torch.Tensor
            Phase function in the range from 0 to 2pi and phase mask in grey
            format with selected number of quantization levels
        """

        incident_field = self._source_amplitude * torch.exp(
            1j * initial_approximation
        )

        number_of_iterations = 0
        current_error = 100.

        while True:

            output_field = forward(incident_field)

            target_field = self._target_amplitude * output_field / torch.abs(
                output_field
            )

            current_target_intensity = torch.pow(torch.abs(target_field), 2)

            source_field = reverse(target_field)

            std = torch.std(current_target_intensity - self._target_intensity)
            print(std)
            if (torch.abs(current_error - std) <= tol) or (
                number_of_iterations >= maxiter
            ):
                phase_function = (
                    torch.angle(incident_field) + 2 * torch.pi
                ) % (2 * torch.pi)
                break

            else:

                incident_field = self._source_amplitude * torch.exp(
                    1j * (
                        (
                            torch.angle(source_field) + 2 * torch.pi
                            ) % (2 * torch.pi)
                        )
                    )
                number_of_iterations += 1
                current_error = std

        phase_function = phase_function % (2 * torch.pi)

        step = 2 * torch.pi / self._number_of_levels
        phase_mask = phase_function // step
        phase_function = phase_mask * step

        return phase_function, phase_mask
