import torch
from typing import Literal
from svetlanna import SimulationParameters
from svetlanna import LinearOpticalSetup
from typing import overload
from typing import Protocol


class SetupLike(Protocol):
    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        ...

    def reverse(self, transmission_field: torch.Tensor) -> torch.Tensor:
        ...



@overload
def retrieve_phase(
    source_intensity: torch.Tensor,
    optical_setup: LinearOpticalSetup | SetupLike,
    target_intensity: torch.Tensor,
    initial_phase: torch.Tensor = None,
    method: Literal['GS', 'HIO'] = 'GS',
    maxiter: int = 500,
    tol: float = 1e-3
):
    ...


@overload
def retrieve_phase(
    source_intensity: torch.Tensor,
    optical_setup: LinearOpticalSetup | SetupLike,
    target_intensity: torch.Tensor,
    target_phase: torch.Tensor,
    target_region: torch.Tensor,
    initial_phase: torch.Tensor = None,
    method: Literal['GS', 'HIO'] = 'GS',
    maxiter: int = 500,
    tol: float = 1e-3
):
    ...


def retrieve_phase(
    source_intensity: torch.Tensor,
    optical_setup: LinearOpticalSetup | SetupLike,
    target_intensity: torch.Tensor,
    target_phase: torch.Tensor | None = None,
    target_region: torch.Tensor = None,
    initial_phase: torch.Tensor = None,
    method: Literal['GS', 'HIO'] = 'GS',
    maxiter: int = 500,
    tol: float = 1e-3
):

    forward_propagation = optical_setup.forward
    reverse_propagation = optical_setup.reverse

    if initial_phase is None:
        initial_phase = 2 * torch.pi * torch.rand_like(source_intensity)

    if method == 'GS':

        phase_distribution = gerchberg_saxton_algorithm(
            target_intensity=target_intensity,
            source_intensity=source_intensity,
            forward=forward_propagation,
            reverse=reverse_propagation,
            initial_approximation=initial_phase,
            tol=tol,
            maxiter=maxiter
        )
    else:
        raise ValueError('Unknown optimization method')

    return phase_distribution


def gerchberg_saxton_algorithm(
    target_intensity: torch.Tensor,
    source_intensity: torch.Tensor,
    forward,
    reverse,
    initial_approximation: torch.Tensor,
    tol: float,
    maxiter: int
):

    source_amplitude = torch.sqrt(source_intensity)
    target_amplitude = torch.sqrt(target_intensity)

    incident_field = source_amplitude * torch.exp(
        1j * initial_approximation
    )

    number_of_iterations = 0
    current_error = 100.

    while True:

        output_field = forward(incident_field)

        target_field = target_amplitude * output_field / torch.abs(
            output_field
        )

        current_target_intensity = torch.pow(torch.abs(target_field), 2)

        source_field = reverse(target_field)

        std = torch.std(current_target_intensity - target_intensity)
        if (torch.abs(current_error - std) <= tol) or (
            number_of_iterations >= maxiter
        ):
            phase_function = (
                torch.angle(incident_field) + 2 * torch.pi
            ) % (2 * torch.pi)
            break

        else:

            incident_field = source_amplitude * torch.exp(
                1j * (
                    (
                        torch.angle(source_field) + 2 * torch.pi
                        ) % (2 * torch.pi)
                    )
                )
            number_of_iterations += 1
            current_error = std

    phase_function = phase_function % (2 * torch.pi)

    return phase_function
