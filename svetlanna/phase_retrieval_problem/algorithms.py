from typing import Callable

import torch

from svetlanna.phase_retrieval_problem import phase_retrieval_result as prr


def gerchberg_saxton_algorithm(
    target_intensity: torch.Tensor,
    source_intensity: torch.Tensor,
    forward: Callable,
    reverse: Callable,
    initial_approximation: torch.Tensor,
    tol: float,
    maxiter: int,
    target_phase: torch.Tensor | None = None,
    target_region: torch.Tensor | None = None,
) -> prr.PhaseRetrievalResult:
    """Gerchberg-Saxton algorithm(GS) for solving the phase retrieval problem

    Parameters
    ----------
    target_intensity : torch.Tensor
        Intensity profile in the Fourier plane
    source_intensity : torch.Tensor
        Intensity distribution before the optical setup(in the image plane)
    forward : Callable
        Function which describes forward propagation through the optical system
    reverse : Callable
        Function which describes reverse propagation through the optical system
    initial_approximation : torch.Tensor
        Initial approximation for the phase profile
    tol : float
        Accuracy for the algorithm
    maxiter : int
        Maximum number of iterations
    target_phase : torch.Tensor | None, optional
        Phase profile on the Fourier plane(optional for the generating
        target intensity profile problem) for reconstructing phase profile
        problem, by default None
    target_region : torch.Tensor | None, optional
        Region to preserve phase and amplitude profiles in the Fourier plane
        for reconstructing phase profile problem(optional for the generating
        target intensity profile problem), by default None

    Returns
    -------
    prr.PhaseRetrievalResult
        Exemplar of class PhaseRetrievalResult which presents result of
        optimization
    """

    cost_func_evolution: list = []

    source_amplitude = torch.sqrt(source_intensity)
    target_amplitude = torch.sqrt(target_intensity)

    input_field = source_amplitude * torch.exp(
        1j * initial_approximation
    )

    number_of_iterations = 0

    while True:

        output_field = forward(input_field)

        output_field_phase = torch.angle(output_field)
        output_field_phase = output_field_phase + (
            2 * torch.pi
        ) * (output_field_phase < 0).float()

        if (target_phase is not None) and (target_region is not None):

            updated_phase = target_phase * target_region + (
                1. - target_region
            ) * output_field_phase

            updated_output_field = target_amplitude * torch.exp(
                1j * updated_phase
            )

        else:

            updated_output_field = target_amplitude * torch.exp(
                1j * output_field_phase
            )

        output_intensity_profile = torch.pow(
            torch.abs(updated_output_field), 2
        )

        updated_input_field = reverse(updated_output_field)

        error = torch.sqrt(
            torch.sum(
                torch.pow(
                    torch.sqrt(output_intensity_profile) - target_amplitude, 2
                )
            ) / torch.sum(target_intensity)
        )

        phase_function = torch.angle(updated_input_field)
        phase_function = phase_function + (
            2 * torch.pi
        ) * (phase_function < 0.).float()

        number_of_iterations += 1
        cost_func_evolution.append(error)

        if (torch.abs(error) <= tol) or (
            number_of_iterations >= maxiter
        ):

            break

        else:

            input_field = source_amplitude * torch.exp(
                1j * phase_function
            )

    phase_retrieval_result = prr.PhaseRetrievalResult(
        solution=phase_function,
        cost_func=error,
        cost_func_evolution=cost_func_evolution,
        number_of_iterations=number_of_iterations,

    )
    return phase_retrieval_result


def hybrid_input_output(
    target_intensity: torch.Tensor,
    source_intensity: torch.Tensor,
    forward: Callable,
    reverse: Callable,
    initial_approximation: torch.Tensor,
    tol: float,
    maxiter: int,
    target_phase: torch.Tensor | None = None,
    target_region: torch.Tensor | None = None,
    constant_factor: float = 0.9
) -> prr.PhaseRetrievalResult:
    """Hybrid Input-Output(HIO) algorithm for for solving the phase retrieval
    problem

    Parameters
    ----------
    target_intensity : torch.Tensor
        Intensity profile in the Fourier plane
    source_intensity : torch.Tensor
        Intensity distribution before the optical setup(in the image plane)
    forward : Callable
        Function which describes forward propagation through the optical system
    reverse : Callable
        Function which describes reverse propagation through the optical system
    initial_approximation : torch.Tensor
        Initial approximation for the phase profile
    tol : float
        Accuracy for the algorithm
    maxiter : int
        Maximum number of iterations
    target_phase : torch.Tensor | None, optional
        Phase profile on the Fourier plane(optional for the generating
        target intensity profile problem) for reconstructing phase profile
        problem, by default None
    target_region : torch.Tensor | None, optional
        Region to preserve phase and amplitude profiles in the Fourier plane
        for reconstructing phase profile problem(optional for the generating
        target intensity profile problem), by default None
    constant_factor: float
        Learning rate value for the HIO algorithm, by default 0.9

    Returns
    -------
    prr.PhaseRetrievalResult
        Exemplar of class PhaseRetrievalResult which presents result of
        optimization
    """
    cost_func_evolution: list = []

    source_amplitude = torch.sqrt(source_intensity)
    target_amplitude = torch.sqrt(target_intensity)

    input_field = source_amplitude * torch.exp(
        1j * initial_approximation
    )

    number_of_iterations = 0

    support_constrain = (
        target_amplitude > torch.max(target_amplitude) / 10
    ).float()

    while True:

        output_field = forward(input_field)

        output_field_phase = torch.angle(output_field)
        output_field_phase = output_field_phase + (
            2 * torch.pi
        ) * (output_field_phase < 0).float()

        if (target_phase is not None) and (target_region is not None):

            updated_phase = target_phase * target_region + (
                1. - target_region
            ) * output_field_phase

            updated_output_field = target_amplitude * torch.exp(
                1j * updated_phase
            ) - (1. - support_constrain) * constant_factor * output_field

        else:

            updated_output_field = target_amplitude * torch.exp(
                1j * output_field_phase
            ) - (1. - support_constrain) * constant_factor * output_field

        updated_input_field = reverse(updated_output_field)

        output_intensity_profile = torch.pow(
            torch.abs(updated_output_field), 2
        )

        error = torch.sqrt(
            torch.sum(
                torch.pow(
                    torch.sqrt(output_intensity_profile) - target_amplitude, 2
                )
            ) / torch.sum(target_intensity)
        )

        phase_function = torch.angle(updated_input_field)
        phase_function = phase_function + (
            2 * torch.pi
        ) * (phase_function < 0.).float()

        number_of_iterations += 1
        cost_func_evolution.append(error)

        if (torch.abs(error) <= tol) or (
            number_of_iterations >= maxiter
        ):
            break

        else:

            input_field = source_amplitude * torch.exp(1j * phase_function)

    phase_retrieval_result = prr.PhaseRetrievalResult(
        solution=phase_function,
        cost_func=error,
        cost_func_evolution=cost_func_evolution,
        number_of_iterations=number_of_iterations
    )

    return phase_retrieval_result
