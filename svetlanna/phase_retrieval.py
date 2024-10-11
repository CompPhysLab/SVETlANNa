import torch
from typing import Literal
from typing import overload
from typing import Protocol

from svetlanna import LinearOpticalSetup


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
    maxiter: int = 100,
    tol: float = 1e-3,
    constant_factor: float = 0.5
) -> torch.Tensor:
    """Function for solving phase retrieval problem: generating target
    intensity profile or reconstructing the phase profile of the field

    Parameters
    ----------
    source_intensity : torch.Tensor
        Intensity distribution before the optical setup
    optical_setup : LinearOpticalSetup | SetupLike
        Optical system through which the beam is propagated
    target_intensity : torch.Tensor
        Intensity profile in the Fourier plane
    target_phase : torch.Tensor | None, optional
        Phase profile on the Fourier plane(optional for the generating
        target intensity profile problem)
    target_region : torch.Tensor, optional
        Region to preserve phase and amplitude profiles in the Fourier plane(
        optional for the generating target intensity profile problem)
    initial_phase : torch.Tensor, optional
        Initial approximation for the phase profile, by default None
    method : Literal[&#39;GS&#39;, &#39;HIO&#39;], optional
        Algorithms for phase retrieval problem, by default 'GS'
    maxiter : int, optional
        Maximum number of iterations, by default 100
    tol : float, optional
        Tolerance, by default 1e-3
    constant_factor : float, optional
        Learning rate parameter for the HIO method, by default 0.5

    Returns
    -------
    torch.Tensor
        Optimized phase profile from 0 to 2pi

    Raises
    ------
    ValueError
        Unknown optimization method
    """

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
            maxiter=maxiter,
            target_phase=target_phase,
            target_region=target_region
        )
    elif method == 'HIO':
        phase_distribution = hybrid_input_output(
            target_intensity=target_intensity,
            source_intensity=source_intensity,
            forward=forward_propagation,
            reverse=reverse_propagation,
            initial_approximation=initial_phase,
            tol=tol,
            maxiter=maxiter,
            target_phase=target_phase,
            target_region=target_region,
            constant_factor=constant_factor
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
    maxiter: int,
    target_phase: torch.Tensor | None = None,
    target_region: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gerchberg-Saxton algorithm(GS) for solving the phase retrieval problem

    Parameters
    ----------
    target_intensity : torch.Tensor
        Intensity profile in the Fourier plane
    source_intensity : torch.Tensor
        Intensity distribution before the optical setup(in the image plane)
    forward : _type_
        Function which describes forward propagation through the optical system
    reverse : _type_
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
    torch.Tensor
        Phase profile from 0 to 2pi
    """
    source_amplitude = torch.sqrt(source_intensity)
    target_amplitude = torch.sqrt(target_intensity)

    incident_field = source_amplitude * torch.exp(
        1j * initial_approximation
    )

    number_of_iterations = 0
    current_error = 100.

    y_nodes, x_nodes = source_amplitude.shape
    number_of_pixels = x_nodes * y_nodes

    while True:

        output_field = forward(incident_field)

        if (target_phase is not None) and (target_region is not None):

            # TODO: ask about surface
            current_phase_target = torch.angle(output_field)
            current_phase_target = current_phase_target + (
                2 * torch.pi
            ) * (current_phase_target < 0).float()

            # TODO: check product
            target_phase_distribution = target_phase * target_region + (
                1. - target_region
            ) * current_phase_target

            target_field = target_amplitude * torch.exp(
                1j * target_phase_distribution
            ) / torch.abs(output_field)

        else:

            target_field = target_amplitude * output_field / torch.abs(
                output_field
            )

        current_target_intensity = torch.pow(torch.abs(output_field), 2)

        source_field = reverse(target_field)

        error = torch.sqrt(
            torch.sum(
                torch.pow(
                    torch.sqrt(current_target_intensity) - target_amplitude, 2
                )
            ) / (torch.sum(target_intensity) * number_of_pixels)
        )

        # print(error)

        if (torch.abs(current_error - error) <= tol) or (
            number_of_iterations >= maxiter
        ):

            phase_function = torch.angle(incident_field)
            phase_function = phase_function + (
                2 * torch.pi
            ) * (phase_function < 0.).float()

            break

        else:

            current_phase_source = torch.angle(source_field)
            current_phase_source = current_phase_source + (
                2 * torch.pi
            ) * (current_phase_source < 0.)

            incident_field = source_amplitude * torch.exp(
                1j * current_phase_source
            )

            number_of_iterations += 1
            current_error = error

    return phase_function


def hybrid_input_output(
    target_intensity: torch.Tensor,
    source_intensity: torch.Tensor,
    forward,
    reverse,
    initial_approximation: torch.Tensor,
    tol: float,
    maxiter: int,
    target_phase: torch.Tensor | None = None,
    target_region: torch.Tensor | None = None,
    constant_factor: float = 0.9
):
    """Hybrid Input-Output algorithm for for solving the phase retrieval
    problem

    target_intensity : torch.Tensor
        Intensity profile in the Fourier plane
    source_intensity : torch.Tensor
        Intensity distribution before the optical setup(in the image plane)
    forward : _type_
        Function which describes forward propagation through the optical system
    reverse : _type_
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
    torch.Tensor
        Phase profile from 0 to 2pi
    """
    source_amplitude = torch.sqrt(source_intensity)
    target_amplitude = torch.sqrt(target_intensity)

    incident_field = source_amplitude * torch.exp(
        1j * initial_approximation
    )

    number_of_iterations = 0
    current_error = 10000.

    y_nodes, x_nodes = source_amplitude.shape
    number_of_pixels = x_nodes * y_nodes

    support_constrain = (
        source_amplitude > torch.max(source_amplitude) / 5
    ).float()

    while True:

        output_field = forward(incident_field)

        if (target_phase is not None) and (target_region is not None):

            # TODO: ask about surface
            current_phase_target = torch.angle(output_field)
            current_phase_target = current_phase_target + (
                2 * torch.pi
            ) * (current_phase_target < 0).float()

            # TODO: check product
            target_phase_distribution = target_phase * target_region + (
                1. - target_region
            ) * current_phase_target

            target_field = target_amplitude * torch.exp(
                1j * target_phase_distribution
            ) / torch.abs(output_field)
        else:

            target_field = target_amplitude * output_field / torch.abs(
                output_field
            )

        source_field = reverse(target_field)

        current_source_intensity = torch.pow(torch.abs(source_field), 2)

        error = torch.sqrt(
            torch.sum(
                current_source_intensity * (1. - support_constrain)
            ) / (
                torch.sum(current_source_intensity * support_constrain) * number_of_pixels  # noqa: E501
            )
        )

        # print(error)

        if (torch.abs(current_error - error) <= tol) or (
            number_of_iterations >= maxiter
        ):

            phase_function = torch.angle(incident_field)
            phase_function = phase_function + (
                2 * torch.pi
            ) * (phase_function < 0.).float()

            break

        else:

            current_phase_source = torch.angle(source_field)
            current_phase_source = current_phase_source + (
                2 * torch.pi
            ) * (current_phase_source < 0.).float()

            incident_field = source_field * support_constrain + (
                1. - support_constrain
            ) * (incident_field - constant_factor * source_field)

            number_of_iterations += 1
            current_error = error

    return phase_function
