from typing import Literal, overload, Protocol, TypeAlias, TypedDict

import torch

from svetlanna import LinearOpticalSetup
from svetlanna.phase_retrieval_problem import algorithms
from svetlanna.phase_retrieval_problem import phase_retrieval_result as prr


class SetupLike(Protocol):
    """A class for phase_retrieval_problem with personal realizations of
    forward and reverse methods instead of methods in
    svetlanna.setup.LinearOpticalSetup

    Parameters
    ----------
    Protocol : _type_
        _description_
    """
    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        ...

    def reverse(self, transmission_field: torch.Tensor) -> torch.Tensor:
        ...


class AlgorithmOptions(TypedDict, total=False):
    tol: float
    maxiter: int
    constant_factor: float
    disp: bool


Method: TypeAlias = Literal['GS', 'HIO']


@overload
def retrieve_phase(
    source_intensity: torch.Tensor,
    optical_setup: LinearOpticalSetup | SetupLike,
    target_intensity: torch.Tensor,
    *,
    initial_phase: torch.Tensor | None = None,
    method: Method = 'GS',
    options: AlgorithmOptions | None = None
) -> prr.PhaseRetrievalResult:
    ...


@overload
def retrieve_phase(
    source_intensity: torch.Tensor,
    optical_setup: LinearOpticalSetup | SetupLike,
    target_intensity: torch.Tensor,
    target_phase: torch.Tensor,
    target_region: torch.Tensor,
    *,
    initial_phase: torch.Tensor | None = None,
    method: Method = 'GS',
    options: AlgorithmOptions | None = None
) -> prr.PhaseRetrievalResult:
    ...


def retrieve_phase(
    source_intensity: torch.Tensor,
    optical_setup: LinearOpticalSetup | SetupLike,
    target_intensity: torch.Tensor,
    target_phase: torch.Tensor | None = None,
    target_region: torch.Tensor | None = None,
    *,
    initial_phase: torch.Tensor | None = None,
    method: Method = 'GS',
    options: AlgorithmOptions | None = None
) -> prr.PhaseRetrievalResult:
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
    options : dict, optional
        Dictionary with optimization parameters, by default {
        'tol': 1e-16,   # criteria for stop optimization
        'maxiter': 100, # maximum number of iterations
        'constant_factor': 0.9,    # convergence parameter for HIO
        'disp': False   # show result of optimization
        }

    Returns
    -------
    prr.PhaseRetrievalResult
        Exemplar of class PhaseRetrievalResult which presents result of
        optimization

    Raises
    ------
    ValueError
        Unknown optimization method
    """

    forward_propagation = optical_setup.forward
    reverse_propagation = optical_setup.reverse

    if initial_phase is None:
        initial_phase = 2 * torch.pi * torch.rand_like(source_intensity)

    # read options
    if options is None:
        options = {}
    tol = options.get('tol', 1e-16)
    maxiter = options.get('maxiter', 100)
    constant_factor = options.get('constant_factor', 0.9)
    disp = options.get('disp', False)

    if method == 'GS':

        result = algorithms.gerchberg_saxton_algorithm(
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
        result = algorithms.hybrid_input_output(
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

    if disp:
        if (target_phase is not None) & (target_region is not None):
            print('Type of problem: phase reconstruction')
        else:
            print('Type of problem: generate intensity profile')
        print('Method:' + str(method))
        print('Current cost function value:' + str(result.cost_func))
        print('Number of iteration:' + str(result.number_of_iterations))

    return result
