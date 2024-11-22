from dataclasses import dataclass
import torch


# TODO: ask for message and status code
@dataclass(frozen=True, slots=True)
class PhaseRetrievalResult:
    """Represents the phase retrieval result
    """

    solution: torch.Tensor
    cost_func: float
    cost_func_evolution: list
    number_of_iterations: int

    # TODO: create metrics
    # intesity_distribution: torch.Tensor
    # target_region: torch.Tensor

    # def get_efficiency(self):

    #     return torch.sum(self._intensity_distribution * self._target_region)

    # def get_power_relation(self):

    #     power_relation = torch.sum(
    #         self._intensity_distribution * self._target_region
    #     ) / torch.sum(self._intensity_distribution)

    #     return power_relation
