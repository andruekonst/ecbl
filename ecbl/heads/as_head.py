import torch
from functools import reduce
from operator import mul
from .interface import Concepts, Rules, ConceptsProbas
from .utils import prepare_admissible_states_mask


class AdmissibleStatesHead(torch.nn.Module):
    """The neural network head that infers joint concept probability distribution,
    by computing ONLY valid components of the joint concept probability distribution.

    It computes low-dimensional probability distribution (only valid states) and concerts it to
    the joint concept probability distribution (all states).

    Args:
        in_features: Input embedding size.
        concepts: Mapping from concept names to number of concept outcomes.

    """
    def __init__(self, in_features: int, concepts: Concepts, rules: Rules):
        super().__init__()
        self.concepts = concepts
        self.concepts_order = list(concepts.keys())
        self.outcomes_shape = list(concepts.values())
        self.in_features = in_features
        self.out_features = reduce(mul, self.outcomes_shape, 1)
        placement_matrix = self._prepare_placement_matrix(prepare_admissible_states_mask(concepts, rules))
        self.register_buffer('placement_matrix', placement_matrix)
        n_valid_states = self.placement_matrix.shape[0]
        self.linear = torch.nn.Linear(self.in_features, n_valid_states)
        self.eps = 1.e-20

    def _prepare_placement_matrix(self, mask):
        """Prepare the placement matrix of shape (n_valid_states, n_total_states).

        The placement matrix contains zeros and ones and can place probabilities of
        only correct states to the full join probability distribution vector.

        """
        valid_states = torch.argwhere(mask).squeeze(1)
        n_valid_states = valid_states.size(0)
        n_total_states = self.out_features
        placement_matrix = torch.zeros((n_valid_states, n_total_states), dtype=torch.float)
        placement_matrix[torch.arange(n_valid_states), valid_states] = 1.
        return placement_matrix

    def forward_intermediate(self, embeddings: torch.Tensor) -> torch.Tensor:
        valid_states_distribution = torch.softmax(self.linear(embeddings), dim=1)
        total_states_distribution = valid_states_distribution @ self.placement_matrix
        return total_states_distribution

    def _reshape_distr(self, total_states_distribution: torch.Tensor) -> torch.Tensor:
        return total_states_distribution.view(-1, *self.outcomes_shape)

    def to_marginal_proba(self, total_states_distribution: torch.Tensor,
                          concept_name: str) -> torch.Tensor:
        distr = self._reshape_distr(total_states_distribution)
        other_dims = tuple(
            i + 1  # shifted by batch dimension
            for i, c in enumerate(self.concepts_order)
            if c != concept_name
        )
        return distr.sum(dim=other_dims)

    def forward(self, embeddings: torch.Tensor) -> ConceptsProbas:
        distr = self._reshape_distr(self.forward_intermediate(embeddings))
        return {
            c: distr.sum(dim=tuple(j + 1 for j in range(len(self.concepts_order)) if j != i))
            for i, c in enumerate(self.concepts_order)
        }
