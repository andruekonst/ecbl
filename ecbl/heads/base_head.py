import torch
from functools import reduce
from operator import mul
from .interface import Concepts, Rules, ConceptsProbas
from .utils import prepare_admissible_states_mask



class BaseHead(torch.nn.Module):
    """The neural network head that infers joint concept probability distribution.

    Args:
        in_features: Input embedding size.
        concepts: Mapping from concept names to number of concept outcomes.

    """
    def __init__(self, in_features: int, concepts: Concepts, rules: Rules, eps: float=1.e-20):
        super().__init__()
        self.concepts = concepts
        self.concepts_order = list(concepts.keys())
        self.outcomes_shape = list(concepts.values())
        self.in_features = in_features
        self.out_features = reduce(mul, self.outcomes_shape, 1)
        self.linear = torch.nn.Linear(self.in_features, self.out_features)
        self.eps = eps
        self.mask = prepare_admissible_states_mask(concepts, rules)

    def forward_intermediate(self, embeddings: torch.Tensor) -> torch.Tensor:
        prior_distribution = torch.softmax(self.linear(embeddings), dim=1)
        mask = self.mask.unsqueeze(0)  # ones correspond to correct states
        # posterior P(C = c | rules(C) = True):
        posterior_distribution = mask * prior_distribution
        posterior_distribution = posterior_distribution / torch.clamp(
            posterior_distribution.sum(1),
            min=self.eps
        ).unsqueeze(1)
        return posterior_distribution

    def _reshape_posterior(self, posterior_distribution: torch.Tensor) -> torch.Tensor:
        return posterior_distribution.view(-1, *self.outcomes_shape)

    def to_marginal_proba(self, posterior_distribution: torch.Tensor,
                          concept_name: str) -> torch.Tensor:
        posterior = self._reshape_posterior(posterior_distribution)
        other_dims = tuple(
            i + 1  # shifted by batch dimension
            for i, c in enumerate(self.concepts_order)
            if c != concept_name
        )
        return posterior.sum(dim=other_dims)

    def forward_intermediate(self, embeddings: torch.Tensor) -> ConceptsProbas:
        posterior = self._reshape_posterior(self.forward_intermediate(embeddings))
        return {
            c: posterior.sum(dim=tuple(j + 1 for j in range(len(self.concepts_order)) if j != i))
            for i, c in enumerate(self.concepts_order)
        }
