from typing import Optional
import torch
from .interface import Concepts, Rules, ConceptsProbas


class IndependentConceptsHead(torch.nn.Module):
    """The neural network head that infers separate (independent)
    concept probability distributions.

    Distinct linear heads are used for different concepts.

    THIS MODULE IS IMPLEMENTED ONLY AS A COMPETITOR FOR COMPARISONS.

    Args:
        in_features: Input embedding size.
        concepts: Mapping from concept names to number of concept outcomes.
        rules: Rules are ignored.

    """
    def __init__(self, in_features: int, concepts: Concepts, rules: Optional[Rules] = None):
        super().__init__()
        self.concepts = concepts
        self.concepts_order = list(concepts.keys())
        self.outcomes_shape = list(concepts.values())
        self.in_features = in_features
        self.linears = torch.nn.ModuleDict({
            concept_name: torch.nn.Linear(self.in_features, n_outcomes)
            for concept_name, n_outcomes in concepts.items()
        })

    def forward(self, embeddings: torch.Tensor) -> ConceptsProbas:
        return {
            concept_name: torch.softmax(linear(embeddings), dim=1)
            for concept_name, linear in self.linears.items()
        }
