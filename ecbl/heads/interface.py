import torch
from typing import Protocol
from ..interface import Rules, Concepts, ConceptsProbas


class ConceptsHead(Protocol):
    def __init__(self, in_features: int,
                 concepts: Concepts,
                 rules: Rules,
                 *args, **kwargs):
        """Concepts Head.

        Args:
            in_features: Input embedding size.
            concepts: Mapping from concept names to their cardinalities (numbers of outcomes).
            rules: List of concept rules.

        """

    def forward(self, x: torch.Tensor) -> ConceptsProbas:
        ...


