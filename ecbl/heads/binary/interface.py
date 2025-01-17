import torch
import numpy as np
from typing import List, Protocol
from ..interface import Rules, Concepts, ConceptsProbas


BinaryConcepts = List[str]
BinaryConceptsProbas = torch.Tensor


class BinaryConceptsHead(Protocol):
    def __init__(self, in_features: int,
                 concepts: BinaryConcepts,
                 rules: Rules,
                 *args, **kwargs):
        """Concepts Head.

        Args:
            in_features: Input embedding size.
            concepts: Mapping from concept names to their cardinalities (numbers of outcomes).
            rules: List of concept rules.

        """

    def forward(self, x: torch.Tensor) -> BinaryConceptsProbas:
        ...


