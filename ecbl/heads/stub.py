import torch
import numpy as np
from typing import Any, Literal, Optional, Union, Callable, Protocol
import logging
import warnings
from .interface import Concepts, Rules, ConceptsProbas
from ..utils import log_duration


class StubHead(torch.nn.Module):
    """Stub concepts head (does not apply constraints at all).

    Args:
        in_features: Input embedding size.
        concepts: Mapping from concept names to number of concept outcomes.
        rules: List of rules.

    """
    def __init__(self, in_features: int, concepts: Concepts, rules: Rules):
        super().__init__()
        self.concepts = concepts
        self.concepts_order = list(concepts.keys())
        self.outcomes_shape = list(concepts.values())
        self.shifts = torch.cumsum(torch.tensor([0] + self.outcomes_shape), 0)
        self.in_features = in_features
        n_total_outcomes = np.sum(self.outcomes_shape)
        self.linear = torch.nn.Linear(self.in_features, n_total_outcomes)

    def forward(self, embeddings: torch.Tensor):
        intermediate_logits = self.linear(embeddings)
        return {
            name: torch.softmax(intermediate_logits[:, self.shifts[i]:self.shifts[i + 1]], dim=1)
            for i, name in enumerate(self.concepts_order)
        }
