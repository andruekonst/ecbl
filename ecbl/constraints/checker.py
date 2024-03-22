"""Constraint-based rule checker.
"""
import numpy as np
from ..interface import Concepts, Rules, ConceptsProbasNumpy
from .conversions import make_all_constraints


def to_flat_probas(probas: ConceptsProbasNumpy, concepts: Concepts) -> np.ndarray:
    return np.concatenate([
        probas[c]
        for c in concepts.keys()
    ], axis=1)


class RuleChecker:
    def __init__(self, concepts: Concepts, rules: Rules, eps: float = 1.e-5) -> np.ndarray:
        self.concepts = concepts
        self.rules = rules
        self.constraints = make_all_constraints(concepts, rules)
        self.eps = eps

    def check(self, probas: ConceptsProbasNumpy) -> np.ndarray:
        """Check that the given set of marginal probability distributions satisfies the rules.
        """
        A, b = self.constraints['ineq']  # A x >= b
        flat_probas = to_flat_probas(probas, self.concepts)
        # flat_probas shape: (n_samples, n_values)
        return np.all(A @ flat_probas.T >= b[:, np.newaxis] - self.eps, axis=0)
