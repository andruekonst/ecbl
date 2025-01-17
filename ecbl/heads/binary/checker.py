"""Constraint-based rule checker for binary concepts.
"""
import numpy as np
from scipy import sparse
from .wrapper import make_sparse_A, Expr, BinaryConcepts
from .interface import BinaryConceptsProbasNumpy


class BinaryRuleChecker:
    def __init__(self, concepts: BinaryConcepts, rules_cnf: Expr, eps: float = 1.e-5) -> np.ndarray:
        self.concepts_order = concepts
        self.concepts_enumeration = {
            s: i
            for i, s in enumerate(self.concepts_order)
        }
        self.rules_cnf = rules_cnf
        sparse_A, _parts = make_sparse_A(rules_cnf, self.concepts_enumeration)
        rows = []
        cols = []
        data = []
        b = []
        for row_id, row_data in sparse_A.items():
            b_elem = 1
            for col_id, d in row_data.items():
                rows.append(row_id)
                cols.append(col_id)
                data.append(d)
                if d == -1:
                    b_elem -= 1
            b.append(b_elem)
        self.sparse_A = sparse.coo_array((data, (rows, cols)), shape=((len(sparse_A), len(self.concepts_enumeration))), dtype=np.float32)
        self.b = np.array(b)
        self.eps = eps

    def check(self, probas: BinaryConceptsProbasNumpy) -> np.ndarray:
        """Check that the given set of marginal probability distributions satisfies the rules.
        """
        # probas shape: (n_samples, n_values)
        return np.all(self.sparse_A @ probas.T >= self.b[:, np.newaxis] - self.eps, axis=0)
