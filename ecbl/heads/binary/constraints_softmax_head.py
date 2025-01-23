import numpy as np
import torch
from itertools import product
from typing import Mapping
from sympy import Symbol, Expr, And, Or, Not, to_cnf
from collections import defaultdict
import logging
from .interface import BinaryConcepts, BinaryConceptsHead, Rules, BinaryConceptsProbas
from ...constraints.polytope_conversions import linear_inequalities_to_vertex_repr
from ...constraints.vertex_layer import VertexConstraintLayer
from ...utils import log_duration
from .constraints import make_sparse_A


def make_A_b_binary(cnf: Expr, concepts_enumeration: Mapping[Symbol, int]):
    """
    Returns:
        System (A, b) such that constraints are `A x >= b`.
    """
    sparse_A, _parts = make_sparse_A(cnf, concepts_enumeration)
    n_concepts = len(concepts_enumeration)
    A = np.zeros((len(sparse_A) + 2 * n_concepts, n_concepts), dtype=np.float32)
    b = np.ones((A.shape[0],), dtype=np.float32)
    for i, cols in sparse_A.items():
        for c, v in cols.items():
            A[i, c] = v
            if v == -1:
                b[i] -= 1

    ge_0_slice = slice(len(sparse_A), len(sparse_A) + n_concepts)
    np.fill_diagonal(A[ge_0_slice], 1.0)
    b[ge_0_slice] = 0.0

    le_1_slice = slice(len(sparse_A) + n_concepts, None)
    np.fill_diagonal(A[le_1_slice], -1.0)
    b[le_1_slice] = -1.0

    return A, b


class ConstraintsSoftmaxHead(torch.nn.Module):
    """Constraints-based Softmax Head.

    Args:
        in_features: Input embedding size.
        concepts: Mapping from concept names to their cardinalities (numbers of outcomes).
        rules_cnf: Rules in CNF.

    """
    def __init__(self, in_features: int,
                 concepts: BinaryConcepts,
                 rules_cnf: Expr):
        super().__init__()
        self.concepts_order = concepts
        self.concepts_enumeration = {
            s: i
            for i, s in enumerate(self.concepts_order)
        }
        self.in_features = in_features
        self.rules_cnf = rules_cnf
        self.outcomes_shape = (2,) * len(self.concepts_order)
        self._process_rules()
        self.linear = torch.nn.Linear(self.in_features, self.c_layer.n_inputs)

    def _process_rules(self):
        logger = logging.getLogger('_process_rules')
        A, b = make_A_b_binary(self.rules_cnf, self.concepts_enumeration)
        with log_duration('constraints_to_vertex_repr', 'Finding polytope vertices', logger=logger):
            vertices, rays = linear_inequalities_to_vertex_repr(-A, -b)
        if len(rays) > 0:
            raise RuntimeError(f'The polytope should be closed, but it is open with #rays = {len(rays)} > 0')
        logger.info(f'Number of vertices: {len(vertices)}')
        vertices = torch.tensor(vertices)
        self.c_layer = VertexConstraintLayer(vertices)

    def forward(self, x: torch.Tensor) -> BinaryConceptsProbas:
        return self.c_layer(self.linear(x))

