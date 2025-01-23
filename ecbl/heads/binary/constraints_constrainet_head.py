import numpy as np
import torch
from itertools import product
from typing import Mapping
from sympy import Symbol, Expr, And, Or, Not, to_cnf
from collections import defaultdict
import logging
from .interface import BinaryConcepts, Rules, BinaryConceptsProbas
from ...constraints.polytope_conversions import linear_inequalities_to_vertex_repr
from ...utils import log_duration
from .constraints import make_sparse_A
from .constraints_softmax_head import make_A_b_binary
from constrainet import LinearConstraints, ConstraiNetLayer


class ConstraintsConstraiNetHead(torch.nn.Module):
    """Constraints-based Head with ConstraiNet.

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
        # self.intermediate = lambda x: x
        self.intermediate = torch.nn.Sigmoid()  # this is very important for faster convergence

    def _process_rules(self):
        A, b = make_A_b_binary(self.rules_cnf, self.concepts_enumeration)
        # self.c_layer = ConstraiNetLayer(constraints=[LinearConstraints(A=-A, b=-b)], mode='ray_shift')
        self.c_layer = ConstraiNetLayer(constraints=[LinearConstraints(A=-A, b=-b)], mode='center_projection')

    def forward(self, x: torch.Tensor) -> BinaryConceptsProbas:
        return self.c_layer(self.intermediate(self.linear(x)))

