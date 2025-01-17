from itertools import product
import torch
from sympy import Symbol, Expr, And, Or, Not, to_cnf
from collections import defaultdict
from .interface import BinaryConcepts, BinaryConceptsHead, Rules, BinaryConceptsProbas


def binary_prepare_admissible_states_mask(concepts: BinaryConcepts, rule: Expr) -> torch.Tensor:
    """Prepare admissible states mask.

    Enumerates through all possible states and generates the mask of admissible states.

    Args:
        concepts: Concepts.
        rules: Rules.

    Returns:
        Mask tensor of shape `(n_1 * ... * n_m)`, where `n_i` is the i-th concept cardinality.
        A mask entry is True if the state is admissible.

    """
    size = 2 ** len(concepts)
    mask = torch.ones(size, dtype=torch.bool)
    variables = [Symbol(c_name) for c_name in concepts]
    for i, values in enumerate(product(*((False, True) for n in range(len(concepts))))):
        mask[i] = bool(rule.subs(zip(variables, values)))
    return mask


class BinaryAdmissibleStatesHead(torch.nn.Module):
    """Admissible States Head for binary concepts.

    Evaluates rules at each combination of concept values.

    Args:
        in_features: Input embedding size.
        concepts: Mapping from concept names to their cardinalities (numbers of outcomes).
        rules: List of concept rules.

    """
    def __init__(self, in_features: int,
                 concepts: BinaryConcepts,
                 rule: Expr):
        super().__init__()
        self.concepts_order = concepts
        self.concepts_enumeration = {
            s: i
            for i, s in enumerate(self.concepts_order)
        }
        self.in_features = in_features
        self.rule = rule
        self.outcomes_shape = (2,) * len(self.concepts_order)
        self._process_rules()
        n_valid_states = self.placement_matrix.shape[0]
        self.linear = torch.nn.Linear(self.in_features, n_valid_states)

    def _process_rules(self):
        self.mask = binary_prepare_admissible_states_mask(self.concepts_order, self.rule)
        valid_states = torch.argwhere(self.mask).squeeze(1)
        n_valid_states = valid_states.size(0)
        n_total_states = 2 ** len(self.concepts_order)
        placement_matrix = torch.zeros((n_valid_states, n_total_states), dtype=torch.float)
        placement_matrix[torch.arange(n_valid_states), valid_states] = 1.
        # self.placement_matrix = placement_matrix
        self.register_buffer('placement_matrix', placement_matrix)

    def forward(self, x: torch.Tensor) -> BinaryConceptsProbas:
        valid_states_distribution = torch.softmax(self.linear(x), dim=1)
        total_states_distribution = valid_states_distribution @ self.placement_matrix
        cube = total_states_distribution.view(-1, *self.outcomes_shape)
        rest_dims = list(range(1, len(self.outcomes_shape)))
        result = torch.stack([
            cube[(slice(None),) * (i + 1) + (1,)].sum(dim=rest_dims)
            for i in range(len(self.outcomes_shape))
        ], dim=1)
        return result

