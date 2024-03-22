import numpy as np
import torch
from sympy import Symbol
from itertools import product
from .interface import Concepts, Rules


def prepare_admissible_states_mask(concepts: Concepts, rules: Rules) -> torch.Tensor:
    """Prepare admissible states mask.

    Enumerates through all possible states and generates the mask of admissible states.

    Args:
        concepts: Concepts.
        rules: Rules.

    Returns:
        Mask tensor of shape `(n_1 * ... * n_m)`, where `n_i` is the i-th concept cardinality.
        A mask entry is True if the state is admissible.

    """
    cardinalities = tuple(concepts.values())
    mask = torch.ones(np.prod(cardinalities), dtype=torch.bool)
    variables = [Symbol(c_name) for c_name in concepts.keys()]
    for i, values in enumerate(product(*(range(n) for n in cardinalities))):
        mask[i] = all(
            bool(r.subs(zip(variables, values)))
            for r in rules
        )
    return mask
