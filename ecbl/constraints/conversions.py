import torch
import numpy as np
from sympy.logic.boolalg import to_cnf
from sympy import Symbol, Eq, And, Or, Ne, Number
from ..interface import Concepts, Rules


def expand_eq_ne(eq_ne, concepts):
    assert isinstance(eq_ne, Eq) or isinstance(eq_ne, Ne)
    if isinstance(eq_ne, Ne):
        if isinstance(eq_ne.args[0], Symbol):
            symb, value = eq_ne.args
        elif isinstance(eq_ne.args[1], Symbol):
            value, symb = eq_ne.args
        else:
            raise ValueError(f'No symbol found in Neq: {eq_ne.args}')
        assert isinstance(value, Number)
        value = int(value)
        return Or(*[Eq(symb, other) for other in range(concepts[str(symb)]) if other != value])
    return eq_ne


def expand_ne_in_cnf(cnf, concepts):
    """Expand all not-equals in CNF to obtain rules on all outcomes.

    For example, given a concept `x` with values `{0, 1, 2}`,
    the rule `x != 1` will be expanded into `x = 0 or x = 2`.

    """
    if isinstance(cnf, And):
        to_consider = cnf.args
    elif isinstance(cnf, Or):  # there is only one disjuction part in CNF
        to_consider = [cnf]
    else:
        raise ValueError(f'Wrong {cnf = !r}')

    parts = []
    for p in to_consider:
        if isinstance(p, Or):
            parts.append(Or(*[expand_eq_ne(eq_ne, concepts) for eq_ne in p.args]))
        elif isinstance(p, Eq) or isinstance(p, Ne):
            parts.append(expand_eq_ne(p, concepts))
        else:
            raise ValueError(f'Element of CNF is neither disjunction (Or), nor Eq or Ne: {p!r}')
    return And(*parts)


def cnf_to_constraints(cnf, concepts, concept_proba_index_lookup=None):
    """Generate left-hand side of constraints system: `A x >= b`.
    The right-hand side is a unit vector `(1, ..., 1)`.

    """
    if concept_proba_index_lookup is None:
        concept_proba_index_lookup = dict()
        for name, size in concepts.items():
            for i in range(size):
                concept_proba_index_lookup[(name, i)] = len(concept_proba_index_lookup)
    cnf = expand_ne_in_cnf(cnf, concepts)
    # cnf = part_1 & ... & part_n
    # part_i = h_i1 | ... | h_ik
    if isinstance(cnf, And):
        parts = cnf.args
    elif isinstance(cnf, Or):
        parts = [cnf]
    else:
        raise ValueError(f'Wrong {cnf = !r}')

    A = np.zeros((len(parts), len(concept_proba_index_lookup)))
    for i, part in enumerate(parts):
        if isinstance(part, Or):
            for eq in part.args:
                assert isinstance(eq, Eq)
                symb, value = eq.args
                A[i, concept_proba_index_lookup[(str(symb), int(value))]] = 1
        elif isinstance(part, Eq):
            symb, value = part.args
            A[i, concept_proba_index_lookup[(str(symb), int(value))]] = 1
        else:
            raise ValueError(f'Wrong part of CNF: {part}')
    return A


def make_proba_distribution_constraints(concepts):
    """Make probability distribution constraints: A x == b,
    such that each concept outcomes probabilities will sum up to 1.

    """
    shifts = np.cumsum([0] + list(concepts.values()))
    total_outcomes = shifts[-1]
    A = np.zeros((len(concepts), total_outcomes))
    for i in range(len(shifts) - 1):
        A[i, shifts[i]:shifts[i + 1]] = 1
    return A


def make_all_constraints(concepts: Concepts, rules: Rules):
    general_rule = And(*rules)
    cnf = to_cnf(general_rule)
    expanded_cnf = expand_ne_in_cnf(cnf, concepts)
    cnf_constraints_A = cnf_to_constraints(expanded_cnf, concepts)
    cnf_constraints_b = np.ones_like(cnf_constraints_A[:, 0])
    # NOTE: A x >= b
    n_total = cnf_constraints_A.shape[1]  # total number of outcomes
    all_ineq_A = np.concatenate((cnf_constraints_A, np.eye(n_total, n_total)), axis=0)
    all_ineq_b = np.concatenate((cnf_constraints_b, np.zeros((n_total,))), axis=0)

    proba_distr_constraints_A = make_proba_distribution_constraints(concepts)
    proba_distr_constraints_b = np.ones_like(proba_distr_constraints_A[:, 0])
    # A x == b

    return {
        'ineq': (all_ineq_A, all_ineq_b),  # A x >= b
        'eq': (proba_distr_constraints_A, proba_distr_constraints_b)  # A x == b
    }


def make_all_constraints_without_eq(concepts, rules):
    """Make constraints of reduced dimension with eliminated equality constraints.
    For each concept we remove the first (0-th) outcome probability and replace it with `1 - sum(rest)`.
    The full collection of marginal probability distributions can be reconstructed by
    inserting `1 - sum(rest)` to each vector.
    We replace system `p_1 + ... + p_n == 1 and p_i >= 0` with `p_2 + ... + p_n <= 1 and p_i >= 0`.

    """
    general_rule = And(*rules)
    cnf = to_cnf(general_rule)
    expanded_cnf = expand_ne_in_cnf(cnf, concepts)
    cnf_constraints_A = cnf_to_constraints(expanded_cnf, concepts)
    cnf_constraints_b = np.ones_like(cnf_constraints_A[:, 0])
    # NOTE: A x >= b
    n_total = cnf_constraints_A.shape[1]  # total number of outcomes
    all_ineq_A = np.concatenate((cnf_constraints_A, np.eye(n_total, n_total)), axis=0)
    all_ineq_b = np.concatenate((cnf_constraints_b, np.zeros((n_total,))), axis=0)

    # now for each concept eliminate the first (0-th) outcome probability, update A and b
    shifts = np.cumsum([0] + list(concepts.values()))
    new_A = all_ineq_A.copy()
    new_b = all_ineq_b.copy()
    for i in range(len(shifts) - 1):
        col_to_subtract = all_ineq_A[:, shifts[i]]  # the first (0-th) column
        new_A[:, shifts[i]:shifts[i + 1]] -= col_to_subtract[:, np.newaxis]
        new_b -= col_to_subtract

    # then we can remove zero columns
    final_A = np.concatenate([
        new_A[:, (shifts[i] + 1):shifts[i + 1]]
        for i in range(len(shifts) - 1)
    ], axis=1)
    final_b = new_b
    return final_A, final_b


def reconstruct_solution_from_constr_without_eq(solution, concepts):
    """Reconstruct full solution from the solution of reduced dimension,
    obtained after solving system `A z >= b`,
    where `A, b = make_all_constraints_without_eq(...)`.

    Implementation: prepend to each outcomes group (concept) probability `1 - sum(probas)`.

    Returns:
        Vector of concatenated marginal probabilities of each concept.

    """
    shifts_in_compressed = np.cumsum([0] + [v - 1 for v in concepts.values()])
    result = []
    for i in range(len(shifts_in_compressed) - 1):
        interval = slice(shifts_in_compressed[i], shifts_in_compressed[i + 1])
        result.append(1. - solution[:, interval].sum(1)[:, None])
        result.append(solution[:, interval])
    if isinstance(result, np.ndarray):
        return np.concatenate(result, axis=1)
    return torch.concat(result, dim=1)
