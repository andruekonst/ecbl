import torch
from typing import Mapping
from sympy import Symbol, Expr, And, Or, Not, to_cnf
from collections import defaultdict


def make_sparse_A(cnf: Expr, concepts_enumeration: Mapping[Symbol, int]):
    if isinstance(cnf, And):
        parts = cnf.args
    elif isinstance(cnf, Or):
        parts = [cnf]
    else:
        raise ValueError(f'Wrong {cnf = !r}')

    sparse_A = defaultdict(dict)

    for i, part in enumerate(parts):
        if isinstance(part, Or):
            for s_ns in part.args:
                assert isinstance(s_ns, Symbol) or isinstance(s_ns, Not)
                if isinstance(s_ns, Symbol):
                    sparse_A[i][concepts_enumeration[str(s_ns)]] = 1
                elif isinstance(s_ns, Not):
                    symb, = s_ns.args
                    assert isinstance(symb, Symbol)
                    sparse_A[i][concepts_enumeration[str(symb)]] = -1
                else:
                    raise ValueError(s_ns)
        elif isinstance(part, Symbol):
            sparse_A[i][concepts_enumeration[str(part)]] = 1
        elif isinstance(part, Not):
            symb, = part.args
            assert isinstance(symb, Symbol)
            sparse_A[i][concepts_enumeration[str(symb)]] = -1
        else:
            raise ValueError(f'Wrong part of CNF: {part}')

    sparse_A = dict(sparse_A)
    return sparse_A, parts


def find_connected_components(sparse_A):
    """
    Returns:
        Tuple (
            Map: components -> list of columns (concepts),
            Map: column (concept) -> component,
            Map: concepts -> list of row ids
        )
    """
    n = 0
    mapping = dict()
    for cols in sparse_A.values():
        cur_comp_id = None
        for c in cols.keys():
            if c in mapping:
                cur_comp_id = mapping[c]
                break
        else:
            cur_comp_id = n
            n += 1
        assert cur_comp_id is not None
        for c in cols.keys():
            mapping[c] = cur_comp_id
    components = defaultdict(list)
    for c, comp in sorted(mapping.items(), key=lambda kv: kv[0]):
        components[comp].append(c)

    connected_components_rows = defaultdict(list)
    for row, cols in sparse_A.items():
        for c in cols:
            connected_components_rows[mapping[c]].append(row)
            break
    return dict(components), mapping, dict(connected_components_rows)
