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
    inv_mapping = defaultdict(set)

    for cols in sparse_A.values():
        cur_comp_ids = []
        for c in cols.keys():
            if c in mapping:
                cur_comp_ids.append(mapping[c])
        if len(cur_comp_ids) == 0:
            cur_comp_ids = [n]
            n += 1
        cur_comp_id = min(cur_comp_ids)
        for c in cols.keys():
            mapping[c] = cur_comp_id
            inv_mapping[cur_comp_id].add(c)
        for comp in cur_comp_ids[1:]:
            for c in inv_mapping[comp]:
                mapping[c] = cur_comp_id
    components = defaultdict(list)
    for c, comp in sorted(mapping.items(), key=lambda kv: kv[0]):
        components[comp].append(c)

    connected_components_rows = defaultdict(list)
    for row, cols in sparse_A.items():
        cur_comp = None
        for c in cols:
            if cur_comp is None:
                cur_comp = mapping[c]
            if cur_comp != mapping[c]:
                raise ValueError(f'{cur_comp=}, {mapping[c]=}, {c=}')
        connected_components_rows[cur_comp].append(row)
    return dict(components), mapping, dict(connected_components_rows)
