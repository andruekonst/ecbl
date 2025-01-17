from typing import Mapping, Type
import torch
from sympy import Symbol, Expr, And, Or, Not, to_cnf
from collections import defaultdict
from .interface import BinaryConcepts, BinaryConceptsHead, Rules, BinaryConceptsProbas


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


class EfficientModuleList(torch.nn.Module):
    def __init__(self, module_list: torch.nn.ModuleList):
        super().__init__()
        self.module_list = module_list

    def forward(self, x):
        return [
            m(x)
            for m in self.module_list
        ]


class BinaryWrapper(torch.nn.Module):
    """Concepts Head Wrapper for a Concept Head, implements the state space reduction for binary concepts.

    Extracts connected components.

    Args:
        in_features: Input embedding size.
        concepts: Mapping from concept names to their cardinalities (numbers of outcomes).
        rules: List of concept rules.

    """
    def __init__(self, in_features: int,
                 concepts: BinaryConcepts,
                 rules_cnf: Expr,
                 head_cls: Type[BinaryConceptsHead]):
        super().__init__()
        self.head_cls = head_cls
        self.concepts_order = concepts
        self.concepts_enumeration = {
            s: i
            for i, s in enumerate(self.concepts_order)
        }
        self.in_features = in_features
        # self.rules_cnf = to_cnf(And(*self.rules))
        self.rules_cnf = rules_cnf
        self._process_rules()

    def _process_rules(self):
        self.used_concepts = set(
            str(s)
            for s in self.rules_cnf.free_symbols
        )
        self.free_concepts = set(self.concepts_order) - self.used_concepts
        self.free_concepts_ids = sorted(list(map(self.concepts_enumeration.get, self.free_concepts)))
        sparse_A, parts = make_sparse_A(self.rules_cnf, self.concepts_enumeration)
        self.components, self.mapping, self.conn_components_rows = find_connected_components(sparse_A)
        self.wrapped_heads = EfficientModuleList(torch.nn.ModuleList([
            self.head_cls(
                self.in_features,
                [self.concepts_order[cid] for cid in self.components[comp]],
                And(*[parts[r] for r in self.conn_components_rows[comp]])
            )
            for comp in range(len(self.components))
        ]))

        if len(self.free_concepts) > 0:
            self.layer_free_concepts = torch.nn.Sequential(
                       torch.nn.Linear(self.in_features, len(self.free_concepts)),
                       torch.nn.Sigmoid()  # predict probabilities
            )
        else:
            self.layer_free_concepts = None

    def forward(self, x: torch.Tensor) -> BinaryConceptsProbas:
        wrapped_heads_preds = self.wrapped_heads(x)

        result = torch.empty((x.shape[0], len(self.concepts_order)), dtype=x.dtype, device=x.device)
        for i, head_preds in enumerate(wrapped_heads_preds):
            result[:, self.components[i]] = head_preds

        if self.layer_free_concepts is not None:
            free_concepts = self.layer_free_concepts(x)
            result[:, self.free_concepts_ids] = free_concepts

        return result
