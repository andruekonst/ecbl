import numpy as np
import torch

from typing import List, Mapping, Optional, Type, Union
from collections import defaultdict

from sympy.logic.boolalg import to_dnf
from sympy.core.numbers import Number
from sympy import symbols, Symbol, Expr, Eq
import sympy
from .interface import Concepts, ConceptsProbas, Rules, ConceptsHead



def find_rule_symbol_value_leaves(rule: Expr):
    """Find leaves representing numbers (concept values).

    Args:
        rule: Rule containing boolean expressions over predicates of equality of a concept to a value.

    """
    leaves = []
    consider = [rule]
    while len(consider) > 0:
        cur = consider.pop()
        if any(isinstance(a, Number) for a in cur.args):
            # leaf found
            leaves.append(cur)
            continue
        consider.extend(cur.args)
    return leaves


def get_used_concept_values(rules: Rules):
    """Get all referenced values for each concept.

    Args:
        rules: List of rules containing boolean expressions over predicates of equality of a concept to a value.

    """
    leaves = [
        l
        for rule in rules
        for l in find_rule_symbol_value_leaves(rule)
    ]
    used_concept_values = defaultdict(list)
    for leaf in leaves:
        if isinstance(leaf.args[0], Number):
            value, concept = leaf.args
        elif isinstance(leaf.args[1], Number):
            concept, value = leaf.args
        else:
            raise RuntimeError(f'Wrong leaf: {leaf}')
        used_concept_values[str(concept)].append(int(value))
    return dict(used_concept_values)


def find_complement_concept_values(concepts: Concepts, used_concept_values: Mapping[str, List[int]]):
    """`For each concept find a list of values that are not referenced in `used_concept_values`.

    Each concept is considered to take values from 0 to its cardinality (excluding).

    Args:
        concepts: Concept cardinalities.
        used_concept_values: Mapping from concept name to a list of used values.

    """
    return {
        c: list(set(range(n_values)) - set(used_concept_values.get(c, [])))
        for c, n_values in concepts.items()
    }


def get_inverse_permutation(ids: Union[List[int], np.ndarray]) -> List[int]:
    """Get inverse permutation of the given permutation.

    Args:
        ids: Input permutation.

    """
    return np.argsort(ids).tolist()


class EfficientModuleDict(torch.nn.Module):
    def __init__(self, module_dict: torch.nn.ModuleDict):
        super().__init__()
        self.module_dict = module_dict

    def forward(self, x):
        return [
            m(x)
            for m in self.module_dict.values()
        ]


class ConceptsHeadWrapper(torch.nn.Module):
    """Concepts Head Wrapper for a Concept Head, implements the state space reduction.

    Extracts a minimal subset of concepts and outcomes and applies
    the underlying concept head only to them.
    The rest concepts (and outcomes) are modeled separately, by different heads
    without rules.

    Args:
        in_features: Input embedding size.
        concepts: Mapping from concept names to their cardinalities (numbers of outcomes).
        rules: List of concept rules.

    """
    def __init__(self, in_features: int,
                 concepts: Concepts,
                 rules: Rules,
                 head_cls: Type[ConceptsHead]):
        super().__init__()
        self.head_cls = head_cls
        self.concepts = concepts
        self.concepts_order = list(concepts.keys())
        self.outcomes_shape = list(concepts.values())
        self.in_features = in_features
        self._process_rules(rules)

    def _process_rules(self, rules):
        self.used_concepts = set(
            s
            for r in rules
            for s in r.free_symbols
        )
        self.unused_concepts = set(self.concepts_order) - self.used_concepts
        self.used_concept_values = get_used_concept_values(rules)
        self.used_concept_complement_values = find_complement_concept_values(
            self.concepts,
            self.used_concept_values
        )
        self.free_concepts = (
            set(self.used_concept_complement_values.keys()) - set(self.used_concept_values.keys())
        )  # concepts that do not present in the joint distribution
        self.semifree_concepts = set(
            c for c, v in self.used_concept_complement_values.items()
            if len(v) >= 1 and c not in self.free_concepts
        )
        self.semifree_concepts_need_complement = set(
            c for c, v in self.used_concept_complement_values.items()
            if len(v) > 1 and c not in self.free_concepts
        )
        self.only_joint_concepts = set(
            c for c, v in self.used_concept_complement_values.items()
            if len(v) == 0
        )

        joint_outcomes_shape = []
        joint_outcomes_order = []
        joint_outcomes_values = []
        for c in self.concepts_order:
            if c in self.free_concepts:
                continue
            if c in self.only_joint_concepts:
                c_n_outcomes = self.concepts[c]
                c_outcomes_values = list(range(self.concepts[c]))
            elif c in self.semifree_concepts:
                # one extra value correspond to the rest values that are not directly mentioned in rules
                c_n_outcomes = len(self.used_concept_values[c]) + 1
                c_outcomes_values = [-1] + sorted(self.used_concept_values[c])
            else:
                raise RuntimeError(f'Unexpected concept: {c!r} not in only joint, semifree or free')
            joint_outcomes_shape.append(c_n_outcomes)
            joint_outcomes_order.append(c)
            joint_outcomes_values.append(c_outcomes_values)

        self.joint_outcomes_order = joint_outcomes_order
        self.joint_outcomes_values = joint_outcomes_values
        self.joint_outcomes = dict(zip(self.joint_outcomes_order, self.joint_outcomes_values))

        if len(self.free_concepts) > 0:
            self.layers_free_concepts = torch.nn.ModuleDict({
                c: torch.nn.Sequential(
                       torch.nn.Linear(self.in_features, self.concepts[c]),
                       torch.nn.Softmax(dim=-1)
                   )
                for c in self.free_concepts
            })
            self.layers_free_concepts = torch.jit.script(EfficientModuleDict(self.layers_free_concepts))
        else:
            self.layers_free_concepts = None

        if len(self.semifree_concepts_need_complement) > 0:
            self.layers_semifree_concepts = torch.nn.ModuleDict({
                c: torch.nn.Sequential(
                       torch.nn.Linear(self.in_features, len(self.used_concept_complement_values[c])),
                       torch.nn.Softmax(dim=-1)
                   )
                for c in self.semifree_concepts_need_complement
            })
            self.layers_semifree_concepts = torch.jit.script(EfficientModuleDict(self.layers_semifree_concepts))
        else:
            self.layers_semifree_concepts = None

        self.joint_outcomes_shape = joint_outcomes_shape
        self.wrapped = self.head_cls(
            self.in_features,
            dict(zip(self.joint_outcomes_order, self.joint_outcomes_shape)),
            rules,
        )

    def forward_intermediate(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.layers_semifree_concepts is not None:
            semifree_concepts_need_complement = self.layers_semifree_concepts(embeddings)
        else:
            semifree_concepts_need_complement = None
        if self.layers_free_concepts is not None:
            free_concepts = self.layers_free_concepts(embeddings)
        else:
            free_concepts = None
        return semifree_concepts_need_complement, free_concepts

    def concept_probas(self, embeddings: torch.Tensor) -> Mapping[str, torch.Tensor]:
        semifree_complement, free_concepts = self.forward_intermediate(embeddings)
        joint_distr_marginal = self.wrapped.forward(embeddings)
        # joint_distr_marginal: concept name -> (n_samples, n_marginal_outcomes)

        # First, we should fix the order of variables in semifree_concepts which
        # have no complementary values.
        # For example, if rules used concept value 0 of a binary concept,
        #     the rest value (which is 1) is encoded as -1, and the order is not correct.

        final_marginal = dict()
        for c in joint_distr_marginal.keys():
            compl_values = self.used_concept_complement_values[c]
            if len(compl_values) == 1:
                compl_value = compl_values[0]
                assert self.joint_outcomes[c][0] == -1
                # feature order at `joint_distr_marginal[c]`
                current_feature_order = [compl_value] + self.joint_outcomes[c][1:]
                # lets make the order right: [0, ..., n] by applying inverse permutation on feature ids
                feature_ids = get_inverse_permutation(current_feature_order)
                final_marginal[c] = joint_distr_marginal[c][:, feature_ids]
                # now `join_distr_marginal[c]` has the desired feature order [0, ..., n]
                # print(f"joint_distr_marginal==1 feature {c}, {feature_ids=}, {current_feature_order=}")

        # For each semifree concept which has more than one complementary value.
        #     For example when a concept has three values: [0, 1, 2], and rules
        #     use only concept value 1.
        #     Then complementary values are [0, 2], we got their probabilities at
        #     `semifree_complement`.
        #     The `joint_distr_marginal` has values [-1, 1], where -1 acts as a placeholder
        #     the complementary concept values, which are [0, 2] in this case.
        #     By using value -1 here we guarantee that the placeholder always goes before other values.
        #     To obtain corrected marginal distribution (with all the original classes),
        #     we need to multiply `semifree_complement` by `joint_distr_marginal` at -1,
        #     and emplace into the final `marginal`.
        #     We can just concatenate multiplied entries with `joint_distr_marginal` without the first
        #     feature, corresponding to -1 class (since we already took it into account).
        #     Then we will get `[0, 2, 1]` class order, and the final marginal probability vector
        #     will be `intermediary_marginal[get_inverse_permutation([0, 2, 1])]`

        if semifree_complement is not None and len(semifree_complement) > 0:
            semifree_complement = dict(zip(self.semifree_concepts_need_complement, semifree_complement))
        else:
            semifree_complement = dict()

        for c in joint_distr_marginal.keys():
            compl_values = self.used_concept_complement_values[c]
            if len(compl_values) == 0:
                final_marginal[c] = joint_distr_marginal[c]
            elif len(compl_values) > 1:
                assert self.joint_outcomes[c][0] == -1
                # replace -1 with complementary values
                current_feature_order = compl_values + self.joint_outcomes[c][1:]
                # calculate probabilities of complementary outcomes and replace with them
                # probability of -1 (placeholder)
                cur_joint_marginal = joint_distr_marginal[c]
                combined_marginal = torch.concat((
                    semifree_complement[c] * cur_joint_marginal[:, 0].unsqueeze(1),
                    cur_joint_marginal[:, 1:]
                ), dim=1)

                feature_ids = get_inverse_permutation(current_feature_order)
                final_marginal[c] = combined_marginal[:, feature_ids]
                # print(f"joint_distr_marginal feature {c}, {feature_ids=}, {current_feature_order=}")
                # print("semifree_complement[c]:", semifree_complement[c].shape)

        if free_concepts is not None:
            free_distr = dict(zip(self.free_concepts, free_concepts))
        else:
            free_distr = dict()

        return {
            **final_marginal,
            **free_distr
        }

    def forward(self, x) -> ConceptsProbas:
        return self.concept_probas(x)

