import torch
import numpy as np
from typing import Any, Literal, Optional, Union, Callable, Protocol
import logging
import warnings

from .interface import Concepts, Rules, ConceptsProbas
from .utils import prepare_admissible_states_mask
from ..constraints.conversions import (
    make_all_constraints,
    make_all_constraints_without_eq,
    reconstruct_solution_from_constr_without_eq,
)
from ..constraints.vertex_layer import VertexConstraintLayer
from ..constraints.polytope_conversions import linear_inequalities_to_vertex_repr
from ..utils import log_duration

try:
    from constrainet import (
        LinearConstraints,
        ConstraiNetLayer,
    )
    from constrainet.layers.fused.lc_qc_diameter import ZeroNaNGradientsFn
except:
    warnings.warn('Cannot import constrainet, layer_type="constrainet" will not work')


def _check_constrainet():
    if 'ConstraiNetLayer' not in globals():
        raise RuntimeError('constrainet is not installed. Please, install it to use ConstraiNet')  # ImportError (?)


class CustomConstraintLayer(Protocol):
    @property
    def n_inputs(self) -> int:
        """Number of layer inputs.
        """


LayerType = Union[Literal['vertex', 'constrainet'], Callable[[np.ndarray, np.ndarray], CustomConstraintLayer]]


def make_constraint_layer(concepts: Concepts, rules: Rules, layer_type, params: dict):
    logger = logging.getLogger('make_constraint_layer')
    with log_duration('make_all_constraints_without_eq', 'Converting to CNF, preparing_ineq, ...', logger=logger):
        A, b = make_all_constraints_without_eq(concepts, rules)

    if isinstance(layer_type, Callable):
        c_layer = layer_type(-A, -b, **params)
    elif isinstance(layer_type, str) and layer_type.lower() == 'vertex':
        with log_duration('constraints_to_vertex_repr', 'Finding polytope vertices', logger=logger):
            vertices, rays = linear_inequalities_to_vertex_repr(-A, -b)
        if len(rays) > 0:
            raise RuntimeError(f'The polytope should be closed, but it is open with #rays = {len(rays)} > 0')
        vertices = torch.tensor(vertices)
        c_layer = VertexConstraintLayer(vertices, **params)
    elif isinstance(layer_type, str) and layer_type.lower() == 'constrainet':
        _check_constrainet()
        c_layer = ConstraiNetLayer(
            constraints=[
                LinearConstraints(
                    A=-A,
                    b=-b,
                ),
            ],
            **params
        )
    else:
        raise ValueError(f'Wrong constraint layer type: {layer_type!r}')
    return c_layer, c_layer.n_inputs


class ConstraintsHead(torch.nn.Module):
    """Constraint-based concepts head.

    The implementation incorporates both Vertex Head and more general Constraints Head.

    Args:
        in_features: Input embedding size.
        concepts: Mapping from concept names to number of concept outcomes.
        rules: List of rules.
        layer_type: Constraint layer type,

            - 'vertex' is for Vertex Head, it finds all vertices and interpolates between them;
            - 'constrainet' uses separate ConstraiNet library that can generate points in polytopes;
            - Custom constraint layer type.

        params: Custom parameter dictionary for the constraint layer, e.g. for ConstraiNet.

    """
    def __init__(self, in_features: int, concepts: Concepts, rules: Rules,
                 layer_type: LayerType = 'constrainet',
                 params: Optional[dict] = None):
        super().__init__()
        self.concepts = concepts
        self.concepts_order = list(concepts.keys())
        self.outcomes_shape = list(concepts.values())
        self.shifts = torch.cumsum(torch.tensor([0] + self.outcomes_shape), 0)
        if params is None:
            params = dict()
        self.c_layer, c_layer_n_inputs = make_constraint_layer(concepts, rules, layer_type, params)
        self.in_features = in_features
        self.linear = torch.nn.Linear(self.in_features, c_layer_n_inputs)

    def proba_vector_to_named_outputs(self, proba_vector):
        return {
            name: proba_vector[:, self.shifts[i]:self.shifts[i + 1]]
            for i, name in enumerate(self.concepts_order)
        }

    def forward(self, embeddings: torch.Tensor):
        intermediate_probas = self.c_layer(ZeroNaNGradientsFn.apply(self.linear(embeddings)))
        concat_probas = reconstruct_solution_from_constr_without_eq(
            intermediate_probas,
            self.concepts
        )
        return self.proba_vector_to_named_outputs(concat_probas)
