import numpy as np
from typing import Tuple
import warnings

try:
    import cdd
except:
    warnings.warn('Cannot import cdd library, polytope conversions will not work')


def _check_cdd():
    if 'cdd' not in globals():
        raise RuntimeError('cdd lib is not installed. Please, install it to handle polytope conversions')  # ImportError (?)


def linear_inequalities_to_vertex_repr(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the polytope defined by the system `A x <= b` into the vertex representation.

    Args:
        A: left-hand side of the inequality system, of shape (n_inequalities, n_variables).
        b: right-hand side of the inequality system, of shape (n_inequalities,).

    Returns:
        A tuple of (vertices, rays), defining the polytope.
        For closed polytopes, the number of rays is 0, so an empty array will be returned.

    """
    _check_cdd()
    # A x <= b then H = [b | -A]
    h_representation = np.concatenate((b[:, np.newaxis], -A), axis=1)
    h_mat = cdd.Matrix(h_representation.tolist(), linear=False)
    h_mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(h_mat)
    v_representation = np.array(poly.get_generators()[:])
    # [t | V]
    t = v_representation[:, 0]
    V = v_representation[:, 1:]
    return V[t == 1], V[t == 0]
