import numpy as np
import torch

from typing import List, Mapping, Optional, Type, Union, OrderedDict
from collections import defaultdict

from sympy.logic.boolalg import to_dnf
from sympy.core.numbers import Number
from sympy import symbols, Symbol, Expr, Eq
import sympy
from typing import Protocol


Concepts = OrderedDict[str, int]
Rules = List[Expr]
ConceptsProbas = Mapping[str, torch.Tensor]
ConceptsProbasNumpy = Mapping[str, np.ndarray]
