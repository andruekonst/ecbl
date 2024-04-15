# ECBL

A PyTorch implementation of Expert Concept-Based Learning,
described in the [paper](https://arxiv.org/abs/2402.14726).


## Idea

Assume that not only a single target $y$ is available, but a set of concepts
$(c^{(1)}, \dots, c^{(m)})$.
Each concept is like a classification target and may take value from
a discrete set: $c^{(i)} \in \mathcal{C}^{(i)} = \{ 0, \dots , n_i - 1 \}$.
So the target vector is $c = (c^{(1)}, \dots, c^{(m)})$.

Let a rule $g(c) = g(c^{(1)}, \dots, c^{(m)})$ be a mapping
$\mathcal{C}^{(1)}\times \dots \times \mathcal{C}^{(m)} \rightarrow \{False, True\}$, i.e. a boolean-valued function of contepts.

The key idea is to make a neural network layer, that can estimate
probability distributions of concepts that satisfy the given rules.

We estimate marginal probabilities of concepts corresponding to the
joint probability distrituion of all the concepts under constraint
that the given rules are satisfied: $P(C = c \vert g(C) = 1)$.
As it is described in the paper, we do not necessarily have to model
the whole joint distribution to satisfy this constraint, and
this library provide several tools for that.

## Usage Example

Lets consider concept vector $c = (y_0, y_1, y_2, y_3)$,
where
$y_0 \in \{0, 1\}, y_1 \in \{0, 1\}, y_2 \in \{0, 1, 2\}, y_3 \in \{0, 1, 2\}$.
For example, the rule can be:
$g(c) = [(y_0 = 1) \Rightarrow ((y_2 = 1) \land (y_1 = 1))]$.

Then the implementation of the layer will be:

```python
# we use SymPy to define rules
from sympy import Symbol, Eq, Implies, Equivalent

from ecbl import (
    ConceptsHeadWrapper,  # basic optimizations
    AdmissibleStatesHead,
    ConstraintsHead,
)

# specify concept cardinalities (number of values):
concepts = {'y_0': 2, 'y_1': 2, 'y_2': 3, 'y_3': 3}

# define a rule set
rules = [
    # (y_0 == 1) => ((y_2 == 1) & (y_1 == 1))
    Implies(
        Eq(Symbol('y_0'), 1),
        Eq(Symbol('y_2'), 1) & Eq(Symbol('y_1'), 1)
    ),
]

concepts_head = ConceptsHeadWrapper(
    in_features=n_in_features,      # neural network embedding size
    concepts=concepts,              # concept cardinalities
    rules=rules,                    # rules
    head_cls=AdmissibleStatesHead,  # core concept-layer class
)

model = torch.nn.Sequential(
    nn_encoder,  # some neural network encoder that maps X to an embedding
    concepts_head,
)

preds = model(X)
# preds["<concept name>"] is a probability distribution of the concept
assert preds['y_0'].shape[1] = 2
assert preds['y_3'].shape[1] = 3

# the neural network can be optimized through `concepts_head`
```

More detailed examples can be found in [notebooks](notebooks/).

## Installation

The package is under development, and can be installed from
this git repository:

```bash
pip install git+https://github.com/andruekonst/ecbl.git
```

Or clone the repo and install in development mode:

```bash
git clone https://github.com/andruekonst/ecbl.git
cd ecbl
pip install -e .
```



## Citation

All the methods are presented in the following preprint:

```bibtex
@article{konstantinov2024incorporating,
  title={Incorporating Expert Rules into Neural Networks in the Framework of Concept-Based Learning},
  author={Konstantinov, Andrei V and Utkin, Lev V},
  journal={arXiv preprint arXiv:2402.14726},
  year={2024}
}
```


