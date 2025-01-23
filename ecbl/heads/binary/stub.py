from itertools import product
import torch
from sympy import Symbol, Expr, And, Or, Not, to_cnf
from collections import defaultdict
from .interface import BinaryConcepts, BinaryConceptsHead, Rules, BinaryConceptsProbas


class BinaryStubHead(torch.nn.Module):
    def __init__(self, in_features: int,
                 concepts: BinaryConcepts,
                 rule: Expr):
        super().__init__()
        self.concepts = concepts
        self.outcomes_shape = (2,) * len(concepts)
        self.linear = torch.nn.Linear(in_features, 2 ** len(concepts))
        s = 1 / torch.sqrt(torch.tensor(max(self.linear.weight.shape)))
        self.linear.weight.data.uniform_(-s, s)
        self.linear.bias.data.fill_(0.0)
        # torch.nn.init.kaiming_uniform_(
        #     self.linear.weight.data,
        #     a=0,
        #     mode='fan_out',
        #     nonlinearity='relu',
        # )
        # self.linear.bias.data.fill_(0.0)

        # self.sum_layer = torch.nn.Linear(2 ** len(concepts), len(concepts))
        # s = 1 / torch.sqrt(torch.tensor(max(self.linear.weight.shape)))
        # self.sum_layer.weight.data.uniform_(-s, s)
        # self.sum_layer.bias.data.fill_(0.0)
        # self.stub_layer = torch.nn.Linear(in_features, len(concepts))
        # print(len(concepts), concepts)

    def forward(self, x: torch.Tensor) -> BinaryConceptsProbas:
        total_states_distribution = torch.softmax(self.linear(x), dim=1)
        cube = total_states_distribution.view(x.shape[0], *self.outcomes_shape)
        rest_dims = list(range(1, len(self.outcomes_shape)))
        result = torch.stack([
            cube[(slice(None),) * (i + 1) + (1,)].sum(dim=rest_dims)
            for i in range(len(self.outcomes_shape))
        ], dim=1)
        return result

        # return torch.sigmoid(self.sum_layer(self.linear(x)))
        # return torch.sigmoid(self.stub_layer(x))

