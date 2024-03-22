import torch


class VertexConstraintLayer(torch.nn.Module):
    def __init__(self, vertices: torch.Tensor):
        """Vertex-based (softmax) layer with output satisfying constraints.

        The output solution is inside a convex polytope over the given vertices.

        Args:
            vertices: Polytope vertices.

        """
        super().__init__()
        self.register_buffer('vertices', vertices)

    @property
    def n_inputs(self) -> int:
        return len(self.vertices)

    def forward(self, x):
        """Maps the given points in barycentric coordinates into points inside the polytope.
        """
        return torch.softmax(x, dim=1) @ self.vertices
