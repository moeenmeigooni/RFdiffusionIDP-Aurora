"""Minimal native-Torch directed graph operations used by RFdiffusion inference.

The upstream SE(3)-Transformer uses DGL's CUDA-only graph backend. RFdiffusion
only needs directed edge construction, destination-normalized edge softmax, and
destination sums at inference, all of which map directly to PyTorch ATen ops on
CUDA, XPU, and CPU.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


class Graph:
    def __init__(self, edges: Tuple[torch.Tensor, torch.Tensor], num_nodes: int):
        src, dst = edges
        if src.shape != dst.shape:
            raise ValueError("source and destination edge tensors must have equal shape")
        self._src = src.long()
        self._dst = dst.long()
        self._num_nodes = int(num_nodes)
        self.edata: Dict[str, torch.Tensor] = {}

    def edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._src, self._dst

    def num_nodes(self) -> int:
        return self._num_nodes

    def to(self, device: torch.device | str) -> "Graph":
        self._src = self._src.to(device)
        self._dst = self._dst.to(device)
        self.edata = {name: value.to(device) for name, value in self.edata.items()}
        return self


def graph(edges: Tuple[torch.Tensor, torch.Tensor], num_nodes: int) -> Graph:
    return Graph(edges, num_nodes)


def copy_e_sum(graph: Graph, edge_features: torch.Tensor) -> torch.Tensor:
    """Sum edge features into their destination nodes (DGL ``copy_e_sum``)."""
    out = torch.zeros(
        (graph.num_nodes(),) + tuple(edge_features.shape[1:]),
        dtype=edge_features.dtype,
        device=edge_features.device,
    )
    return out.index_add_(0, graph._dst, edge_features)


def e_dot_v(graph: Graph, edge_features: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
    """Dot each edge feature with the feature at its destination node."""
    return (edge_features * node_features[graph._dst]).sum(dim=-1, keepdim=True)


def edge_softmax(graph: Graph, scores: torch.Tensor) -> torch.Tensor:
    """Destination-normalized edge softmax, equivalent to DGL ``edge_softmax``."""
    if scores.ndim != 2:
        raise ValueError(f"expected (edges, heads) scores, received {tuple(scores.shape)}")
    dst = graph._dst[:, None].expand_as(scores)
    max_per_dst = torch.full(
        (graph.num_nodes(), scores.shape[1]),
        -torch.inf,
        dtype=scores.dtype,
        device=scores.device,
    )
    max_per_dst.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
    shifted = (scores - max_per_dst[graph._dst]).exp()
    normalizer = torch.zeros_like(max_per_dst)
    normalizer.scatter_add_(0, dst, shifted)
    return shifted / normalizer[graph._dst]
