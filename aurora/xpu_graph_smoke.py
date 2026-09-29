"""Compute-node checks for RFdiffusion's XPU backend and native graph ops."""

from __future__ import annotations

import torch

from rfdiffusion.aurora_runtime import select_device
from se3_transformer.model.graph import copy_e_sum, edge_softmax, graph


def main() -> None:
    device = select_device()
    assert device.type == "xpu", f"Aurora smoke expected XPU, got {device}"
    print(f"torch={torch.__version__}")
    print(f"device={device}")
    print(f"properties={torch.xpu.get_device_properties(device)}")

    lhs = torch.ones((256, 256), device=device)
    result = (lhs @ lhs).sum()
    torch.xpu.synchronize(device)
    assert result.item() == 16_777_216.0
    print(f"torch_xpu_matmul={result.item()}")

    native_graph = graph(
        (
            torch.tensor([0, 2, 1, 2], device=device),
            torch.tensor([1, 1, 0, 0], device=device),
        ),
        num_nodes=3,
    ).to(device)
    scores = torch.tensor([[0.0], [1.0], [0.0], [1.0]], device=device)
    probabilities = edge_softmax(native_graph, scores)
    expected_probabilities = torch.tensor(
        [[0.26894143], [0.7310586], [0.26894143], [0.7310586]], device=device
    )
    assert torch.allclose(probabilities, expected_probabilities, atol=1e-6)
    reduced = copy_e_sum(native_graph, torch.arange(4, device=device, dtype=torch.float32)[:, None])
    assert torch.equal(reduced.cpu(), torch.tensor([[5.0], [1.0], [0.0]]))
    torch.xpu.synchronize(device)
    print("native_torch_graph_ops=passed")

    from rfdiffusion.inference.model_runners import Sampler  # noqa: F401

    print("rfdiffusion_xpu_import=passed")


if __name__ == "__main__":
    main()
