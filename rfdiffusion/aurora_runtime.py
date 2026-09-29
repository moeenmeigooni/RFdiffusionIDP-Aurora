"""Backend selection shared by RFdiffusion Aurora inference paths."""

import torch

# Aurora's framework PyTorch exposes torch.xpu natively. Do not import IPEX
# here: its TorchScript fusion pass rewrites the legacy SE(3) basis script and
# fails on integer indexing arithmetic before the model's first forward pass.
# RFdiffusion does not require IPEX-specific APIs for inference.


def select_device():
    """Return XPU first on Aurora, then CUDA, then CPU."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def accelerator_name():
    """Return a human-readable active accelerator name, or ``None``."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.xpu.get_device_name(torch.xpu.current_device())
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device())
    return None


def active_accelerator_kind():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return None
