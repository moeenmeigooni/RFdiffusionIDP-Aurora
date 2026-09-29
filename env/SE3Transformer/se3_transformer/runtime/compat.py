"""CUDA-profiling compatibility shims for portable inference."""

from contextlib import contextmanager


@contextmanager
def nvtx_range(_name):
    """No-op replacement for CUDA NVTX ranges on XPU and CPU."""
    yield
