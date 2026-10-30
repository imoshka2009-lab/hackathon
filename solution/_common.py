"""Small helpers shared by the patches."""

import torch


def on_gpu(*tensors):
    """True when every argument is a real CUDA tensor. Every patch checks this first and falls back
    to the original code otherwise (CPU, fake modules in Soup's regression tests, meta tensors)."""
    return all(isinstance(t, torch.Tensor) and t.is_cuda for t in tensors)


def autocast_dtype():
    """The dtype a matmul runs in right now: the autocast dtype if autocast is on, else None."""
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return None


def round_mode(dtype):
    """0: keep fp32, 1: round through fp16, 2: round through bf16 (a constexpr for the kernels)."""
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    return 0
