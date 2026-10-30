"""Shared bits of the tests: run on a CUDA GPU, or on the CPU with TRITON_INTERPRET=1."""

import os

import pytest
import torch

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else ("cpu" if INTERPRET else None)
needs_device = pytest.mark.skipif(DEVICE is None, reason="needs a CUDA GPU or TRITON_INTERPRET=1")
SMALL = DEVICE != "cuda"   # the interpreter is slow: smaller shapes on the CPU
TOL = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}
