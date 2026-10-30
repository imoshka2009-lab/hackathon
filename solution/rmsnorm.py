"""Kernel 1: Qwen2RMSNorm, forward + backward in one Triton pass each.

Original (transformers 5.16.1, modeling_qwen2.py):
    input_dtype = x.dtype
    x = x.to(float32)
    x = x * rsqrt(mean(x^2) + eps)
    return weight * x.to(input_dtype)

In this task the residual stream is fp32 (prepare_model_for_kbit_training upcasts embeddings and
norms to fp32, bitsandbytes casts its output back to the input dtype), so `.to(input_dtype)` is
usually a no-op. The kernel rounds only when the input really is fp16/bf16 (ROUND constexpr).

Math:  r = 1/sqrt(mean(x^2)+eps);  y = w * round(x*r)
       g = dy*w;  dx = r * (g - x * r^2 * mean(g*x))      (weight is frozen: no dw)
"""

import torch
import triton
import triton.language as tl

from . import _common


@triton.jit
def rmsnorm_fwd_kernel(X, W, Y, RSTD, x_row_stride, y_row_stride, N, eps,
                       BLOCK: tl.constexpr, ROUND: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    tl.store(RSTD + row, rstd)  # one float per row is all the backward needs besides x
    xn = x * rstd
    if ROUND == 1:
        xn = xn.to(tl.float16).to(tl.float32)  # the original's .to(input_dtype)
    elif ROUND == 2:
        xn = xn.to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * y_row_stride + cols, (w * xn).to(Y.dtype.element_ty), mask=mask)


@triton.jit
def rmsnorm_bwd_kernel(DY, X, W, RSTD, DX, dy_row_stride, x_row_stride, dx_row_stride, N,
                       BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * dy_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row)
    g = dy * w
    dot = tl.sum(g * x, axis=0) / N
    dx = rstd * (g - x * (rstd * rstd * dot))
    tl.store(DX + row * dx_row_stride + cols, dx.to(DX.dtype.element_ty), mask=mask)


def _warps(block):
    return 8 if block >= 2048 else 4


class TritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, out_dtype):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        M, N = x2.shape
        y = torch.empty((M, N), dtype=out_dtype, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        block = triton.next_power_of_2(N)
        rmsnorm_fwd_kernel[(M,)](x2, weight, y, rstd, x2.stride(0), y.stride(0), N, float(eps),
                                 BLOCK=block, ROUND=_common.round_mode(x.dtype),
                                 num_warps=_warps(block))
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape, ctx.block = shape, block
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        M, N = x2.shape
        dy2 = dy.reshape(M, N)
        if dy2.stride(-1) != 1:
            dy2 = dy2.contiguous()
        dx = torch.empty_like(x2)
        rmsnorm_bwd_kernel[(M,)](dy2, x2, weight, rstd, dx, dy2.stride(0), x2.stride(0),
                                 dx.stride(0), N, BLOCK=ctx.block, num_warps=_warps(ctx.block))
        return dx.view(ctx.shape), None, None, None


def rmsnorm(x, weight, eps):
    """Same result and dtype as Qwen2RMSNorm.forward (weight * x_normed.to(x.dtype))."""
    out_dtype = torch.promote_types(weight.dtype, x.dtype)
    return TritonRMSNorm.apply(x, weight, eps, out_dtype)


def patch():
    from transformers.models.qwen2 import modeling_qwen2 as m

    original = m.Qwen2RMSNorm.forward
    if getattr(original, "_solution_patch", False):
        return

    def forward(self, hidden_states):
        weight = getattr(self, "weight", None)
        if not (_common.on_gpu(hidden_states, weight) and weight.dim() == 1
                and hidden_states.shape[-1] == weight.shape[0] and not weight.requires_grad
                and hidden_states.dtype in (torch.float32, torch.float16, torch.bfloat16)):
            return original(self, hidden_states)
        return rmsnorm(hidden_states, weight, self.variance_epsilon)

    forward._solution_patch = True
    m.Qwen2RMSNorm.forward = forward
