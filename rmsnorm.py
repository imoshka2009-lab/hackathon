"""Qwen2RMSNorm, forward and backward, one Triton program per row.

Base Soup runs each of the 57 RMSNorms of the model (two per decoder layer and the final one) as
separate PyTorch ops in fp32: cast, pow, mean, add, rsqrt, mul, cast, mul by the weight, and the
same chain again in backward. Here forward reads the row once and writes the output and 1/rms;
backward reads the row and the incoming gradient once and writes the input gradient. The norm
weights are frozen in this task (LoRA trains q_proj and v_proj only), so no weight gradient is
computed; a norm with a trainable weight takes the original path.

The math follows Qwen2RMSNorm.forward exactly: x in fp32, x * rsqrt(mean(x^2) + eps), cast to the
input dtype, times the weight; the output dtype is the promotion of the weight and input dtypes.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, out_ptr, rstd_ptr, n_cols, eps,
                       ROUND_TO_INPUT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.math.rsqrt(var + eps)
    y = x * rstd
    if ROUND_TO_INPUT:  # `hidden_states.to(input_dtype)` when the input is not fp32
        y = y.to(x_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * n_cols + cols, (w * y).to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def rmsnorm_bwd_kernel(grad_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, n_cols,
                       ROUND_TO_INPUT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    dy = tl.load(grad_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    g = dy * w
    if ROUND_TO_INPUT:  # the gradient passes the cast to the input dtype
        g = g.to(x_ptr.dtype.element_ty).to(tl.float32)
    x_hat = x * rstd
    dot = tl.sum(g * x_hat, axis=0) / n_cols
    dx = rstd * (g - x_hat * dot)
    tl.store(dx_ptr + row * n_cols + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        n_rows, n_cols = x2.shape
        out_dtype = torch.promote_types(weight.dtype, x.dtype)
        out = torch.empty((n_rows, n_cols), dtype=out_dtype, device=x.device)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        block = triton.next_power_of_2(n_cols)
        round_to_input = x.dtype != torch.float32
        warps = 8 if block >= 2048 else 4
        rmsnorm_fwd_kernel[(n_rows,)](x2, weight, out, rstd, n_cols, eps,
                                      ROUND_TO_INPUT=round_to_input, BLOCK=block,
                                      num_warps=warps)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape, ctx.block, ctx.round_to_input, ctx.warps = shape, block, round_to_input, warps
        return out.view(shape)

    @staticmethod
    def backward(ctx, grad):
        x2, weight, rstd = ctx.saved_tensors
        n_rows, n_cols = x2.shape
        grad2 = grad.reshape(n_rows, n_cols).contiguous()
        dx = torch.empty_like(x2)
        rmsnorm_bwd_kernel[(n_rows,)](grad2, x2, weight, rstd, dx, n_cols,
                                      ROUND_TO_INPUT=ctx.round_to_input, BLOCK=ctx.block,
                                      num_warps=ctx.warps)
        return dx.view(ctx.shape), None, None


def rmsnorm(x, weight, eps):
    return RMSNormFunction.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    if getattr(cls.forward, "_solution", False):
        return
    original = cls.forward

    def forward(self, hidden_states):
        w = self.weight
        if (not hidden_states.is_cuda or w.requires_grad or hidden_states.shape[-1] > 8192
                or hidden_states.dtype not in (torch.float32, torch.float16)
                or w.dtype not in (torch.float32, torch.float16)):
            return original(self, hidden_states)
        return rmsnorm(hidden_states, w, float(self.variance_epsilon))

    forward._solution = True
    cls.forward = forward
