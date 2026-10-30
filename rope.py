"""Rotary position embedding of q and k, forward and backward, in Triton.

Base: `q * cos + rotate_half(q) * sin` is five PyTorch kernels per tensor (two slices into a cat,
a negation, two products, a sum) in forward and as many in backward, 28 layers, q and k. Here one
program takes one (batch, position) and every head of it: it reads cos and sin once and writes
the rotated rows. Backward is the same rotation transposed:
dx1 = dy1 * cos1 + dy2 * sin2, dx2 = dy2 * cos2 - dy1 * sin1.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, seq_len, n_heads,
                sxb, sxh, sxt, sxd, sob, soh, sot, sod, scb, sct, scd,
                HALF: tl.constexpr, BLOCK_H: tl.constexpr, BACKWARD: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // seq_len
    t = pid % seq_len
    heads = tl.arange(0, BLOCK_H)[:, None]
    dims = tl.arange(0, HALF)[None, :]
    mask = heads < n_heads
    c = cos_ptr + b * scb + t * sct
    s = sin_ptr + b * scb + t * sct
    cos1 = tl.load(c + dims * scd).to(tl.float32)
    cos2 = tl.load(c + (dims + HALF) * scd).to(tl.float32)
    sin1 = tl.load(s + dims * scd).to(tl.float32)
    sin2 = tl.load(s + (dims + HALF) * scd).to(tl.float32)
    xp = x_ptr + b * sxb + t * sxt + heads * sxh
    x1 = tl.load(xp + dims * sxd, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(xp + (dims + HALF) * sxd, mask=mask, other=0.0).to(tl.float32)
    if BACKWARD:
        y1 = x1 * cos1 + x2 * sin2
        y2 = x2 * cos2 - x1 * sin1
    else:
        y1 = x1 * cos1 - x2 * sin1
        y2 = x2 * cos2 + x1 * sin2
    op = out_ptr + b * sob + t * sot + heads * soh
    tl.store(op + dims * sod, y1.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(op + (dims + HALF) * sod, y2.to(out_ptr.dtype.element_ty), mask=mask)


def _rotate(x, cos, sin, backward):
    """x [batch, heads, seq, head_dim], cos/sin [batch or 1, seq, head_dim]."""
    out = torch.empty_like(x)  # keeps x's strides, as the base elementwise ops do
    batch, heads, seq, dim = x.shape
    rope_kernel[(batch * seq,)](
        x, cos, sin, out, seq, heads,
        *x.stride(), *out.stride(),
        0 if cos.shape[0] == 1 else cos.stride(0), cos.stride(1), cos.stride(2),  # batch 1 broadcasts
        HALF=dim // 2, BLOCK_H=triton.next_power_of_2(heads), BACKWARD=backward,
        num_warps=4)
    return out


class RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        ctx.save_for_backward(cos, sin)
        return _rotate(q, cos, sin, False), _rotate(k, cos, sin, False)

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        dq = None if dq is None else _rotate(dq, cos, sin, True)
        dk = None if dk is None else _rotate(dk, cos, sin, True)
        return dq, dk, None, None


def _fits(q, k, cos, sin):
    if not (q.is_cuda and q.dim() == 4 and k.dim() == 4 and cos.dim() == 3):
        return False
    dim = q.shape[-1]
    if dim % 2 or (dim // 2) & (dim // 2 - 1) or k.shape[-1] != dim:  # half must be a power of 2
        return False
    if cos.shape != sin.shape or cos.shape[-1] != dim or cos.shape[1] != q.shape[2]:
        return False
    if cos.shape[0] not in (1, q.shape[0]) or k.shape[0] != q.shape[0] or k.shape[2] != q.shape[2]:
        return False
    if cos.requires_grad or sin.requires_grad or cos.stride() != sin.stride():
        return False
    dtypes = (torch.float32, torch.float16)
    if q.dtype not in dtypes or k.dtype not in dtypes or cos.dtype not in dtypes:
        return False
    # the result dtype of `q * cos`
    return torch.promote_types(q.dtype, cos.dtype) == q.dtype and \
        torch.promote_types(k.dtype, cos.dtype) == k.dtype


def patch_qwen2_rope():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.apply_rotary_pos_emb
    if getattr(original, "_solution", False):
        return

    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        if unsqueeze_dim != 1 or not _fits(q, k, cos, sin):
            return original(q, k, cos, sin, unsqueeze_dim)
        return RoPEFunction.apply(q, k, cos, sin)

    apply_rotary_pos_emb._solution = True
    modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb
