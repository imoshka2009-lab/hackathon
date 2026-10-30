"""Kernel 2: rotary position embedding for q and k in one launch (forward and backward).

Original (modeling_qwen2.apply_rotary_pos_emb, unsqueeze_dim=1):
    q_out = q*cos + rotate_half(q)*sin,   rotate_half(x) = cat(-x2, x1)
With halves (x1, x2), (c1, c2), (s1, s2):
    forward:  o1 = x1*c1 - x2*s1          o2 = x2*c2 + x1*s2
    backward: d1 = g1*c1 + g2*s2          d2 = g2*c2 - g1*s1
Qwen builds cos/sin as cat(freqs, freqs), so c1 == c2 and s1 == s2 and the backward is the forward
with sin negated (the guide's shortcut). The kernel uses the general form, which costs nothing and
stays right if the two halves ever differ.

q and k come in as (B, H, S, D) views of (B, S, H, D) buffers: the kernel takes real strides and
writes outputs with the same layout as the inputs (empty_like keeps the strides), like the original
element-wise ops do. One program per (batch, position) handles every q head and every k head.
"""

import torch
import triton
import triton.language as tl

from . import _common


@triton.jit
def rope_qk_kernel(Q, K, QO, KO, COS, SIN, S, HQ, HK,
                   sq_b, sq_h, sq_s, sk_b, sk_h, sk_s,
                   sqo_b, sqo_h, sqo_s, sko_b, sko_h, sko_s,
                   sc_b, sc_s,
                   HALF: tl.constexpr, BLOCK_HQ: tl.constexpr, BLOCK_HK: tl.constexpr,
                   BACKWARD: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // S
    s = pid % S
    dims = tl.arange(0, HALF)[None, :]
    c_base = COS + b * sc_b + s * sc_s
    s_base = SIN + b * sc_b + s * sc_s
    c1 = tl.load(c_base + dims).to(tl.float32)
    c2 = tl.load(c_base + HALF + dims).to(tl.float32)
    s1 = tl.load(s_base + dims).to(tl.float32)
    s2 = tl.load(s_base + HALF + dims).to(tl.float32)
    if BACKWARD:  # transpose of the rotation: d1 = g1*c1 + g2*s2, d2 = g2*c2 - g1*s1
        s_first = s1
        s1 = -s2
        s2 = -s_first
    # q heads: all of them in one 2-D tile (BLOCK_HQ x HALF)
    hq = tl.arange(0, BLOCK_HQ)[:, None]
    mq = hq < HQ
    pq = Q + b * sq_b + s * sq_s + hq * sq_h + dims
    pqo = QO + b * sqo_b + s * sqo_s + hq * sqo_h + dims
    q1 = tl.load(pq, mask=mq, other=0.0).to(tl.float32)
    q2 = tl.load(pq + HALF, mask=mq, other=0.0).to(tl.float32)
    tl.store(pqo, (q1 * c1 - q2 * s1).to(QO.dtype.element_ty), mask=mq)
    tl.store(pqo + HALF, (q2 * c2 + q1 * s2).to(QO.dtype.element_ty), mask=mq)
    # k heads (2 for Qwen2.5-1.5B), same rotation
    hk = tl.arange(0, BLOCK_HK)[:, None]
    mk = hk < HK
    pk = K + b * sk_b + s * sk_s + hk * sk_h + dims
    pko = KO + b * sko_b + s * sko_s + hk * sko_h + dims
    k1 = tl.load(pk, mask=mk, other=0.0).to(tl.float32)
    k2 = tl.load(pk + HALF, mask=mk, other=0.0).to(tl.float32)
    tl.store(pko, (k1 * c1 - k2 * s1).to(KO.dtype.element_ty), mask=mk)
    tl.store(pko + HALF, (k2 * c2 + k1 * s2).to(KO.dtype.element_ty), mask=mk)


def _launch(q, k, cos, sin, q_out, k_out, backward):
    B, HQ, S, D = q.shape
    HK = k.shape[1]
    rope_qk_kernel[(B * S,)](
        q, k, q_out, k_out, cos, sin, S, HQ, HK,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
        q_out.stride(0), q_out.stride(1), q_out.stride(2),
        k_out.stride(0), k_out.stride(1), k_out.stride(2),
        cos.stride(0), cos.stride(1),
        HALF=D // 2, BLOCK_HQ=triton.next_power_of_2(HQ), BLOCK_HK=triton.next_power_of_2(HK),
        BACKWARD=backward, num_warps=4)


def _unit_last(t):
    return t if t.stride(-1) == 1 else t.contiguous()


class TritonRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        B, _, S, _ = q.shape
        q, k = _unit_last(q), _unit_last(k)
        # (B or 1, S, D) -> (B, S, D) view with stride 0 on a broadcast batch
        cos = _unit_last(cos).expand(B, S, -1)
        sin = _unit_last(sin).expand(B, S, -1)
        q_out = torch.empty_like(q, dtype=torch.promote_types(q.dtype, cos.dtype))
        k_out = torch.empty_like(k, dtype=torch.promote_types(k.dtype, cos.dtype))
        _launch(q, k, cos, sin, q_out, k_out, backward=False)
        ctx.save_for_backward(cos, sin)  # no activations: the backward only needs cos/sin
        ctx.q_dtype, ctx.k_dtype = q.dtype, k.dtype
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq_out, dk_out):
        cos, sin = ctx.saved_tensors
        dq_out, dk_out = _unit_last(dq_out), _unit_last(dk_out)
        dq = torch.empty_like(dq_out, dtype=ctx.q_dtype)
        dk = torch.empty_like(dk_out, dtype=ctx.k_dtype)
        _launch(dq_out, dk_out, cos, sin, dq, dk, backward=True)
        return dq, dk, None, None


def _supported(q, k, cos, sin, unsqueeze_dim):
    if not _common.on_gpu(q, k, cos, sin) or unsqueeze_dim != 1:
        return False
    if q.dim() != 4 or k.dim() != 4 or cos.dim() != 3 or sin.shape != cos.shape:
        return False
    B, _, S, D = q.shape
    if k.shape[0] != B or k.shape[2] != S or k.shape[3] != D or D % 2 or (D & (D - 1)):
        return False
    if cos.shape[0] not in (1, B) or cos.shape[1] != S or cos.shape[2] != D:
        return False
    return not (cos.requires_grad or sin.requires_grad)


def patch():
    from transformers.models.qwen2 import modeling_qwen2 as m

    original = m.apply_rotary_pos_emb
    if getattr(original, "_solution_patch", False):
        return

    def apply_rotary_pos_emb(q, k, cos, sin, *args, **kwargs):
        unsqueeze_dim = kwargs.get("unsqueeze_dim", args[0] if args else 1)
        if len(args) > 1 or set(kwargs) - {"unsqueeze_dim"} or \
                not _supported(q, k, cos, sin, unsqueeze_dim):
            return original(q, k, cos, sin, *args, **kwargs)
        return TritonRoPE.apply(q, k, cos, sin)

    apply_rotary_pos_emb._solution_patch = True
    m.apply_rotary_pos_emb = apply_rotary_pos_emb
