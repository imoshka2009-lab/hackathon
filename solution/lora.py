"""Kernel 4: LoRA dropout + cast and scaled add, around peft's 4-bit LoRA layer.

In this task q_proj / v_proj are `peft.tuners.lora.bnb.Linear4bit` (the model is 4-bit), not
`peft.tuners.lora.Linear`. Its forward (peft 0.20.0), under the fp16 autocast of the Trainer:
    result = base_layer(x).clone()                  # bnb: fp16 matmul, cast back to x.dtype (fp32)
    out = lora_B(lora_A(dropout(x))) * scaling       # dropout on fp32 x, autocast casts to fp16
    result = result + out                            # fp32 + fp16 -> fp32
Here:
    xd = dropout_cast(x)       one Triton pass: Philox mask from a seed (no stored mask), 1/(1-p)
                               scale and the cast to the autocast dtype; backward = same kernel,
                               same seed, mask regenerated
    lo = lora_B(lora_A(xd))    unchanged torch matmuls (LoRA weight grads stay exactly autograd's)
    out = result + lo*scaling  one Triton pass, no defensive clone (the output is a new tensor)
Dropout stays 0.05 (training) and 0 in eval. The mask comes from our own Philox stream, not from
PyTorch's CUDA RNG, so the masks differ from the base run (allowed: checks compare curves within
1-3%). Set TORCH_DROPOUT_RNG = True to use torch's own dropout (identical masks) and fuse only the
cast and the add.
"""

import torch
import triton
import triton.language as tl

from . import _common

TORCH_DROPOUT_RNG = False


@triton.jit
def lora_dropout_scale_kernel(IN, OUT, N, in_row_stride, out_row_stride, seed, p, scale,
                              BLOCK_N: tl.constexpr, DROPOUT: tl.constexpr):
    """OUT[row] = IN[row] * keep * scale, keep ~ Bernoulli(1-p) from Philox(seed, flat index).
    Forward of the dropout (scale = 1/(1-p)), its backward (same seed -> same mask) and, with
    DROPOUT off, a plain scale-and-cast."""
    row = tl.program_id(0).to(tl.int64)
    for col0 in range(0, N, BLOCK_N):
        cols = col0 + tl.arange(0, BLOCK_N)
        inside = cols < N
        v = tl.load(IN + row * in_row_stride + cols, mask=inside, other=0.0).to(tl.float32)
        if DROPOUT:
            flat = (row * N + cols).to(tl.int32)
            keep = tl.rand(seed, flat) > p
            v = tl.where(keep, v * scale, 0.0)
        else:
            v = v * scale
        tl.store(OUT + row * out_row_stride + cols, v.to(OUT.dtype.element_ty), mask=inside)


@triton.jit
def lora_scaled_add_kernel(BASE, LO, OUT, N, base_row_stride, lo_row_stride, out_row_stride,
                           scaling, BLOCK_N: tl.constexpr):
    """OUT[row] = BASE[row] + LO[row] * scaling   (base fp32 from bnb, lo fp16 from autocast)."""
    row = tl.program_id(0).to(tl.int64)
    for col0 in range(0, N, BLOCK_N):
        cols = col0 + tl.arange(0, BLOCK_N)
        inside = cols < N
        b = tl.load(BASE + row * base_row_stride + cols, mask=inside, other=0.0).to(tl.float32)
        lo = tl.load(LO + row * lo_row_stride + cols, mask=inside, other=0.0).to(tl.float32)
        tl.store(OUT + row * out_row_stride + cols, (b + lo * scaling).to(OUT.dtype.element_ty),
                 mask=inside)


def _rows(t):
    t2 = t.reshape(-1, t.shape[-1])
    return t2 if t2.stride(-1) == 1 else t2.contiguous()


def _block(n):
    # One block size for q_proj (1536) and v_proj (256): one compile per kernel variant, not two.
    # Triton compile time counts in the measured wall time (caches start empty).
    return 1024


def _dropout_scale(src, out_dtype, seed, p, scale):
    s2 = _rows(src)
    out = torch.empty(s2.shape, dtype=out_dtype, device=src.device)
    M, N = s2.shape
    lora_dropout_scale_kernel[(M,)](s2, out, N, s2.stride(0), out.stride(0), int(seed), float(p),
                                    float(scale), BLOCK_N=_block(N), DROPOUT=p > 0,
                                    num_warps=4)
    return out.view(src.shape)


class DropoutCast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, seed, out_dtype):
        scale = 1.0 / (1.0 - p) if p > 0 else 1.0
        ctx.p, ctx.seed, ctx.scale, ctx.x_dtype = p, seed, scale, x.dtype
        return _dropout_scale(x, out_dtype, seed, p, scale)

    @staticmethod
    def backward(ctx, g):
        return _dropout_scale(g, ctx.x_dtype, ctx.seed, ctx.p, ctx.scale), None, None, None


class ScaledAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, lo, scaling):
        b2, l2 = _rows(base), _rows(lo)
        out_dtype = torch.promote_types(base.dtype, lo.dtype)
        out = torch.empty(b2.shape, dtype=out_dtype, device=base.device)
        M, N = b2.shape
        lora_scaled_add_kernel[(M,)](b2, l2, out, N, b2.stride(0), l2.stride(0), out.stride(0),
                                     float(scaling), BLOCK_N=_block(N), num_warps=4)
        ctx.scaling, ctx.base_dtype, ctx.lo_dtype = scaling, base.dtype, lo.dtype
        return out.view(base.shape)

    @staticmethod
    def backward(ctx, g):
        d_base = g if g.dtype == ctx.base_dtype else g.to(ctx.base_dtype)
        d_lo = _dropout_scale(g, ctx.lo_dtype, 0, 0.0, ctx.scaling)   # g * scaling, cast
        return d_base, d_lo, None


class _Seeds:
    """Our own Philox seeds: drawn from a private generator seeded from the run's torch seed, so
    the run stays reproducible and no other random stream (data order, init) is disturbed."""
    gen = None

    @classmethod
    def next(cls):
        if cls.gen is None:
            cls.gen = torch.Generator().manual_seed(torch.initial_seed() & 0xFFFF_FFFF)
        return int(torch.randint(0, 2**31 - 1, (1,), generator=cls.gen))


def _single_adapter(layer):
    if layer.disable_adapters or layer.merged:
        return None
    active = [a for a in layer.active_adapters if a in layer.lora_A.keys()]
    if len(active) != 1:
        return None
    a = active[0]
    if a in getattr(layer, "lora_variant", {}):          # DoRA and friends: original path
        return None
    lora_a, lora_b = layer.lora_A[a], layer.lora_B[a]
    if not (type(lora_a) is torch.nn.Linear and type(lora_b) is torch.nn.Linear
            and lora_a.bias is None and lora_b.bias is None):
        return None
    if not isinstance(layer.lora_dropout[a], (torch.nn.Dropout, torch.nn.Identity)):
        return None
    return a


def patch():
    try:
        from peft.tuners.lora import bnb as lora_bnb
    except ImportError:
        return
    cls = getattr(lora_bnb, "Linear4bit", None)
    if cls is None:
        return
    original = cls.forward
    if getattr(original, "_solution_patch", False):
        return

    def forward(self, x, *args, **kwargs):
        compute_dtype = _common.autocast_dtype()
        adapter = None
        if not args and not kwargs and compute_dtype is not None and _common.on_gpu(x) \
                and x.dtype in (torch.float32, torch.float16, torch.bfloat16):
            adapter = _single_adapter(self)
        if adapter is None:
            return original(self, x, *args, **kwargs)
        result = self.base_layer(x)
        drop = self.lora_dropout[adapter]
        p = float(drop.p) if isinstance(drop, torch.nn.Dropout) and drop.training else 0.0
        if TORCH_DROPOUT_RNG and p > 0:
            xd = DropoutCast.apply(drop(x), 0.0, 0, compute_dtype)
        else:
            xd = DropoutCast.apply(x, p, _Seeds.next() if p > 0 else 0, compute_dtype)
        lo = self.lora_B[adapter](self.lora_A[adapter](xd))
        return ScaledAdd.apply(result, lo, self.scaling[adapter])

    forward._solution_patch = True
    cls.forward = forward
