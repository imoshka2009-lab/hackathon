"""Our own NF4 dequantization in Triton, and the 4-bit linear layers that use it.

What bitsandbytes does in a training step of this task: for every 4-bit layer (7 per decoder
layer, 196 in total) it rebuilds the fp16 weight from the packed NF4 nibbles and the double-quantized
block scales, once in forward and once in backward, then calls cuBLAS. Here that rebuild is one
Triton kernel per layer call: it reads the packed bytes and the 8-bit scales and writes the fp16
weight; the scale dequantization (code2[q] * absmax2 + offset) is folded into the same kernel instead
of a separate pass. The math is the one of bitsandbytes' dequantize_4bit, so the weights come out
the same.

Second change, in the MLP only: base Soup casts every 4-bit layer output back to the input dtype,
which is fp32 here (prepare_model_for_kbit_training keeps the norms in fp32, so the MLP input is
fp32). gate_proj and up_proj then give fp32 tensors of 2048 x 8960 that are kept for backward. The
values are fp16 values anyway (the matmul runs in fp16), and the next consumer (down_proj) casts
back to fp16, so we keep them fp16: the same numbers, half the activation memory.
"""

import torch
import triton
import triton.language as tl

DEQUANT_BLOCK = 1024  # packed bytes per program -> 2048 weights


@triton.jit
def nf4_dequant_kernel(packed_ptr, absmax_ptr, code2_ptr, absmax2_ptr, nf4_ptr, out_ptr,
                       offset, n_bytes,
                       BLOCKSIZE: tl.constexpr, BLOCKSIZE2: tl.constexpr,
                       NESTED: tl.constexpr, BLOCK: tl.constexpr):
    """One byte holds two weights: the high nibble is the first one. Weight i uses scale block
    i // BLOCKSIZE; with double quantization that scale is code2[absmax_q] * absmax2[block // 256]
    + offset (all fp32, like bitsandbytes)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_bytes
    byte = tl.load(packed_ptr + offs, mask=mask, other=0).to(tl.int32)
    block = (offs * 2) // BLOCKSIZE  # both weights of a byte are in the same block
    if NESTED:
        q = tl.load(absmax_ptr + block, mask=mask, other=0).to(tl.int32)
        scale = tl.load(code2_ptr + q, mask=mask, other=0.0) * \
            tl.load(absmax2_ptr + block // BLOCKSIZE2, mask=mask, other=0.0)
        scale = scale + offset
    else:
        scale = tl.load(absmax_ptr + block, mask=mask, other=0.0)
    hi = tl.load(nf4_ptr + ((byte >> 4) & 15)) * scale
    lo = tl.load(nf4_ptr + (byte & 15)) * scale
    pair = tl.interleave(hi, lo).to(out_ptr.dtype.element_ty)
    out_offs = pid * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    tl.store(out_ptr + out_offs, pair, mask=out_offs < 2 * n_bytes)


_NF4_TABLES = {}


def _nf4_table(device):
    table = _NF4_TABLES.get(device)
    if table is None:
        import bitsandbytes.functional as bnbf

        table = bnbf.get_4bit_type("nf4", device=device).float().contiguous()
        _NF4_TABLES[device] = table
    return table


def supported(quant_state):
    return (quant_state is not None and quant_state.quant_type == "nf4"
            and len(quant_state.shape) == 2 and quant_state.blocksize % 2 == 0
            and (not quant_state.nested or quant_state.state2.blocksize % 2 == 0))


def dequantize_nf4(packed, quant_state, dtype=torch.float16):
    """The weight [out_features, in_features] in `dtype`, from the packed NF4 bytes."""
    shape = tuple(quant_state.shape)
    out = torch.empty(shape, dtype=dtype, device=packed.device)
    n_bytes = out.numel() // 2
    packed = packed.reshape(-1)
    if quant_state.nested:
        absmax = quant_state.absmax.contiguous()  # uint8 codes of the block scales
        code2 = quant_state.state2.code.float().contiguous()
        absmax2 = quant_state.state2.absmax.float().contiguous()
        offset = float(quant_state.offset)
        blocksize2 = quant_state.state2.blocksize
    else:
        absmax = quant_state.absmax.float().contiguous()
        code2 = absmax2 = absmax
        offset, blocksize2 = 0.0, 1
    grid = (triton.cdiv(n_bytes, DEQUANT_BLOCK),)
    nf4_dequant_kernel[grid](packed, absmax, code2, absmax2, _nf4_table(packed.device), out,
                             offset, n_bytes, BLOCKSIZE=quant_state.blocksize,
                             BLOCKSIZE2=blocksize2, NESTED=bool(quant_state.nested),
                             BLOCK=DEQUANT_BLOCK)
    return out


class NF4Linear(torch.autograd.Function):
    """y = x @ W.T + b with W rebuilt by our kernel in forward and again in backward (nothing
    dense is kept between them, as in bitsandbytes). The weight and bias are frozen."""

    @staticmethod
    def forward(ctx, x, packed, quant_state, bias):
        weight = dequantize_nf4(packed, quant_state, x.dtype)
        out = torch.nn.functional.linear(x, weight, bias)
        ctx.packed, ctx.quant_state = packed, quant_state
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = None
        if ctx.needs_input_grad[0]:
            weight = dequantize_nf4(ctx.packed, ctx.quant_state, grad_out.dtype)
            grad_x = torch.matmul(grad_out, weight)
        return grad_x, None, None, None


def _prepare(module, x):
    """The part of bitsandbytes' Linear4bit.forward before the matmul; returns (x, bias,
    input dtype), or None when this layer should take the original path."""
    from bitsandbytes.nn.modules import fix_4bit_weight_quant_state_from_module

    fix_4bit_weight_quant_state_from_module(module)
    quant_state = module.weight.quant_state
    if (not x.is_cuda or not supported(quant_state) or module.weight.requires_grad
            or (module.bias is not None and module.bias.requires_grad)):
        return None
    if not module.compute_type_is_set:
        module.set_compute_type(x)
        module.compute_type_is_set = True
    inp_dtype = x.dtype
    if module.compute_dtype is not None:
        x = x.to(module.compute_dtype)
    if x.dtype not in (torch.float16, torch.float32):
        return None
    bias = module.bias
    if bias is not None:
        if bias.dtype != x.dtype:
            bias.data = bias.data.to(x.dtype)  # what bitsandbytes does too
        bias = bias.to(module.compute_dtype)
    return x, bias, inp_dtype


def linear4bit(module, x, keep_compute_dtype=False):
    """bitsandbytes' Linear4bit.forward with our dequant kernel. With keep_compute_dtype the
    output stays in the compute dtype (fp16) instead of being cast back to the input dtype."""
    prepared = _prepare(module, x)
    if prepared is None:
        return _ORIGINAL["forward"](module, x)
    x, bias, inp_dtype = prepared
    out = NF4Linear.apply(x, module.weight, module.weight.quant_state, bias)
    return out if keep_compute_dtype else out.to(inp_dtype)


_ORIGINAL = {}


def patch_linear4bit():
    import bitsandbytes.nn.modules as bnb_modules

    cls = bnb_modules.Linear4bit
    if "forward" in _ORIGINAL:
        return
    _ORIGINAL["forward"] = cls.forward

    def forward(self, x):
        return linear4bit(self, x)

    cls.forward = forward


def project(layer, x):
    """A projection of the MLP that keeps its output in fp16 when it is a plain 4-bit layer.
    Anything else (a LoRA wrapper, a plain nn.Linear) runs as it is."""
    import bitsandbytes.nn.modules as bnb_modules

    if type(layer) is bnb_modules.Linear4bit or type(layer) is bnb_modules.LinearNF4:
        return linear4bit(layer, x, keep_compute_dtype=True)
    return layer(x)
