"""Correctness of your kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`. Add a test like this for every kernel you write.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


# ---------------------------------------------------------------- our kernels


def test_nf4_dequant_matches_bitsandbytes():
    import bitsandbytes.functional as bnbf
    from solution.nf4 import dequantize_nf4

    torch.manual_seed(0)
    for shape in [(8960, 1536), (256, 1536), (96, 320)]:
        w = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.05
        for nested in (True, False):
            packed, state = bnbf.quantize_4bit(w, blocksize=64, compress_statistics=nested,
                                               quant_type="nf4")
            ref = bnbf.dequantize_4bit(packed, state)
            ours = dequantize_nf4(packed, state, torch.float16)
            torch.testing.assert_close(ours, ref, rtol=0, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rmsnorm_matches_qwen2(dtype):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(0)
    norm = Qwen2RMSNorm(1536, eps=1e-6).cuda()
    norm.weight.data = (torch.randn(1536, device="cuda") * 0.3 + 1).float()
    norm.weight.requires_grad_(False)
    x = (torch.randn(4, 300, 1536, device="cuda") * 3).to(dtype).requires_grad_(True)
    grad = torch.randn(4, 300, 1536, device="cuda")
    xf = x.float()
    ref = norm.weight * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(dtype)
    ref.backward(grad.to(ref.dtype))
    ref_dx, x.grad = x.grad.clone(), None
    out = rmsnorm(x, norm.weight, 1e-6)
    out.backward(grad.to(out.dtype))
    assert out.dtype == ref.dtype
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    tol = 2e-3 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(x.grad.float(), ref_dx.float(), rtol=tol, atol=tol)


def test_rope_matches_qwen2():
    from solution.rope import RoPEFunction

    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    torch.manual_seed(0)
    b, t, d = 4, 333, 128
    q0 = torch.randn(b, t, 12, d, device="cuda").transpose(1, 2).requires_grad_(True)
    k0 = torch.randn(b, t, 2, d, device="cuda").transpose(1, 2).requires_grad_(True)
    inv = 1.0 / (1e6 ** (torch.arange(0, d, 2, device="cuda").float() / d))
    freqs = torch.arange(t, device="cuda").float()[None, :, None] * inv
    emb = torch.cat([freqs, freqs], -1).expand(b, t, d)
    cos, sin = emb.cos(), emb.sin()
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    rq, rk = q0 * c + rotate_half(q0) * s, k0 * c + rotate_half(k0) * s
    gq, gk = torch.randn_like(rq), torch.randn_like(rk)
    ((rq * gq).sum() + (rk * gk).sum()).backward()
    q1 = q0.detach().clone().requires_grad_(True)
    k1 = k0.detach().clone().requires_grad_(True)
    oq, ok = RoPEFunction.apply(q1, k1, cos, sin)
    ((oq * gq).sum() + (ok * gk).sum()).backward()
    for a, r in ((oq, rq), (ok, rk), (q1.grad, q0.grad), (k1.grad, k0.grad)):
        torch.testing.assert_close(a, r, rtol=1e-5, atol=1e-5)


def test_fused_ce_matches_hf_loss_and_trl_metrics():
    from transformers.loss.loss_utils import ForCausalLMLoss
    from solution.fused_ce import fused_head_loss

    torch.manual_seed(0)
    b, t, h, v = 4, 300, 1536, 151936
    hidden = torch.randn(b, t, h, device="cuda").requires_grad_(True)
    weight = (torch.randn(v, h, device="cuda") * 0.02).half().float()
    labels = torch.randint(0, v, (b, t), device="cuda")
    labels[:, :40] = -100
    mask = torch.ones(b, t, dtype=torch.long, device="cuda")
    mask[1, 200:] = 0
    labels[1, 200:] = -100
    items = (labels[:, 1:] != -100).sum()

    h2 = hidden.detach().clone().requires_grad_(True)
    logits = h2.half() @ weight.half().t()
    ref = ForCausalLMLoss(logits, labels, v, num_items_in_batch=items)
    (ref * 1024).backward()
    loss, stats = fused_head_loss(hidden, weight, labels, mask, items)
    (loss * 1024).backward()
    torch.testing.assert_close(loss, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(hidden.grad, h2.grad, rtol=2e-2, atol=2e-3)
    with torch.no_grad():
        lp = torch.log_softmax(logits.float(), -1)
        ent = -(lp.exp() * lp).sum(-1)
        ref_entropy = (ent * mask).sum() / mask.sum()
        pred = logits[:, :-1].argmax(-1)
        lab = labels[:, 1:]
        ref_correct = ((pred == lab) & (lab != -100)).sum()
    torch.testing.assert_close(stats.entropy_sum / mask.sum(), ref_entropy, rtol=1e-4, atol=1e-4)
    assert stats.correct.item() == ref_correct.item()
    assert stats.labelled.item() == (lab != -100).sum().item()
