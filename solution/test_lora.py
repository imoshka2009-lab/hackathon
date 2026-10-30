import pytest
import torch

from solution._testing import DEVICE, SMALL, needs_device

pytestmark = needs_device


@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
def test_dropout_cast_no_dropout_is_a_cast(out_dtype):
    from solution.lora import DropoutCast

    x = torch.randn(3, 7, 1536, device=DEVICE, requires_grad=True)
    y = DropoutCast.apply(x, 0.0, 0, out_dtype)
    assert y.dtype == out_dtype
    torch.testing.assert_close(y, x.detach().to(out_dtype), atol=0, rtol=0)
    g = torch.randn_like(y)
    y.backward(g)
    assert x.grad.dtype == x.dtype
    torch.testing.assert_close(x.grad, g.float(), atol=0, rtol=0)


def test_dropout_mask_rate_scale_and_backward():
    from solution.lora import DropoutCast

    p = 0.05
    rows = 64 if SMALL else 2048
    x = torch.randn(rows, 1536, device=DEVICE, requires_grad=True)
    y = DropoutCast.apply(x, p, 1234, torch.float32)
    kept = y != 0
    rate = 1 - kept.float().mean().item()
    assert abs(rate - p) < 0.01, rate                          # ~5% dropped
    torch.testing.assert_close(y[kept], x.detach()[kept] / (1 - p))
    y.backward(torch.ones_like(y))                             # same seed -> same mask
    torch.testing.assert_close(x.grad, kept.float() / (1 - p))
    y2 = DropoutCast.apply(x.detach(), p, 1234, torch.float32)
    assert torch.equal(y2 != 0, kept)                          # deterministic per seed
    y3 = DropoutCast.apply(x.detach(), p, 4321, torch.float32)
    assert not torch.equal(y3 != 0, kept)                      # a new seed, a new mask


@pytest.mark.parametrize("out_features", [1536, 256])         # q_proj, v_proj
def test_scaled_add_matches_peft(out_features):
    from solution.lora import ScaledAdd

    torch.manual_seed(0)
    base = torch.randn(2, 5, out_features, device=DEVICE, requires_grad=True)       # fp32 (bnb)
    lo = torch.randn(2, 5, out_features, device=DEVICE, dtype=torch.float16,
                     requires_grad=True)                                            # autocast
    base_ref = base.detach().clone().requires_grad_()
    lo_ref = lo.detach().clone().requires_grad_()
    out_ref = base_ref.clone() + lo_ref * 2.0                  # peft: result.clone() + out
    out = ScaledAdd.apply(base, lo, 2.0)
    assert out.dtype == out_ref.dtype
    torch.testing.assert_close(out, out_ref, atol=1e-6, rtol=1e-6)
    g = torch.randn_like(out_ref)
    out_ref.backward(g)
    out.backward(g)
    torch.testing.assert_close(base.grad, base_ref.grad)
    assert lo.grad.dtype == lo_ref.grad.dtype
    torch.testing.assert_close(lo.grad, lo_ref.grad, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(DEVICE != "cuda", reason="bitsandbytes 4-bit needs CUDA")
def test_patched_lora_layer_matches_peft_without_dropout():
    """The whole patched peft bnb.Linear4bit forward/backward vs the original (dropout off)."""
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model

    from solution import lora

    torch.manual_seed(0)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = bnb.nn.Linear4bit(1536, 1536, bias=True, compute_dtype=torch.float16,
                                            quant_type="nf4")

        def forward(self, x):
            return self.q_proj(x)

    model = Tiny().cuda()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                             target_modules=["q_proj"]))
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.02)
    layer = model.base_model.model.q_proj
    x = torch.randn(4, 64, 1536, device="cuda", requires_grad=True)
    cls = type(layer)
    original = cls.forward
    with torch.autocast("cuda", dtype=torch.float16):
        ref = original(layer, x)
    ref.sum().backward()
    ref_grads = [x.grad.clone()] + [p.grad.clone() for p in model.parameters() if p.requires_grad]
    model.zero_grad()
    x.grad = None
    lora.patch()
    with torch.autocast("cuda", dtype=torch.float16):
        out = layer(x)
    out.sum().backward()
    grads = [x.grad] + [p.grad for p in model.parameters() if p.requires_grad]
    assert out.dtype == ref.dtype
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)
    for got, want in zip(grads, ref_grads):
        torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
