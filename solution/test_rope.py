import pytest
import torch

from solution._testing import DEVICE, SMALL, TOL, needs_device

pytestmark = needs_device


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def reference(q, k, cos, sin):
    """modeling_qwen2.apply_rotary_pos_emb (unsqueeze_dim=1), transformers 5.16.1."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def qwen_cos_sin(batch, seq, dim, dtype):
    inv_freq = 1.0 / (1e6 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    pos = torch.arange(seq, dtype=torch.float)
    emb = torch.cat([torch.outer(pos, inv_freq)] * 2, dim=-1)
    return (emb.cos()[None].expand(batch, -1, -1).to(DEVICE, dtype),
            emb.sin()[None].expand(batch, -1, -1).to(DEVICE, dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("halves_equal", [True, False])
@pytest.mark.parametrize("batch_cos", [1, None])
def test_rope_matches_qwen2(dtype, halves_equal, batch_cos):
    from solution.rope import TritonRoPE

    torch.manual_seed(0)
    B, S, HQ, HK, D = (2, 6, 12, 2, 128) if SMALL else (4, 512, 12, 2, 128)
    # q, k as the attention builds them: (B, S, H, D) buffers seen as (B, H, S, D)
    q = torch.randn(B, S, HQ, D, device=DEVICE, dtype=dtype).transpose(1, 2).requires_grad_()
    k = torch.randn(B, S, HK, D, device=DEVICE, dtype=dtype).transpose(1, 2).requires_grad_()
    if halves_equal:
        cos, sin = qwen_cos_sin(batch_cos or B, S, D, dtype)
    else:  # general rotation: the backward must not rely on equal halves
        cos = torch.randn(batch_cos or B, S, D, device=DEVICE, dtype=dtype)
        sin = torch.randn(batch_cos or B, S, D, device=DEVICE, dtype=dtype)
    q_ref = q.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()

    qo_ref, ko_ref = reference(q_ref, k_ref, cos, sin)
    qo, ko = TritonRoPE.apply(q, k, cos, sin)
    tol = TOL[dtype] * (1 if halves_equal else 4)
    for got, want in ((qo, qo_ref), (ko, ko_ref)):
        assert got.dtype == want.dtype and got.shape == want.shape
        torch.testing.assert_close(got, want, atol=tol, rtol=tol)

    gq, gk = torch.randn_like(qo_ref), torch.randn_like(ko_ref)
    torch.autograd.backward((qo_ref, ko_ref), (gq, gk))
    torch.autograd.backward((qo, ko), (gq, gk))
    torch.testing.assert_close(q.grad, q_ref.grad, atol=tol, rtol=tol)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=tol, rtol=tol)
