import pytest
import torch

from solution._testing import DEVICE, SMALL, TOL, needs_device

pytestmark = needs_device


def reference(x, weight, eps):
    """Qwen2RMSNorm.forward, transformers 5.16.1, line for line."""
    input_dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return weight * h.to(input_dtype)


@pytest.mark.parametrize("x_dtype,w_dtype", [
    (torch.float32, torch.float32),   # the real run: residual stream and norm weight are fp32
    (torch.float16, torch.float32),   # the guide's assumption
    (torch.float16, torch.float16),
])
@pytest.mark.parametrize("hidden", [1536, 1000])
def test_rmsnorm_matches_qwen2(x_dtype, w_dtype, hidden):
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(0)
    shape = (2, 5, hidden) if SMALL else (4, 512, hidden)
    weight = (1 + 0.1 * torch.randn(hidden)).to(DEVICE, w_dtype)   # not all ones
    x = torch.randn(shape, device=DEVICE, dtype=x_dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()

    y_ref = reference(x_ref, weight, 1e-6)
    y = rmsnorm(x, weight, 1e-6)
    assert y.dtype == y_ref.dtype and y.shape == y_ref.shape
    tol = TOL[x_dtype]
    torch.testing.assert_close(y, y_ref, atol=tol, rtol=tol)

    g = torch.randn_like(y_ref)
    y_ref.backward(g)
    y.backward(g)
    assert x.grad.dtype == x_ref.grad.dtype
    torch.testing.assert_close(x.grad, x_ref.grad, atol=tol, rtol=tol)
