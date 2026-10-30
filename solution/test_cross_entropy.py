import pytest
import torch

from solution._testing import DEVICE, SMALL, needs_device

pytestmark = needs_device


def reference(hidden, weight, labels, num_items_in_batch, matmul_dtype):
    """What the base run computes: lm_head logits in the autocast dtype, ForCausalLMLoss on the
    fp32 upcast, TRL's entropy (all positions) and accuracy, Soup's per-token CE for rewind."""
    from trl.trainer.utils import entropy_from_logits

    logits = (hidden.to(matmul_dtype) @ weight.to(matmul_dtype).t()).float()
    shift = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    flat_logits, flat_shift = logits.view(-1, logits.shape[-1]), shift.reshape(-1)
    if num_items_in_batch is None:
        loss = torch.nn.functional.cross_entropy(flat_logits, flat_shift, ignore_index=-100)
    else:
        loss = torch.nn.functional.cross_entropy(flat_logits, flat_shift, ignore_index=-100,
                                                 reduction="sum") / num_items_in_batch
    with torch.no_grad():
        entropy = entropy_from_logits(logits)
        token_loss = torch.nn.functional.cross_entropy(
            flat_logits, flat_shift, ignore_index=-100, reduction="none").view(shift.shape)
        pred = logits[:, :-1].argmax(-1)
        mask = labels[:, 1:] != -100
        correct = ((pred == labels[:, 1:]) & mask).sum()
    return loss, entropy, token_loss, correct, mask.sum()


@pytest.mark.parametrize("vocab", [1000, 4096, 9001])
@pytest.mark.parametrize("with_num_items", [True, False])
def test_fused_ce_matches_base(vocab, with_num_items):
    from solution.cross_entropy import fused_causal_lm_loss

    torch.manual_seed(0)
    B, T, H = (2, 9, 32) if SMALL else (4, 512, 1536)
    gpu = DEVICE == "cuda"
    hidden = torch.randn(B, T, H, device=DEVICE, requires_grad=True)
    weight = 0.05 * torch.randn(vocab, H, device=DEVICE)
    labels = torch.randint(0, vocab, (B, T), device=DEVICE)
    labels[:, : T // 3] = -100   # prompt tokens
    labels[1, -2:] = -100        # padding
    num_items = (labels[:, 1:] != -100).sum() if with_num_items else None
    hidden_ref = hidden.detach().clone().requires_grad_()

    with torch.autocast(DEVICE, dtype=torch.float16, enabled=gpu):
        loss, step = fused_causal_lm_loss(hidden, weight, labels, num_items_in_batch=num_items,
                                          chunk_rows=7 if SMALL else 512)
    matmul_dtype = torch.float16 if gpu else torch.float32
    loss_ref, ent_ref, tok_ref, correct_ref, total_ref = reference(
        hidden_ref, weight, labels, num_items, matmul_dtype)

    tol = 2e-3 if gpu else 1e-5
    torch.testing.assert_close(loss, loss_ref, atol=tol, rtol=tol)
    torch.testing.assert_close(step.token_entropy, ent_ref, atol=tol, rtol=tol)
    torch.testing.assert_close(step.token_loss, tok_ref, atol=tol, rtol=tol)
    assert int(step.labelled) == int(total_ref)
    assert abs(int(step.correct) - int(correct_ref)) <= (2 if gpu else 0)  # argmax ties in fp16

    # the GradScaler multiplies the loss before backward: the gradient must follow
    (loss * 1024.0).backward()
    (loss_ref * 1024.0).backward()
    torch.testing.assert_close(hidden.grad, hidden_ref.grad, atol=5e-3 if gpu else 1e-4,
                               rtol=2e-2 if gpu else 1e-4)


def test_rewind_row_losses_from_token_loss():
    """Soup's rewind recorder reads step.token_loss when there are no logits: its row means must
    equal what `row_losses` computes from the logits."""
    from solution.cross_entropy import fused_causal_lm_loss

    torch.manual_seed(1)
    B, T, H, V = 3, 8, 16, 500
    hidden = torch.randn(B, T, H, device=DEVICE)
    weight = 0.1 * torch.randn(V, H, device=DEVICE)
    labels = torch.randint(0, V, (B, T), device=DEVICE)
    labels[:, :3] = -100
    with torch.no_grad():
        _, step = fused_causal_lm_loss(hidden, weight, labels)
        logits = hidden @ weight.t()
        for i in range(B):
            targets = labels[i, 1:]
            ce = torch.nn.functional.cross_entropy(logits[i, :-1], targets, reduction="none",
                                                   ignore_index=-100)
            mask = targets != -100
            want = float((ce * mask).sum() / mask.sum())
            got = float(step.token_loss[i].sum() / mask.sum())
            assert abs(got - want) <= 1e-5 * max(1.0, abs(want))
