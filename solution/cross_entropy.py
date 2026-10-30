"""Kernel 3: fused, chunked cross-entropy + entropy + argmax (token accuracy) over the 151,936 vocab.

What the base run does every step (transformers 5.16.1 + trl 0.29.1 + soup-cli 0.77.0):
  Qwen2ForCausalLM.forward: logits = lm_head(hidden)        [4, 512, 151936] fp16 under autocast
  ForCausalLMLoss:          logits.float() -> cross_entropy(sum) / num_items_in_batch
  TRL SFTTrainer.compute_loss: entropy_from_logits(outputs.logits) weighted by attention_mask,
                               argmax(logits[:, :-1]) == labels[:, 1:] on labelled tokens
  Soup rewind (rewind_hf.row_losses): per-row mean CE of the labelled tokens
plus the fp32 copies and the logits gradient: several GB of the peak.

Here the full logits never exist. The rows go through in chunks of CHUNK_ROWS tokens:
  1. logits_chunk = hidden_chunk @ W.T          (cuBLAS, in the autocast dtype like the original)
  2. one Triton program per row, two passes over the vocab:
       pass 1: online max / log-sum-exp, sum(p*z) for the entropy, first argmax, z[label]
       pass 2: dz = (softmax - onehot) * scale, written IN PLACE over the logits chunk
  3. d_hidden_chunk = dz @ W, logits chunk freed
The gradient of the hidden states is taken in forward (the head is frozen) and the backward only
multiplies it by the incoming loss gradient (the GradScaler's scale under fp16), the same scheme as
Soup's own `stream_loss_chunk` (soup_cli/utils/stream_loss_chunk.py), which we read for the
interface; idea credits: Liger-Kernel / Unsloth chunked CE. The code here is our own.

Hooks:
  - Qwen2ForCausalLM.forward: only when "armed" by our compute_loss (training steps), returns the
    loss and logits=None; any other call goes to the original.
  - trl.SFTTrainer.compute_loss: arms the fused forward, then logs entropy / num_tokens /
    mean_token_accuracy by TRL's own formulas from what the kernel produced, and leaves the
    per-token losses where Soup's rewind recorder looks for them when there are no logits
    (`_soup_chunked_step`, read in soup_cli/trainer/rewind_hf.py).
"""

import torch
import triton
import triton.language as tl

from . import _common

IGNORE_INDEX = -100
CHUNK_ROWS = 512            # 512 x 151,936 fp16 = 148 MiB alive at a time
BLOCK_V = 4096
# dz is stored in fp16 before the d_hidden matmul. (p - onehot) / num_tokens would underflow fp16
# for most of the vocabulary, so dz is stored multiplied by GRAD_GAIN (a power of two, exact) and
# divided back in fp32. 2**16 is also the GradScaler's initial scale, so dz has about the magnitude
# the base run rounds it at.
GRAD_GAIN = 65536.0
STEP_ATTR = "_soup_chunked_step"   # soup_cli.utils.stream_loss_chunk.STEP_ATTR


@triton.jit
def fused_ce_kernel(LOGITS, TARGET, LOSS, ENTROPY, PRED, row_stride, V, grad_scale,
                    BLOCK_V: tl.constexpr, WRITE_GRAD: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = LOGITS + row * row_stride
    label = tl.load(TARGET + row)
    labelled = label != -100
    cols = tl.arange(0, BLOCK_V)

    m = float("-inf")      # running max
    s = 0.0                # sum exp(z - m)
    t = 0.0                # sum exp(z - m) * z      -> sum(p * z) = t / s
    best = float("-inf")   # argmax value
    best_idx = 0           # argmax index (first occurrence, as torch.argmax)
    for off in range(0, V, BLOCK_V):
        idx = off + cols
        mask = idx < V
        z = tl.load(base + idx, mask=mask, other=float("-inf")).to(tl.float32)
        blk_max = tl.max(z, axis=0)
        m_new = tl.maximum(m, blk_max)
        e = tl.where(mask, tl.exp(z - m_new), 0.0)
        alpha = tl.exp(m - m_new)
        s = s * alpha + tl.sum(e, axis=0)
        t = t * alpha + tl.sum(e * tl.where(mask, z, 0.0), axis=0)
        blk_arg = tl.min(tl.where(z == blk_max, idx, V), axis=0)
        best_idx = tl.where(blk_max > best, blk_arg, best_idx)
        best = tl.maximum(best, blk_max)
        m = m_new

    lse = m + tl.log(s)
    z_label = tl.load(base + tl.where(labelled, label, 0)).to(tl.float32)
    tl.store(LOSS + row, tl.where(labelled, lse - z_label, 0.0))
    tl.store(ENTROPY + row, lse - t / s)
    tl.store(PRED + row, best_idx.to(tl.int64))

    if WRITE_GRAD:
        scale = tl.where(labelled, grad_scale, 0.0)
        for off in range(0, V, BLOCK_V):
            idx = off + cols
            mask = idx < V
            z = tl.load(base + idx, mask=mask, other=0.0).to(tl.float32)
            dz = (tl.exp(z - lse) - tl.where(idx == label, 1.0, 0.0)) * scale
            tl.store(base + idx, dz.to(LOGITS.dtype.element_ty), mask=mask)


class FusedLinearCrossEntropy(torch.autograd.Function):
    """loss = sum_rows CE(hidden @ W.T, target) / denominator, plus per-token loss, entropy and
    argmax, without the full logits. W is frozen: the gradient goes to `hidden` only."""

    @staticmethod
    def forward(ctx, hidden, weight, target, denominator, matmul_dtype, chunk_rows):
        shape = hidden.shape
        h2 = hidden.reshape(-1, shape[-1])
        tgt = target.reshape(-1).to(device=hidden.device, dtype=torch.int64).contiguous()
        n_rows, vocab = h2.shape[0], weight.shape[0]
        dev = hidden.device
        with_grad = ctx.needs_input_grad[0]
        denom = float(denominator)
        grad_scale = GRAD_GAIN / denom if denom else 0.0

        token_loss = torch.empty(n_rows, dtype=torch.float32, device=dev)
        entropy = torch.empty(n_rows, dtype=torch.float32, device=dev)
        pred = torch.empty(n_rows, dtype=torch.int64, device=dev)
        grad_h = torch.empty((n_rows, shape[-1]), dtype=torch.float32, device=dev) \
            if with_grad else None
        with torch.autocast(device_type=dev.type, enabled=False):
            w = weight.to(matmul_dtype)  # one cast per step (the base's autocast does the same)
            for start in range(0, n_rows, chunk_rows):
                stop = min(start + chunk_rows, n_rows)
                logits = h2[start:stop].to(matmul_dtype) @ w.t()
                fused_ce_kernel[(stop - start,)](
                    logits, tgt[start:stop], token_loss[start:stop], entropy[start:stop],
                    pred[start:stop], logits.stride(0), vocab, grad_scale,
                    BLOCK_V=BLOCK_V, WRITE_GRAD=with_grad, num_warps=8)
                if with_grad:
                    grad_h[start:stop] = logits @ w   # logits now hold dz * GRAD_GAIN
                del logits
            del w
        loss = token_loss.sum() / denominator
        ctx.save_for_backward(grad_h)
        ctx.shape, ctx.dtype = shape, hidden.dtype
        ctx.mark_non_differentiable(token_loss, entropy, pred)
        return loss, token_loss.view(shape[:-1]), entropy.view(shape[:-1]), pred.view(shape[:-1])

    @staticmethod
    def backward(ctx, grad_loss, _grad_token_loss, _grad_entropy, _grad_pred):
        (grad_h,) = ctx.saved_tensors
        if grad_h is None:
            return None, None, None, None, None, None
        grad = grad_h * (grad_loss.float() / GRAD_GAIN)
        return grad.view(ctx.shape).to(ctx.dtype), None, None, None, None, None


class ChunkedStep:
    """What the fused forward leaves for the metrics. Same fields as Soup's
    stream_loss_chunk.ChunkedStep, so Soup's rewind recorder can read `token_loss`."""

    def __init__(self, token_entropy, token_loss, pred, target):
        self.token_entropy = token_entropy          # [B, T]
        self.token_loss = token_loss                # [B, T] fp32, 0 where not labelled
        labelled = target != IGNORE_INDEX
        self.labelled = labelled.sum()
        self.correct = ((pred == target) & labelled).sum()


def shift_labels_of(labels):
    """labels moved one left, as ForCausalLMLoss does (last position scored against nothing)."""
    return torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]


def fused_causal_lm_loss(hidden, weight, labels, num_items_in_batch=None, shift_labels=None,
                         chunk_rows=CHUNK_ROWS):
    """ForCausalLMLoss(hidden @ weight.T, labels) and a ChunkedStep, without the full logits."""
    target = shift_labels if shift_labels is not None else shift_labels_of(labels)
    target = target.to(hidden.device)
    if num_items_in_batch is None:      # fixed_cross_entropy: mean over labelled tokens
        denominator = (target != IGNORE_INDEX).sum()
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(hidden.device)
    else:
        denominator = num_items_in_batch
    matmul_dtype = _common.autocast_dtype() or torch.promote_types(hidden.dtype, weight.dtype)
    loss, token_loss, entropy, pred = FusedLinearCrossEntropy.apply(
        hidden, weight, target, denominator, matmul_dtype, chunk_rows)
    return loss, ChunkedStep(entropy, token_loss, pred, target)


class _State:
    armed = False   # set by our compute_loss for exactly one model forward
    step = None


STATE = _State()


def _head_ok(lm):
    head = getattr(lm, "lm_head", None)
    return (type(head) is torch.nn.Linear and head.bias is None
            and not head.weight.requires_grad and _common.on_gpu(head.weight)
            and getattr(lm.config, "final_logit_softcapping", None) is None)


def _patch_causal_lm():
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2 as m

    original = m.Qwen2ForCausalLM.forward
    if getattr(original, "_solution_patch", False):
        return

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        if not (STATE.armed and labels is not None and _common.on_gpu(labels)
                and isinstance(logits_to_keep, int) and logits_to_keep == 0
                and kwargs.get("return_dict", True) is not False and _head_ok(self)):
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            logits_to_keep=logits_to_keep, **kwargs)
        STATE.armed = False
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        loss, STATE.step = fused_causal_lm_loss(
            outputs.last_hidden_state, self.lm_head.weight, labels,
            num_items_in_batch=kwargs.get("num_items_in_batch"),
            shift_labels=kwargs.get("shift_labels"))
        return CausalLMOutputWithPast(loss=loss, logits=None,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    forward._solution_patch = True
    m.Qwen2ForCausalLM.forward = forward


def _causal_lm_of(trainer, model):
    unwrap = getattr(getattr(trainer, "accelerator", None), "unwrap_model", None)
    inner = unwrap(model) if callable(unwrap) else model
    get_base = getattr(inner, "get_base_model", None)
    return get_base() if callable(get_base) else inner


def _can_fuse(trainer, model, inputs):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    args = getattr(trainer, "args", None)
    if not (isinstance(model, torch.nn.Module) and isinstance(inputs, dict) and args is not None):
        return False
    if not getattr(trainer.model, "training", False):          # eval / predict: untouched
        return False
    if "labels" not in inputs or "shift_labels" in inputs or "attention_mask" not in inputs:
        return False
    if not _common.on_gpu(inputs["labels"], inputs["attention_mask"]):
        return False
    if getattr(args, "use_liger_kernel", False) or getattr(args, "loss_type", "nll") != "nll":
        return False
    if getattr(trainer, "num_virtual_tokens", 0) or getattr(trainer, "aux_loss_enabled", False):
        return False
    if getattr(trainer, "compute_loss_func", None) is not None or \
            getattr(trainer, "label_smoother", None) is not None:
        return False
    if getattr(trainer, "_soup_chunked_head_loss", None) is not None:   # Soup's own chunking
        return False
    if not all(hasattr(trainer, name) for name in ("_metrics", "_total_train_tokens")):
        return False
    lm = _causal_lm_of(trainer, model)
    return isinstance(lm, Qwen2ForCausalLM) and _head_ok(lm)


def _log_metrics(trainer, mode, inputs, step):
    """entropy, num_tokens and mean_token_accuracy exactly as trl 0.29.1 SFTTrainer.compute_loss
    logs them (non-liger path, attention_mask present, no virtual tokens)."""
    gather = trainer.accelerator.gather_for_metrics
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        entropy = torch.sum(step.token_entropy * attention_mask) / attention_mask.sum()
        entropy = gather(entropy).mean().item()
    trainer._metrics[mode]["entropy"].append(entropy)
    if mode == "train":
        trainer._total_train_tokens += gather(attention_mask.sum()).sum().item()
    trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    with torch.no_grad():
        correct = gather(step.correct)
        total = gather(step.labelled)
        total_sum = total.sum()
        accuracy = (correct.sum() / total_sum).item() if total_sum > 0 else 0.0
    trainer._metrics[mode]["mean_token_accuracy"].append(accuracy)


def _patch_trainer():
    from trl import SFTTrainer

    original = SFTTrainer.compute_loss
    if getattr(original, "_solution_patch", False):
        return

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not _can_fuse(self, model, inputs):
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        mode = "train" if self.model.training else "eval"
        inputs.pop("_prediction_loss_only", None)
        labels = inputs["labels"]
        inputs["use_cache"] = False
        STATE.armed, STATE.step = True, None
        try:
            loss, outputs = super(SFTTrainer, self).compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        finally:
            STATE.armed = False
        step, STATE.step = STATE.step, None
        if step is None:
            # The model took the original path after all (it has logits): same metrics from them.
            from trl.trainer.utils import entropy_from_logits

            with torch.no_grad():
                logits = outputs.logits
                step = ChunkedStep(entropy_from_logits(logits), None, logits.argmax(dim=-1),
                                   shift_labels_of(labels))
            setattr(self, STEP_ATTR, None)   # rewind reads the logits themselves
        else:
            setattr(self, STEP_ATTR, step)   # rewind reads step.token_loss (no logits)
        _log_metrics(self, mode, inputs, step)
        return (loss, outputs) if return_outputs else loss

    compute_loss._solution_patch = True
    SFTTrainer.compute_loss = compute_loss


def patch():
    _patch_causal_lm()
    _patch_trainer()
