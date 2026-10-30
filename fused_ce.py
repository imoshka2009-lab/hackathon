"""The output head and the loss without the full logits tensor.

Base Soup (through transformers and TRL) builds logits for every position of the batch,
[4, 512, 151936] in fp16 (622 MB), upcasts them to fp32 for the loss (1.2 GB), keeps the fp32
log-softmax for backward (1.2 GB more), builds fp32 gradients of the same size, and reads the
logits again for TRL's entropy and token accuracy and for Soup's per-row rewind losses. That is
where gigabytes of the 14 GB peak go.

Here, in the training forward:
1. only the positions that matter are kept: real tokens (for the entropy metric) and positions
   with a label (for the loss); padding is dropped. Rows with labels go first;
2. the head runs on CHUNK rows at a time (fp16 matmul, as autocast does in base) into one reused
   [CHUNK, vocab] fp16 buffer;
3. one Triton program per row reads the row of logits once and computes everything that base
   computes from it: logsumexp, the cross-entropy at the label, the entropy, the argmax (token
   accuracy); a second sweep overwrites the logits in place with the gradient softmax - onehot;
4. the gradient on the hidden states (grad @ W) is taken right there, chunk by chunk, so nothing
   of vocabulary size outlives its chunk. Backward only scales it.

The loss is the one of transformers' ForCausalLMLoss (sum over labelled positions / num_items_in_batch,
or the mean), TRL's metrics are logged by TRL's own formulas from the per-row results, and Soup's
rewind recorder gets each position's loss through the hook it already has for steps without logits
(the `_soup_chunked_step` attribute it reads, a [batch, seq] `token_loss`).

The idea of a fused linear cross-entropy that never materialises the logits is from Liger Kernel's
FusedLinearCrossEntropy and Apple's Cut Cross-Entropy (Wijmans et al., 2024); the code is ours.
"""

import torch
import triton
import triton.language as tl

CHUNK = 512  # rows of logits alive at a time: 512 x 151936 x 2 B = 155 MB
IGNORE = -100

try:  # the attribute Soup's rewind recorder reads when a step has no logits
    from soup_cli.utils.stream_loss_chunk import STEP_ATTR
except Exception:  # noqa: BLE001
    STEP_ATTR = "_soup_chunked_step"


@triton.jit
def ce_row_kernel(logits_ptr, stride_row, target_ptr, loss_ptr, entropy_ptr, correct_ptr,
                  n_vocab: tl.constexpr, WRITE_GRAD: tl.constexpr, BLOCK_V: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride_row
    target = tl.load(target_ptr + row).to(tl.int32)
    has_target = target >= 0
    cols = tl.arange(0, BLOCK_V)

    # pass 1, per lane: running max m, sum of exp(x - m), sum of exp(x - m) * x, best value/index
    m = tl.full([BLOCK_V], float("-inf"), tl.float32)
    s = tl.zeros([BLOCK_V], tl.float32)
    q = tl.zeros([BLOCK_V], tl.float32)
    best = tl.full([BLOCK_V], float("-inf"), tl.float32)
    best_idx = tl.zeros([BLOCK_V], tl.int32)
    for start in range(0, n_vocab, BLOCK_V):
        offs = start + cols
        mask = offs < n_vocab
        x = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)
        better = x > best  # strict: the first index wins a tie, like torch.argmax
        best_idx = tl.where(better, offs, best_idx)
        best = tl.where(better, x, best)
        m_new = tl.maximum(m, x)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m - m_safe)
        e = tl.exp(x - m_safe)
        s = s * alpha + e
        q = q * alpha + tl.where(mask, e * x, 0.0)
        m = m_new
    m_all = tl.max(m, axis=0)
    w = tl.exp(m - m_all)
    s_all = tl.sum(s * w, axis=0)
    q_all = tl.sum(q * w, axis=0)
    lse = m_all + tl.log(s_all)
    entropy = lse - q_all / s_all  # -sum p log p = lse - sum p x

    top = tl.max(best, axis=0)
    argmax = tl.min(tl.where(best == top, best_idx, n_vocab), axis=0)

    safe_target = tl.where(has_target, target, 0)
    x_target = tl.load(base + safe_target).to(tl.float32)
    loss = tl.where(has_target, lse - x_target, 0.0)
    correct = (argmax == target) & has_target
    tl.store(loss_ptr + row, loss)
    tl.store(entropy_ptr + row, entropy)
    tl.store(correct_ptr + row, correct.to(tl.int32))

    if WRITE_GRAD:  # pass 2: d loss_row / d logits = softmax - onehot (0 for unlabelled rows)
        scale = tl.where(has_target, 1.0, 0.0)
        for start in range(0, n_vocab, BLOCK_V):
            offs = start + cols
            mask = offs < n_vocab
            x = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
            g = tl.exp(x - lse) * scale - tl.where(offs == target, 1.0, 0.0)
            tl.store(base + offs, g.to(logits_ptr.dtype.element_ty), mask=mask)


def ce_rows(logits, target, write_grad):
    """Per-row loss, entropy and correct flag of [n, vocab] logits; with write_grad the logits
    are overwritten by softmax - onehot."""
    n, vocab = logits.shape
    loss = torch.empty(n, dtype=torch.float32, device=logits.device)
    entropy = torch.empty(n, dtype=torch.float32, device=logits.device)
    correct = torch.empty(n, dtype=torch.int32, device=logits.device)
    if n:
        ce_row_kernel[(n,)](logits, logits.stride(0), target, loss, entropy, correct, vocab,
                            WRITE_GRAD=write_grad, BLOCK_V=2048, num_warps=8)
    return loss, entropy, correct


class FusedLinearCE(torch.autograd.Function):
    """Rows of hidden states [n, hidden] -> sum of their cross-entropies / denominator.

    Rows [0, n_labelled) have a label, the rest only need the metrics. The head weight is frozen;
    the gradient on the hidden states is computed in forward and only scaled in backward."""

    @staticmethod
    def forward(ctx, hidden, weight16, target, n_labelled, denominator):
        n, width = hidden.shape
        vocab = weight16.shape[0]
        with_grad = ctx.needs_input_grad[0]
        hidden16 = hidden.to(weight16.dtype)
        grad = torch.zeros((n, width), dtype=torch.float32, device=hidden.device) \
            if with_grad else None
        buffer = torch.empty((min(CHUNK, max(n, 1)), vocab), dtype=weight16.dtype,
                             device=hidden.device)
        losses, entropies, corrects = [], [], []
        # both matmul operands are fp16 already (as autocast makes them in base); out= ops are
        # not autocast, so autocast is off here explicitly
        with torch.autocast(hidden.device.type, enabled=False):
            for start in range(0, n, CHUNK):
                stop = min(n, start + CHUNK)
                logits = buffer[: stop - start]
                torch.mm(hidden16[start:stop], weight16.t(), out=logits)
                labelled = max(0, min(stop, n_labelled) - start)
                write_grad = with_grad and labelled > 0
                loss, entropy, correct = ce_rows(logits, target[start:stop], write_grad)
                losses.append(loss)
                entropies.append(entropy)
                corrects.append(correct)
                if write_grad:
                    grad[start:start + labelled] = torch.mm(logits[:labelled], weight16)
        del buffer
        row_loss = torch.cat(losses) if losses else hidden.new_zeros(0, dtype=torch.float32)
        row_entropy = torch.cat(entropies) if entropies else row_loss
        row_correct = torch.cat(corrects) if corrects else row_loss.int()
        total = row_loss.sum() / denominator
        if with_grad:
            grad /= denominator
            ctx.save_for_backward(grad)
        ctx.mark_non_differentiable(row_loss, row_entropy, row_correct)
        return total, row_loss, row_entropy, row_correct

    @staticmethod
    def backward(ctx, grad_total, _l, _e, _c):
        (grad,) = ctx.saved_tensors
        return grad * grad_total, None, None, None, None


class Stats:
    __slots__ = ("token_loss", "entropy_sum", "correct", "labelled")

    def __init__(self, token_loss, entropy_sum, correct, labelled):
        self.token_loss, self.entropy_sum = token_loss, entropy_sum
        self.correct, self.labelled = correct, labelled


class _Step:
    """What Soup's rewind recorder reads for a step without logits."""

    __slots__ = ("token_loss",)

    def __init__(self, token_loss):
        self.token_loss = token_loss


def fused_head_loss(hidden, weight, labels, attention_mask, num_items_in_batch):
    """ForCausalLMLoss(lm_head(hidden), labels) and TRL's metrics without the logits."""
    batch, seq, width = hidden.shape
    device = hidden.device
    labels = labels.to(device)
    target = torch.nn.functional.pad(labels, (0, 1), value=IGNORE)[:, 1:].reshape(-1)
    has_label = target != IGNORE
    if attention_mask is None:
        real = torch.ones_like(has_label)
    else:
        real = attention_mask.to(device).reshape(-1) != 0
    other = real & ~has_label
    label_rows = has_label.nonzero().squeeze(1)
    other_rows = other.nonzero().squeeze(1)
    rows = torch.cat([label_rows, other_rows])
    n_labelled = label_rows.numel()

    if num_items_in_batch is None:
        denominator = torch.tensor(float(n_labelled), device=device)
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(device)
    else:
        denominator = num_items_in_batch

    weight16 = weight if weight.dtype == torch.float16 else weight.detach().to(torch.float16)
    picked = hidden.reshape(-1, width).index_select(0, rows)
    loss, row_loss, row_entropy, row_correct = FusedLinearCE.apply(
        picked, weight16, target.index_select(0, rows), n_labelled, denominator)

    with torch.no_grad():
        token_loss = torch.zeros(batch * seq, dtype=torch.float32, device=device)
        token_loss.index_copy_(0, rows, row_loss)
        token_entropy = torch.zeros(batch * seq, dtype=torch.float32, device=device)
        token_entropy.index_copy_(0, rows, row_entropy)
        if attention_mask is None:
            entropy_sum = token_entropy.sum()
        else:
            entropy_sum = (token_entropy * attention_mask.to(device).reshape(-1)).sum()
        stats = Stats(token_loss=token_loss.view(batch, seq), entropy_sum=entropy_sum,
                      correct=row_correct[:n_labelled].sum(),
                      labelled=torch.tensor(n_labelled, device=device))
    return loss, stats


class _State:
    active = False
    stats = None


STATE = _State()
_ORIGINAL = {}


def _causal_lm(model):
    get_base_model = getattr(model, "get_base_model", None)
    return get_base_model() if callable(get_base_model) else model


def _head_fits(causal_lm):
    from transformers.models.qwen2 import modeling_qwen2

    if not isinstance(causal_lm, modeling_qwen2.Qwen2ForCausalLM):
        return False
    head = getattr(causal_lm, "lm_head", None)
    weight = getattr(head, "weight", None)
    return (type(head) is torch.nn.Linear and head.bias is None and weight is not None
            and not weight.requires_grad and weight.is_cuda
            and weight.dtype in (torch.float32, torch.float16)
            and getattr(causal_lm.config, "final_logit_softcapping", None) is None)


def patch_qwen2_causal_lm():
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2ForCausalLM
    if "causal_forward" in _ORIGINAL:
        return
    original = cls.forward
    _ORIGINAL["causal_forward"] = original

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        if (not STATE.active or labels is None or not self.training
                or not (isinstance(logits_to_keep, int) and logits_to_keep == 0)
                or kwargs.get("shift_labels") is not None or kwargs.get("return_dict") is False
                or not _head_fits(self)):
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            logits_to_keep=logits_to_keep, **kwargs)
        kwargs.pop("return_dict", None)  # None or True: a ModelOutput, as can_return_tuple gives
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        loss, STATE.stats = fused_head_loss(outputs.last_hidden_state, self.lm_head.weight,
                                            labels, attention_mask,
                                            kwargs.get("num_items_in_batch"))
        return CausalLMOutputWithPast(loss=loss, logits=None,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    cls.forward = forward


def _can_fuse(trainer, model, inputs):
    args = getattr(trainer, "args", None)
    if args is None or not isinstance(model, torch.nn.Module) or not model.training:
        return False
    if getattr(args, "use_liger_kernel", False) or getattr(args, "loss_type", "nll") != "nll":
        return False
    if getattr(trainer, "compute_loss_func", None) is not None:
        return False
    if getattr(trainer, "label_smoother", None) is not None:
        return False
    if getattr(trainer, "num_virtual_tokens", 0) or getattr(trainer, "aux_loss_enabled", False):
        return False
    if not hasattr(trainer, "_metrics") or not hasattr(trainer, "_total_train_tokens"):
        return False
    if not isinstance(inputs, dict) or "labels" not in inputs or "attention_mask" not in inputs:
        return False
    if "shift_labels" in inputs or "position_ids" in inputs and "attention_mask" not in inputs:
        return False
    return _head_fits(_causal_lm(model))


def patch_sft_trainer():
    import trl
    from transformers import Trainer

    cls = trl.SFTTrainer
    if "compute_loss" in _ORIGINAL:
        return
    original = cls.compute_loss
    _ORIGINAL["compute_loss"] = original

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not _can_fuse(self, model, inputs):
            if getattr(self, STEP_ATTR, None) is not None:
                setattr(self, STEP_ATTR, None)
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        mode = "train"
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        STATE.active, STATE.stats = True, None
        try:
            loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                                 num_items_in_batch=num_items_in_batch)
        finally:
            STATE.active = False
        stats, STATE.stats = STATE.stats, None
        if stats is None:
            raise RuntimeError("solution: the fused head loss did not run in this step")

        # TRL's metrics, by TRL's formulas (SFTTrainer.compute_loss, trl 0.29.1)
        gather = self.accelerator.gather_for_metrics
        attention_mask = inputs["attention_mask"]
        with torch.no_grad():
            entropy = stats.entropy_sum / attention_mask.sum()
            entropy = gather(entropy).mean().item()
        self._metrics[mode]["entropy"].append(entropy)
        num_tokens_in_batch = gather(attention_mask.sum()).sum().item()
        self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
        with torch.no_grad():
            correct_tokens = gather(stats.correct)
            total_tokens = gather(stats.labelled)
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        setattr(self, STEP_ATTR, _Step(stats.token_loss))  # for Soup's rewind recorder
        return (loss, outputs) if return_outputs else loss

    cls.compute_loss = compute_loss
