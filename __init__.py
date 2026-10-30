"""Our solution: Triton kernels for the Soup training step of Qwen2.5-1.5B (4-bit NF4 + LoRA)
on a Kaggle T4.

`apply()` is called by the harness before Soup builds the model and the trainer. Every patch is
on a class or a module function and looks itself up at call time; each one falls back to the
original code when its input is not what it was written for (CPU tensors, fake modules in Soup's
tests, a trainable norm or head, other dtypes).

Our kernels (all `@triton.jit`, all in this folder):
- nf4.py       nf4_dequant_kernel            NF4 + double-quant scales -> fp16 weight, for every
                                             4-bit layer, forward and backward;
- rmsnorm.py   rmsnorm_fwd/bwd_kernel        Qwen2RMSNorm in one pass each way;
- rope.py      rope_kernel                   rotary embedding of q and k, forward and backward;
- fused_ce.py  ce_row_kernel                 loss, entropy, argmax and softmax gradient of a row
                                             of logits, with the head run in chunks so the full
                                             logits are never built.
The example (swiglu.py, kernels prefixed `example_`) still runs inside our MLP forward; it does
not count as ours.

Switches below are for experiments on Kaggle; all on is the submitted configuration.
"""

USE_NF4 = True        # our dequant kernel for all 4-bit layers
USE_FP16_MLP = True   # keep gate/up/down outputs in fp16 (needs USE_NF4)
USE_RMSNORM = True
USE_ROPE = True
USE_FUSED_CE = True


def apply():
    from .swiglu import swiglu

    if USE_NF4:
        from .nf4 import patch_linear4bit

        patch_linear4bit()
    if USE_RMSNORM:
        from .rmsnorm import patch_qwen2_rmsnorm

        patch_qwen2_rmsnorm()
    if USE_ROPE:
        from .rope import patch_qwen2_rope

        patch_qwen2_rope()
    if USE_FUSED_CE:
        from .fused_ce import patch_qwen2_causal_lm, patch_sft_trainer

        patch_qwen2_causal_lm()
        patch_sft_trainer()
    _patch_mlp(swiglu, fp16=USE_NF4 and USE_FP16_MLP)


def _patch_mlp(swiglu, fp16):
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls.forward, "_solution", False):
        return
    original = cls.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act != "silu" or not x.is_cuda:
            return original(self, x)
        if fp16:
            from .nf4 import project

            gate, up = project(self.gate_proj, x), project(self.up_proj, x)
            if gate.dtype != up.dtype:
                up = up.to(gate.dtype)
            return project(self.down_proj, swiglu(gate, up))
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))

    forward._solution = True
    cls.forward = forward
