"""Soup Kernel Challenge solution: our own Triton kernels patched into Soup's training run.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. Every patch is installed on a class or a module
attribute that is looked up at call time, and every patched function falls back to the original
when its inputs are not real CUDA tensors / a real model (Soup's regression tests use fakes).

Kernels (all in this folder, all Python + Triton):
  rmsnorm.py        Qwen2RMSNorm.forward                      fwd + bwd
  rope.py           modeling_qwen2.apply_rotary_pos_emb        fwd + bwd (q and k in one launch)
  cross_entropy.py  Qwen2ForCausalLM.forward + trl.SFTTrainer.compute_loss
                    chunked lm_head + CE + entropy + argmax, no full logits
  lora.py           peft.tuners.lora.bnb.Linear4bit.forward    dropout+cast, scaled add
  swiglu.py         the kit's example (kept for its memory saving; does not count as ours)
"""

PATCHES = ("rmsnorm", "rope", "cross_entropy", "lora")

_applied = set()


def apply(include_example=True):
    """Install every patch (idempotent). Called by the harness before Soup builds anything."""
    from importlib import import_module

    for name in PATCHES:
        if name not in _applied:
            import_module(f"{__name__}.{name}").patch()
            _applied.add(name)
    if include_example and "swiglu" not in _applied:
        from .swiglu import patch_qwen2_mlp

        patch_qwen2_mlp()  # the example: does not count as our kernel
        _applied.add("swiglu")
