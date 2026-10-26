"""Strict, minimal PEFT LoRA checkpoint save/load checks for episodic SetJev.

PyTorch's strict=False is necessary when restoring only adapter tensors into
a full language model, but *cannot* alone prove that a named adapter matched.
This module verifies all expected LoRA tensors, their shapes, and restored
values, including old checkpoints that did not retain the LoRA hyperparameters.
"""
from __future__ import annotations

from typing import Mapping

import torch


def lora_tensors(model) -> dict[str, torch.Tensor]:
    """Export exactly the LoRA tensors present in the model."""
    state = model.state_dict()
    tensors = {k: v.detach().cpu().clone() for k, v in state.items() if "lora_" in k}
    if not tensors:
        raise ValueError("model has no LoRA tensors to save")
    return tensors


def load_lora_tensors_verified(model, saved_lora: Mapping[str, torch.Tensor]) -> int:
    """Restore the entire expected adapter subset, not an arbitrary intersection.

    We intentionally call load_state_dict(strict=False) because the BASE weights
    are absent from the adapter-only checkpoint. All LoRA keys and shapes must
    match exactly; after load, compare recovered adapter values byte-for-byte.
    """
    if not isinstance(saved_lora, dict) or not saved_lora:
        raise ValueError("empty/malformed LoRA checkpoint")
    expected_state = model.state_dict()
    expected_keys = {key for key in expected_state if "lora_" in key}
    provided = set(saved_lora)
    if not expected_keys:
        raise ValueError("evaluation model has no LoRA tensors")
    if provided != expected_keys:
        missing = sorted(expected_keys - provided)
        unexpected = sorted(provided - expected_keys)
        raise ValueError(
            f"LoRA checkpoint keys mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected; examples {missing[:3]} / {unexpected[:3]}"
        )
    for key, tensor in saved_lora.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"LoRA checkpoint entry {key} is not a Tensor")
        if expected_state[key].shape != tensor.shape:
            raise ValueError(
                f"LoRA tensor shape mismatch for {key}: "
                f"expected {tuple(expected_state[key].shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"LoRA checkpoint contains nonfinite tensor {key}")
    result = model.load_state_dict(dict(saved_lora), strict=False)
    # The non-adapter base parameters are EXPECTED missing, not unexpected.
    if result.unexpected_keys or set(result.missing_keys) != (set(expected_state) - expected_keys):
        raise RuntimeError("LoRA-only load produced inconsistent missing/unexpected keys")
    reloaded = model.state_dict()
    for key, tensor in saved_lora.items():
        if not torch.equal(reloaded[key].detach().cpu(), tensor.to(reloaded[key].dtype).cpu()):
            raise RuntimeError(f"LoRA tensor failed exact post-load check: {key}")
    return len(expected_keys)


def check_lora_gradients(model) -> dict:
    """Fail loudly if LoRA forward path did not produce any useful gradient."""
    candidates = [
        (name, p) for name, p in model.named_parameters()
        if "lora_" in name and p.requires_grad
    ]
    if not candidates:
        raise RuntimeError("no trainable LoRA parameters found")
    with_grad = [(name, p.grad) for name, p in candidates if p.grad is not None]
    if not with_grad:
        raise RuntimeError("LoRA parameters received no gradients: check backbone forward path")
    total = 0.0
    for name, grad in with_grad:
        if not torch.isfinite(grad).all().item():
            raise FloatingPointError(f"nonfinite LoRA gradient: {name}")
        total += float(grad.detach().float().square().sum().item())
    if total <= 0:
        raise RuntimeError("LoRA gradient norm is zero: check trainable adapter connection")
    return {
        "trainable_lora_tensors": len(candidates),
        "lora_tensors_with_gradient": len(with_grad),
        "lora_gradient_l2": total ** 0.5,
    }
