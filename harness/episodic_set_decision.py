"""Causal cross-decision memory for Recurrent SetJev.

BDH-CQ-inspired interface, NOT the unpublished BDH-CQ architecture:
  prime memory using demonstrations available at reset
  decide(context_t, option_logits_t, memory_t) via internal latent recurrence
  execute the selected set in the external environment
  observe(context_t, selected_set_t, observation_{t+1}, memory_t)
  repeat, explicitly reset at episode boundaries

The persistent memory differs from the per-query reasoning workspace.
No gold labels or future observations are arguments to decide().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from recurrent_set_decision import (
        RecurrentSetConfig,
        RecurrentSetDecision,
    )
except ModuleNotFoundError:
    from harness.recurrent_set_decision import (
        RecurrentSetConfig,
        RecurrentSetDecision,
    )


@dataclass(frozen=True)
class EpisodicSetConfig:
    context_dim: int
    latent_dim: int = 128
    max_options: int = 32
    inner_steps: int = 4
    feedback: str = "set"

    def __post_init__(self):
        if self.context_dim < 1 or self.latent_dim < 4 or self.max_options < 2:
            raise ValueError("invalid context, latent or candidate dimensions")
        if not 0 <= self.inner_steps <= 64:
            raise ValueError("inner_steps must be in [0, 64]")
        if self.feedback not in ("set", "latent_only", "static", "mlp"):
            raise ValueError("unsupported decision feedback mode")
        if self.feedback == "mlp" and self.inner_steps > 8:
            raise ValueError("the existing mlp control supports at most 8 steps")


class EpisodicSetDecision(nn.Module):
    """Per-task persistent memory plus per-query recurrent set decoder.

    Caller owns memory Tensor [B, latent_dim]. There are no hidden mutable
    tensors inside the model: concurrent episodes cannot contaminate each
    other. Memory remains differentiable across time during BPTT.
    """

    def __init__(self, config: EpisodicSetConfig):
        super().__init__()
        self.config = config
        d = config.latent_dim
        self.decision = RecurrentSetDecision(
            RecurrentSetConfig(
                context_dim=config.context_dim,
                latent_dim=d,
                max_options=config.max_options,
                feedback=config.feedback,
                max_steps=max(8, config.inner_steps),
            )
        )
        self.transition_context = nn.Linear(config.context_dim, d)
        self.transition_action = nn.Linear(config.max_options, d, bias=False)
        self.transition_observation = nn.Linear(config.context_dim, d)
        self.transition_observed_flag = nn.Linear(1, d, bias=False)
        self.transition_norm = nn.LayerNorm(d)
        self.memory_cell = nn.GRUCell(d, d)

    def empty_memory(self, batch_size: int, *, device: torch.device | str) -> Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return torch.zeros(
            batch_size, self.config.latent_dim,
            device=device, dtype=torch.float32,
        )

    def reset(self, memory: Tensor, episode_start: Tensor) -> Tensor:
        """Reset exactly the batch elements beginning new episodes."""
        self._check_memory(memory)
        if episode_start.ndim != 1 or episode_start.shape[0] != memory.shape[0]:
            raise ValueError("episode_start must be [B]")
        if episode_start.dtype != torch.bool:
            raise ValueError("episode_start must be bool")
        return torch.where(episode_start[:, None], torch.zeros_like(memory), memory)

    def prime(
        self,
        memory: Tensor,
        demonstrations: Tensor,
        demonstration_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Ingest demonstration features observed before the first decision.

        Demonstrations must be facts available to the agent (e.g. known
        input-output example pairs), not future held-out oracle labels.
        """
        self._check_memory(memory)
        zero_context = memory.new_zeros(
            (memory.shape[0], self.config.context_dim)
        )
        return self.decision.encode_memory(
            zero_context,
            demonstrations,
            demonstration_mask,
            initial_memory=memory,
        )

    def decide(
        self,
        memory: Tensor,
        context: Tensor,
        native_logits: Tensor,
        candidate_mask: Tensor,
        *,
        inner_steps: Optional[int] = None,
    ) -> list[dict[str, Tensor]]:
        """Reason on a new query without mutating memory.

        The per-query workspace is reinitialized by the core but conditioned
        on persistent memory. No gold, reward or future observation is read.
        """
        self._check_memory(memory)
        if context.shape != (memory.shape[0], self.config.context_dim):
            raise ValueError("context shape mismatch")
        if candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be bool")
        if context.device != memory.device or native_logits.device != memory.device:
            raise ValueError("features and memory must be on same device")
        steps = self.config.inner_steps if inner_steps is None else inner_steps
        return self.decision(
            context,
            native_logits,
            candidate_mask,
            steps=steps,
            memory_state=memory,
        )

    def observe(
        self,
        memory: Tensor,
        context: Tensor,
        executed_set: Tensor,
        candidate_mask: Tensor,
        post_observation: Optional[Tensor] = None,
        *,
        done: Optional[Tensor] = None,
    ) -> Tensor:
        """Update persistent memory after executing the selected action set.

        post_observation is a causally available environmental signal arriving
        AFTER the action. Do not pass hidden gold target labels in its place.
        done resets terminated batch elements after the update.
        """
        self._check_memory(memory)
        b = memory.shape[0]
        if context.shape != (b, self.config.context_dim):
            raise ValueError("context shape mismatch")
        if executed_set.ndim != 2 or executed_set.shape != candidate_mask.shape:
            raise ValueError("executed_set and candidate_mask must be B x K")
        if executed_set.shape[0] != b or executed_set.shape[1] > self.config.max_options:
            raise ValueError("executed_set dimensions are inconsistent")
        if candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be bool")
        if not bool(torch.all((executed_set == 0) | (executed_set == 1))):
            raise ValueError("executed_set must be binary")
        if not bool(torch.all(executed_set[~candidate_mask] == 0)):
            raise ValueError("padded action positions must be zero")
        if context.device != memory.device or executed_set.device != memory.device:
            raise ValueError("context, executed set and memory must share device")
        if post_observation is None:
            observed = context.new_zeros((b, self.config.context_dim))
            flag = context.new_zeros((b, 1))
        else:
            if post_observation.shape != (b, self.config.context_dim):
                raise ValueError("post_observation must be B x context_dim")
            if post_observation.device != memory.device:
                raise ValueError("observation/memory device mismatch")
            observed = post_observation.float()
            flag = context.new_ones((b, 1))
        if not bool(torch.isfinite(context).all() and torch.isfinite(observed).all()):
            raise ValueError("nonfinite environment observation")
        action = F.pad(
            executed_set.float() * candidate_mask.float(),
            (0, self.config.max_options - executed_set.shape[1]),
        )
        signal = self.transition_norm(
            torch.tanh(self.transition_context(context.float()))
            + self.transition_action(action)
            + self.transition_observation(observed)
            + self.transition_observed_flag(flag)
        )
        next_memory = self.memory_cell(signal, memory)
        if done is not None:
            next_memory = self.reset(next_memory, done)
        return next_memory

    def _check_memory(self, memory: Tensor) -> None:
        if not isinstance(memory, Tensor) or memory.ndim != 2:
            raise ValueError("memory must be a B x latent_dim Tensor")
        if memory.shape[1] != self.config.latent_dim:
            raise ValueError("memory latent width mismatch")
        if not bool(torch.isfinite(memory).all()):
            raise ValueError("nonfinite persistent memory")
