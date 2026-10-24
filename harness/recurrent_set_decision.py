"""Trainable, non-verbal recurrent set decision head (BDH-CQ-inspired, NOT BDH-CQ).

The decoder is an exact cardinality-potential distribution:
    P(S|x) proportional to exp(sum(scores[i] for i in S) + a[|S|]).
The recurrent state can read the *current* candidate belief and revise both
membership scores and count correction. All parameters are shared across steps.

Only the published *memory/workspace interface* of BDH-CQ is borrowed. Its
internal BDH-CQ update equations and training recipe are proprietary.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence

import torch
from torch import Tensor, nn

try:
    from set_distillation import torch_log_partitions
except ModuleNotFoundError:
    from harness.set_distillation import torch_log_partitions


@dataclass(frozen=True)
class RecurrentSetConfig:
    context_dim: int
    latent_dim: int = 128
    max_options: int = 32
    feedback: str = "set"  # set | latent_only | static | mlp
    max_steps: int = 8

    def __post_init__(self):
        if self.context_dim < 1 or self.latent_dim < 4:
            raise ValueError("context_dim must be positive and latent_dim >= 4")
        if self.max_options < 2:
            raise ValueError("max_options must be >= 2")
        if self.feedback not in ("set", "latent_only", "static", "mlp"):
            raise ValueError("unknown feedback mode")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")


def set_log_partition(
    scores: Tensor, correction: Tensor, allowed: Sequence[int]
) -> Tensor:
    """Exact O(K^2) log-partition over permitted non/empty subsets."""
    if scores.ndim != 1 or correction.ndim != 1:
        raise ValueError("scores and correction must be 1D")
    k = scores.numel()
    if correction.numel() < k + 1:
        raise ValueError("correction has insufficient cardinality entries")
    support = sorted(set(int(m) for m in allowed))
    if not support or support[0] < 0 or support[-1] > k:
        raise ValueError("invalid permitted cardinalities")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("nonfinite candidate scores")
    # Restoring m * center is essential: removing it changes the count prior.
    center = scores.detach().max()
    log_z = torch_log_partitions(scores - center)
    counts = torch.as_tensor(support, device=scores.device, dtype=torch.long)
    return torch.logsumexp(log_z[counts] + center * counts + correction[counts], dim=0)


def gold_set_nll(
    scores: Tensor,
    correction: Tensor,
    gold: Tensor,
    allowed: Sequence[int],
    normalize: bool = True,
) -> Tensor:
    """Differentiable -log P(S_gold|x), no oracle count used at inference."""
    if gold.shape != scores.shape or gold.ndim != 1:
        raise ValueError("gold and scores must be same-shaped 1D tensors")
    if not bool(torch.all((gold == 0) | (gold == 1))):
        raise ValueError("gold must be multi-hot")
    m = int(gold.sum().item())
    if m not in allowed:
        raise ValueError("gold cardinality violates public protocol")
    loss = set_log_partition(scores, correction, allowed)
    loss = loss - torch.dot(scores, gold.to(scores.dtype)) - correction[m]
    return loss / scores.numel() if normalize else loss


def revision_auxiliary_bce(
    scores: Tensor,
    previous_set: Tensor,
    target_set: Tensor,
    *,
    unchanged_weight: float = 0.0,
) -> Tensor:
    """Auxiliary loss emphasizing options changed from the *previous* set.

    This term alone has direct per-logit gradients only for changed options
    when unchanged_weight=0. It is NOT a normalized set likelihood. Combine it
    with gold_set_nll to retain coherent set / cardinality supervision.
    """
    import torch.nn.functional as F

    if scores.ndim != 1 or previous_set.shape != scores.shape or target_set.shape != scores.shape:
        raise ValueError("scores/previous_set/target_set must be equal-length vectors")
    if not bool(torch.all((previous_set == 0) | (previous_set == 1))):
        raise ValueError("previous set must be binary")
    if not bool(torch.all((target_set == 0) | (target_set == 1))):
        raise ValueError("target set must be binary")
    if not math.isfinite(unchanged_weight) or not 0.0 <= unchanged_weight <= 1.0:
        raise ValueError("unchanged_weight must be finite in [0,1]")
    changed = (previous_set != target_set).to(scores.dtype)
    weights = changed + unchanged_weight * (1.0 - changed)
    return (
        F.binary_cross_entropy_with_logits(
            scores, target_set.to(scores.dtype), reduction="none"
        ) * weights
    ).mean()


def revision_nll(
    scores: Tensor,
    correction: Tensor,
    prev: Tensor,
    new: Tensor,
    allowed: Sequence[int],
    *,
    normalize: bool = True,
    unchanged_weight: float = 0.0,
) -> Tensor:
    """Exact set NLL PLUS auxiliary change-weighted membership BCE.

    This is not a 'pure revision likelihood'. Even with unchanged_weight=0,
    the exact set NLL backpropagates into *all* scores via the partition Z;
    the unchanged candidates are not inherited or frozen. Likewise
    unchanged_weight=1 produces set NLL + BCE, NOT plain re-prediction.
    """
    aux = revision_auxiliary_bce(
        scores, prev, new, unchanged_weight=unchanged_weight
    )
    base = gold_set_nll(scores, correction, new, allowed, normalize=True)
    combined = base + aux
    return combined if normalize else combined * scores.numel()


@torch.no_grad()
def set_map(scores: Tensor, correction: Tensor, allowed: Sequence[int]) -> dict:
    """Exact MAP set: compare top-m sums + cardinality potential, NOT count mode."""
    if scores.ndim != 1 or correction.ndim != 1:
        raise ValueError("expected vectors")
    k = scores.numel()
    support = sorted(set(int(m) for m in allowed))
    if not support or support[0] < 0 or support[-1] > k or correction.numel() < k + 1:
        raise ValueError("invalid allowed counts")
    order = sorted(range(k), key=lambda i: (-float(scores[i]), i))
    prefix = [0.0]
    for i in order:
        prefix.append(prefix[-1] + float(scores[i]))
    m_best = max(support, key=lambda m: (prefix[m] + float(correction[m]), -m))
    lognorm = set_log_partition(scores, correction, support)
    center = scores.detach().max()
    log_z = torch_log_partitions(scores - center)
    indices = torch.tensor(support, device=scores.device)
    count_prob = torch.softmax(
        log_z[indices] + center * indices + correction[indices], dim=0
    )
    rho = [0.0] * (k + 1)
    for m, p in zip(support, count_prob.tolist()):
        rho[m] = float(p)
    return {
        "indices": sorted(order[:m_best]),
        "count_probs": rho,
        "set_log_probability": prefix[m_best]
        + float(correction[m_best])
        - float(lognorm),
    }


class RecurrentSetDecision(nn.Module):
    """Separate optional demonstration memory and repeated decision workspace.

    Inputs:
      context: B x H frozen backbone answer-slot vectors
      native_logits: B x K padded original option-letter logits
      mask: B x K boolean validity mask (all valid positions are prefix)
      demonstrations: optional B x D x H feature vectors, never gold answers
      demonstration_mask: optional B x D boolean mask

    The initial step (r=0) is EXACTLY the native BCE set distribution
    conditioned on allowed counts, since residual cardinality potential is 0.
    New textual CoT tokens are never generated.
    """

    def __init__(self, config: RecurrentSetConfig):
        super().__init__()
        self.config = config
        d = config.latent_dim
        self.context_proj = nn.Sequential(
            nn.Linear(config.context_dim, d), nn.LayerNorm(d)
        )
        self.demo_proj = nn.Sequential(nn.Linear(config.context_dim, d), nn.Tanh())
        self.memory_cell = nn.GRUCell(d, d)
        self.init_state = nn.Sequential(nn.Linear(2 * d, d), nn.Tanh())
        self.position_emb = nn.Embedding(config.max_options, d)
        self.candidate_proj = nn.Sequential(nn.Linear(4, d), nn.Tanh())
        self.state_cell = nn.GRUCell(d, d)
        # Diagnostic unshared feed-forward control ("mlp"): new layer each
        # step and no carried latent state, but still feeds back set scores.
        # This control is NOT parameter- or FLOP-matched to the GRU and cannot
        # isolate recurrence without additional compute-matched experiments.
        self.state_stack = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, d), nn.Tanh()) for _ in range(config.max_steps)]
        )
        self.score_norm = nn.LayerNorm(d)
        self.score_delta = nn.Linear(d, 1)
        self.count_delta = nn.Linear(d, config.max_options + 1)

        # Start exactly at the trained native option-logit baseline.
        nn.init.zeros_(self.score_delta.weight)
        nn.init.zeros_(self.score_delta.bias)
        nn.init.zeros_(self.count_delta.weight)
        nn.init.zeros_(self.count_delta.bias)

    def encode_memory(
        self,
        context: Tensor,
        demonstrations: Optional[Tensor],
        demonstration_mask: Optional[Tensor],
    ) -> Tensor:
        b, _ = context.shape
        d = self.config.latent_dim
        memory = context.new_zeros((b, d))
        if demonstrations is None:
            if demonstration_mask is not None:
                raise ValueError("demo mask supplied without demonstrations")
            return memory
        if (
            demonstrations.ndim != 3
            or demonstrations.shape[0] != b
            or demonstrations.shape[-1] != self.config.context_dim
        ):
            raise ValueError("demonstrations must be B x D x context_dim")
        if (
            demonstration_mask is not None
            and demonstration_mask.shape != demonstrations.shape[:2]
        ):
            raise ValueError("demonstration mask shape mismatch")
        for t in range(demonstrations.shape[1]):
            proposal = self.memory_cell(self.demo_proj(demonstrations[:, t]), memory)
            if demonstration_mask is None:
                memory = proposal
            else:
                memory = torch.where(
                    demonstration_mask[:, t, None].bool(), proposal, memory
                )
        return memory

    def forward(
        self,
        context: Tensor,
        native_logits: Tensor,
        mask: Tensor,
        *,
        steps: int = 4,
        feedback: Optional[str] = None,
        demonstrations: Optional[Tensor] = None,
        demonstration_mask: Optional[Tensor] = None,
    ) -> List[dict]:
        if steps < 0 or steps > 64:
            raise ValueError("steps must be 0..64")
        if context.ndim != 2 or context.shape[-1] != self.config.context_dim:
            raise ValueError("context must be B x context_dim")
        if native_logits.ndim != 2 or native_logits.shape[0] != context.shape[0]:
            raise ValueError("native_logits must be B x K")
        b, k = native_logits.shape
        if not 2 <= k <= self.config.max_options or mask.shape != native_logits.shape:
            raise ValueError("invalid candidate mask or candidate count")
        mask = mask.bool()
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every question needs candidates")
        if not bool((mask.long().cummin(dim=1).values == mask.long()).all()):
            raise ValueError("candidate mask must be prefix-valid")
        if (
            not torch.isfinite(context).all()
            or not torch.isfinite(native_logits[mask]).all()
        ):
            raise ValueError("nonfinite model features")
        mode = feedback or self.config.feedback
        if mode not in ("set", "latent_only", "static", "mlp"):
            raise ValueError("unknown feedback mode")
        if mode == "mlp" and steps > self.config.max_steps:
            raise ValueError("mlp mode requires steps <= max_steps")

        context = context.float()
        native = torch.where(mask, native_logits.float(), 0.0)
        question = self.context_proj(context)
        memory = self.encode_memory(context, demonstrations, demonstration_mask)
        initial = self.init_state(torch.cat((question, memory), dim=-1))
        latent = initial
        scores = native.clone()
        correction = torch.zeros(
            b, self.config.max_options + 1, device=context.device, dtype=torch.float32
        )
        history = [{"scores": scores, "correction": correction, "latent": latent}]
        positions = self.position_emb(torch.arange(k, device=context.device))[None]
        for r in range(steps):
            incoming = scores if mode in ("set", "mlp") else native
            p = torch.sigmoid(incoming)
            # Scaled logits limit numerical dominance, not output logits.
            evidence = torch.stack(
                (incoming / 10.0, p, p * (1 - p), (incoming - native) / 10.0), dim=-1
            )
            candidate = self.candidate_proj(evidence) + positions
            reference = initial if mode == "static" else latent
            attn = (candidate * reference[:, None]).sum(dim=-1) / math.sqrt(
                self.config.latent_dim
            )
            attn = attn.masked_fill(~mask, -torch.inf)
            attn = torch.softmax(attn, dim=1)
            summary = (candidate * attn[..., None]).sum(dim=1)
            if mode == "mlp":
                next_latent = self.state_stack[r](question + memory + summary)
            else:
                next_latent = self.state_cell(question + memory + summary, reference)
            update = self.score_norm(candidate + next_latent[:, None])
            delta = self.score_delta(torch.tanh(update)).squeeze(-1)
            scores = scores + torch.where(mask, delta, 0.0)
            correction = self.count_delta(next_latent)
            latent = initial if mode == "static" else next_latent
            history.append(
                {"scores": scores, "correction": correction, "latent": next_latent}
            )
        return history


def loss_over_steps(
    history: Sequence[dict],
    gold: Tensor,
    mask: Tensor,
    allowed: Sequence[Sequence[int]],
    *,
    step_supervision: str = "all",
    monotonic_weight: float = 0.0,
    revision_previous: Optional[Tensor] = None,
    revision_weight: float = 0.0,
    unchanged_weight: float = 0.0,
) -> Tensor:
    """Exact-set NLL plus optional change-weighted revision auxiliary.

    revision_previous is the prior predicted/action set for the SAME problem or
    the immediately preceding environment state, not the oracle gold answer.
    The revision auxiliary supervises only r>=1, because r=0 is the frozen
    native readout. The exact-set NLL always remains in the objective.
    """
    if step_supervision not in ("all", "last"):
        raise ValueError("step_supervision must be all or last")
    if not history or gold.ndim != 2 or len(allowed) != gold.shape[0] or mask.shape != gold.shape:
        raise ValueError("history/gold/mask/allowed mismatch")
    if not math.isfinite(monotonic_weight) or monotonic_weight < 0:
        raise ValueError("monotonic_weight must be finite and nonnegative")
    if not math.isfinite(revision_weight) or revision_weight < 0:
        raise ValueError("revision_weight must be finite and nonnegative")
    if revision_weight:
        if len(history) < 2 or revision_previous is None or revision_previous.shape != gold.shape:
            raise ValueError("revision training requires steps>=1 and previous sets matching gold")
    losses, auxiliary = [], []
    for step_index, step in enumerate(history):
        per_row, changes = [], []
        for i, counts in enumerate(allowed):
            k = int(mask[i].sum().item())
            scores = step["scores"][i, :k]
            per_row.append(
                gold_set_nll(
                    scores, step["correction"][i, : k + 1], gold[i, :k], counts
                )
            )
            if revision_weight and step_index > 0:
                changes.append(
                    revision_auxiliary_bce(
                        scores,
                        revision_previous[i, :k],
                        gold[i, :k],
                        unchanged_weight=unchanged_weight,
                    )
                )
        losses.append(torch.stack(per_row))
        if changes:
            auxiliary.append(torch.stack(changes).mean())
    selected = losses if step_supervision == "all" else losses[-1:]
    objective = torch.stack(selected).mean()
    if auxiliary:
        used = auxiliary if step_supervision == "all" else auxiliary[-1:]
        objective = objective + revision_weight * torch.stack(used).mean()
    if monotonic_weight > 0 and len(losses) > 1:
        regressions = [
            torch.relu(losses[i + 1] - losses[i]).mean()
            for i in range(len(losses) - 1)
        ]
        objective = objective + monotonic_weight * torch.stack(regressions).mean()
    return objective


def config_as_dict(model: RecurrentSetDecision) -> dict:
    return asdict(model.config)
