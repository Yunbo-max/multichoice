#!/usr/bin/env python3
"""C experiment: on-policy episodic training with a LoRA-trainable Qwen backbone.

Unlike ``train_episodic_features.py`` (frozen Qwen, cached features), this loads
Qwen with LoRA adapters and computes answer-slot features on the fly, so the
backbone learns the dynamic decision representation jointly with the persistent
memory head. Training uses the model's OWN executed actions for the memory update
and gold labels only for the loss (DAgger-style on-policy aggregation).

Input is raw episode JSONL (from ``gen_text_episodes_shift.py``), not a cached
feature tensor, because the Qwen features change every step during training.
"""

from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from episodic_set_decision import EpisodicSetConfig, EpisodicSetDecision
from recurrent_set_decision import gold_set_nll, set_map
from set_distillation import allowed_counts
from episodic_lora_checkpoint import check_lora_gradients, lora_tensors


class TrainableQwenEncoder:
    """LoRA Qwen decision encoder (on-the-fly, gradients flow through it)."""

    def __init__(self, checkpoint, *, modules, lora_r, lora_alpha, device, dtype):
        from transformers import AutoConfig, AutoTokenizer
        from transformers import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
        from peft import LoraConfig, get_peft_model

        try:
            from train_multiselect import option_labels, make_prompt
        except ModuleNotFoundError:
            from harness.train_multiselect import option_labels, make_prompt

        self.device = torch.device(device)
        self.checkpoint_path = str(checkpoint)
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_modules = list(modules)
        self.dtype_name = dtype
        self.make_prompt = make_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.labels = option_labels(self.tokenizer)
        self.label_ids = [
            self.tokenizer.encode(s, add_special_tokens=False)[0] for s in self.labels
        ]
        cfg = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        if dtype == "bf16" and getattr(cfg, "mamba_ssm_dtype", None) == "float32":
            cfg.mamba_ssm_dtype = "bfloat16"
        self.model_ssm_dtype = str(getattr(cfg, "mamba_ssm_dtype", None))
        cls = (
            Qwen3_5ForConditionalGeneration
            if cfg.model_type == "qwen3_5"
            else Qwen3_5ForCausalLM
        )
        resolved = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[dtype]
        # The previous implementation loaded 4B twice and discarded the
        # SSM dtype override from its first load. Keep exactly one load.
        self.model = cls.from_pretrained(
            checkpoint,
            config=cfg,
            dtype=resolved,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self.model.requires_grad_(False)
        self.model = get_peft_model(
            self.model,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.0,
                target_modules=modules,
                bias="none",
            ),
        )
        self.model.to(self.device)
        base = self.model.model if hasattr(self.model, "model") else self.model
        self.backbone = getattr(base.model, "language_model", base.model)
        head = base.get_output_embeddings()
        self.label_weight = head.weight.index_select(
            0, torch.tensor(self.label_ids, device=self.device, dtype=torch.long)
        ).float()
        self.hidden_size = int(self.label_weight.shape[1])

    def _slot(self, prompts):
        encoded = self.tokenizer(
            prompts, padding=True, add_special_tokens=False, return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        hidden = self.backbone(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        index = encoded["attention_mask"].sum(-1) - 1
        return hidden[torch.arange(len(prompts), device=self.device), index].float()

    def _text(self, text):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def encode_question(self, row):
        context = self._slot([self.make_prompt(self.tokenizer, row, self.labels)])
        scores = F.linear(context, self.label_weight)[:, : len(row["options"])]
        mask = torch.ones_like(scores, dtype=torch.bool)
        return context, scores, mask

    def encode_observation(self, text):
        return self._slot(
            [self._text("Observed environment feedback after action:\n" + text)]
        )

    def encode_demonstrations(self, observations):
        if not observations:
            return torch.zeros(1, 0, self.hidden_size, device=self.device)
        return self._slot(
            [self._text("Known demonstration:\n" + o) for o in observations]
        ).unsqueeze(0)

    def encode_option_meanings(self, row):
        return self._slot(
            [
                self._text("Possible action (semantic meaning only):\n" + o["desc"])
                for o in row["options"]
            ]
        ).unsqueeze(0)

    def parameters(self):
        return self.model.parameters()


def replay(gen, encoder, agent, episode, *, return_stats=False):
    """On-policy replay: model actions drive the memory, gold drives the loss.

    return_stats=True is intended for inference-time DEV evaluation, where
    executed action set accuracy and NLL must both be measured. Existing
    training calls keep returning a scalar differentiable mean loss.
    """
    active = _initial(gen, episode)
    memory = agent.empty_memory(1, device=encoder.device)
    demos = encoder.encode_demonstrations(episode.get("demonstrations", []))
    if demos.shape[1]:
        memory = agent.prime(memory, demos)
    # Option meanings are constant across the whole episode: encode once, reuse
    # every step (the on-policy loop is otherwise dominated by redundant forwards).
    option_features = encoder.encode_option_meanings(episode["steps"][0]["decision"])
    losses = []
    correct_steps = 0
    all_correct = True
    k = len(gen.CELLS)
    for t, oracle in enumerate(episode["steps"]):
        rule = oracle["oracle_rule_before_decision"]
        row = dict(oracle["decision"])
        row["state"] = gen.state_text(active)
        context, logits, mask = encoder.encode_question(row)
        final = agent.decide(memory, context, logits, mask)[-1]
        support = allowed_counts(row)
        gold = gen.shift_set(active, rule)
        gold_t = torch.tensor(gold, dtype=torch.float32, device=encoder.device)
        losses.append(
            gold_set_nll(
                final["scores"][0], final["correction"][0, : k + 1], gold_t, support
            )
        )
        # model's own action (decoded), not the oracle
        idx = set_map(final["scores"][0], final["correction"][0, : k + 1], support)[
            "indices"
        ]
        executed = [int(i in idx) for i in range(k)]
        correct = executed == gold
        correct_steps += int(correct)
        all_correct = all_correct and correct
        feedback = (
            "After this decision the active components are "
            + gen.active_text(executed)
            + "."
        )
        if t + 1 < len(episode["steps"]):
            following = episode["steps"][t + 1]["oracle_rule_before_decision"]
            if following != rule:
                feedback += " The rule changed. " + gen.demo_text(following)
        obs = encoder.encode_observation(feedback)
        action = torch.tensor(executed, dtype=torch.float32, device=encoder.device)[
            None
        ]
        memory = agent.observe(
            memory,
            context,
            action,
            mask,
            obs,
            option_features=option_features,
            done=torch.tensor(
                [t == len(episode["steps"]) - 1],
                dtype=torch.bool,
                device=encoder.device,
            ),
        )
        active = executed
    avg_nll = torch.stack(losses).mean()
    if return_stats:
        return {
            "loss": avg_nll,
            "step_em": correct_steps / len(losses),
            "trajectory_em": float(all_correct),
            "n_steps": len(losses),
        }
    return avg_nll


@torch.no_grad()
def evaluate_onpolicy_dev(gen, encoder, agent, episodes):
    """Choose checkpoints on actual on-policy DEV states, never training loss.

    This is intentionally a bounded fixed development subset for expensive
    4B + LoRA pilots. For an official result, run a larger locked DEV split,
    then one untouched TEST evaluation using the selected checkpoint.
    """
    if not episodes:
        raise ValueError("dev episodes cannot be empty")
    encoder.model.eval()
    agent.eval()
    losses, correct, trajectories, total_steps = 0.0, 0.0, 0.0, 0
    for episode in episodes:
        report = replay(gen, encoder, agent, episode, return_stats=True)
        losses += float(report["loss"].detach()) * report["n_steps"]
        correct += report["step_em"] * report["n_steps"]
        trajectories += report["trajectory_em"]
        total_steps += report["n_steps"]
    return {
        "step_em": correct / total_steps,
        "trajectory_em": trajectories / len(episodes),
        "normalized_set_nll": losses / total_steps,
        "n_episodes": len(episodes),
        "n_steps": total_steps,
    }


def _initial(gen, episode):
    line = episode["steps"][0]["decision"]["state"]
    payload = line[len("The active components are: ") : -1]
    labels = [] if payload == "none" else payload.split(", ")
    return [int(cell in labels) for cell in gen.CELLS]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--dev-jsonl", type=Path, required=True)
    p.add_argument("--qwen-model", type=Path, required=True)
    p.add_argument("--modules", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--inner-steps", type=int, default=4)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="0 = all episodes")
    p.add_argument("--dev-eval-episodes", type=int, default=12,
                   help="fixed DEV subset for checkpoint selection; 0 = entire DEV")
    p.add_argument("--model-identity", default="BCE_S44_FINAL",
                   help="identifier for the exact pretrained decision model")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--accumulate-episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = p.parse_args()
    if args.limit < 0 or args.dev_eval_episodes < 0 or args.epochs < 1:
        p.error("limit/eval-limit must be nonnegative; epochs must be positive")
    if args.accumulate_episodes < 1 or args.grad_clip <= 0 or args.lr <= 0:
        p.error("invalid gradient accumulation, clipping or learning rate")
    if args.lora_r < 1 or args.lora_alpha <= 0:
        p.error("LoRA rank and alpha must be positive")
    try:
        import gen_text_episodes_shift as gen
    except ModuleNotFoundError:
        from harness import gen_text_episodes_shift as gen

    train = [json.loads(l) for l in args.train_jsonl.open() if l.strip()]
    if args.limit:
        train = train[: args.limit]
    dev = [json.loads(l) for l in args.dev_jsonl.open() if l.strip()]
    for ep in train + dev:
        gen.validate_episode(ep)
    if not train or not dev:
        raise ValueError("train and dev episodes must be nonempty")
    train_ids = [ep["episode_id"] for ep in train]
    dev_ids = [ep["episode_id"] for ep in dev]
    if (len(set(train_ids)) != len(train_ids)
        or len(set(dev_ids)) != len(dev_ids)
        or set(train_ids) & set(dev_ids)):
        raise ValueError("train/dev episode IDs must be unique and disjoint")
    rng = random.Random(args.seed)
    dev_rng = random.Random(args.seed + 100_003)
    dev_eval = (
        dev_rng.sample(dev, min(len(dev), args.dev_eval_episodes))
        if args.dev_eval_episodes else dev
    )

    modules = json.loads(args.modules.read_text())
    encoder = TrainableQwenEncoder(
        args.qwen_model,
        modules=modules,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        device=args.device,
        dtype=args.dtype,
    )
    config = EpisodicSetConfig(
        context_dim=encoder.hidden_size,
        latent_dim=args.latent_dim,
        max_options=len(gen.CELLS),
        inner_steps=args.inner_steps,
    )
    agent = EpisodicSetDecision(config).to(encoder.device)
    trainable_params = [
        p for p in list(encoder.parameters()) + list(agent.parameters())
        if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=False)
    best = None
    best_epoch = 0
    gradient_report = None
    for epoch in range(args.epochs):
        rng.shuffle(train)
        encoder.model.train()
        agent.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        group = []

        def take_optimizer_step(losses):
            nonlocal gradient_report
            if not losses:
                return
            loss_batch = torch.stack(losses).mean()
            if not torch.isfinite(loss_batch).item():
                raise FloatingPointError("nonfinite episodic LoRA training loss")
            loss_batch.backward()
            # A backbone LoRA forward which silently bypasses adapters may
            # still yield gradients for the tiny memory head. Detect that.
            if gradient_report is None:
                gradient_report = check_lora_gradients(encoder.model)
                print(json.dumps({"first_lora_backward": gradient_report}), flush=True)
            torch.nn.utils.clip_grad_norm_(
                trainable_params, args.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for episode in train:
            loss = replay(gen, encoder, agent, episode)
            group.append(loss)
            total += float(loss.detach())
            if len(group) == args.accumulate_episodes:
                take_optimizer_step(group)
                group.clear()
        if group:
            take_optimizer_step(group)
            group.clear()

        dev_metrics = evaluate_onpolicy_dev(gen, encoder, agent, dev_eval)
        record = {
            "epoch": epoch + 1,
            "train_normalized_set_nll": total / len(train),
            "dev_onpolicy": dev_metrics,
            "dev_selection_metric": "step_em_then_lower_normalized_set_nll",
        }
        print(json.dumps(record), flush=True)
        candidate = (
            dev_metrics["step_em"],
            -dev_metrics["normalized_set_nll"],
        )
        if best is None or candidate > best:
            best = candidate
            best_epoch = epoch + 1
            checkpoint = {
                "schema": "episodic-set-head-v1",
                "checkpoint_type": "qwen3.5_lora_plus_episodic_head",
                "config": {
                    "context_dim": encoder.hidden_size,
                    "latent_dim": args.latent_dim,
                    "max_options": len(gen.CELLS),
                    "inner_steps": args.inner_steps,
                },
                "model": {
                    k: v.detach().cpu().clone()
                    for k, v in agent.state_dict().items()
                },
                "lora": lora_tensors(encoder.model),
                "lora_config": {
                    "r": encoder.lora_r,
                    "alpha": encoder.lora_alpha,
                    "target_modules": encoder.lora_modules,
                    "dtype": encoder.dtype_name,
                    "mamba_ssm_dtype": encoder.model_ssm_dtype,
                    "base_checkpoint": encoder.checkpoint_path,
                },
                "model_identity": args.model_identity,
                "selected_epoch": epoch + 1,
                "train_episodes": len(train),
                "dev_selection": dev_metrics,
                "first_lora_backward": gradient_report,
            }
            tmp_path = args.output / "best.pt.tmp"
            torch.save(checkpoint, tmp_path)
            tmp_path.replace(args.output / "best.pt")
        (args.output / "training.json").write_text(
            json.dumps({
                "selected_epoch": best_epoch,
                "last_epoch": epoch + 1,
                "dev_selection": dev_metrics,
                "train_nll_last": total / len(train),
                "first_lora_backward": gradient_report,
            }, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
