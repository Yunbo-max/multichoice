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
        self.model = cls.from_pretrained(
            checkpoint,
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
        self.backbone = getattr(self.model.model, "language_model", self.model.model)
        head = self.model.get_output_embeddings()
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


def replay(gen, encoder, agent, episode):
    """On-policy replay: the model's own action drives the memory, gold drives the loss."""
    active = _initial(gen, episode)
    memory = agent.empty_memory(1, device=encoder.device)
    demos = encoder.encode_demonstrations(episode.get("demonstrations", []))
    if demos.shape[1]:
        memory = agent.prime(memory, demos)
    losses = []
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
        option_features = encoder.encode_option_meanings(row)
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
    return torch.stack(losses).mean()


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
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--accumulate-episodes", type=int, default=2)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = p.parse_args()
    try:
        import gen_text_episodes_shift as gen
    except ModuleNotFoundError:
        from harness import gen_text_episodes_shift as gen

    train = [json.loads(l) for l in args.train_jsonl.open() if l.strip()]
    dev = [json.loads(l) for l in args.dev_jsonl.open() if l.strip()]
    for ep in train + dev:
        gen.validate_episode(ep)

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
    params = list(encoder.parameters()) + list(agent.parameters())
    optimizer = torch.optim.AdamW([p for p in params if p.requires_grad], lr=args.lr)
    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        rng.shuffle(train)
        agent.train()
        total = 0.0
        group = []
        for index, episode in enumerate(train):
            loss = replay(gen, encoder, agent, episode)
            group.append(loss)
            if len(group) == args.accumulate_episodes:
                (torch.stack(group).mean()).backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in params if p.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total += sum(float(x.detach()) for x in group)
                group = []
        if group:
            (torch.stack(group).mean()).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total += sum(float(x.detach()) for x in group)
        print(json.dumps({"epoch": epoch + 1, "train_nll": total / len(train)}))
        torch.save(
            {
                "schema": "episodic-set-head-v1",
                "config": {
                    "context_dim": encoder.hidden_size,
                    "latent_dim": args.latent_dim,
                    "max_options": len(gen.CELLS),
                    "inner_steps": args.inner_steps,
                },
                "model": {k: v.detach().cpu() for k, v in agent.state_dict().items()},
                "lora": {
                    k: v.detach().cpu()
                    for k, v in encoder.model.state_dict().items()
                    if "lora" in k
                },
                "model_identity": "BCE_S44_FINAL",
            },
            args.output / "best.pt",
        )


if __name__ == "__main__":
    main()
