"""Frozen Qwen3.5 decision encoder for an episodic SetJev agent.

Outputs exactly the same answer-slot option logits and last hidden state as
the existing SetJev train_multiselect implementation. Also embeds observed
environment feedback and semantic option descriptions so changing action menus
can be carried across episode steps without treating 'A' as a stable action ID.

This is an inference-time bridge; NO LLM generation, API, or weight updates.
Importing this module does not load transformers or model weights.
"""
from __future__ import annotations

from pathlib import Path

import torch


class FrozenQwenDecisionEncoder:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cuda",
        dtype: str = "bf16",
        max_input_tokens: int = 8192,
    ):
        from transformers import AutoConfig, AutoTokenizer
        from transformers import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
        try:
            from train_multiselect import option_labels, make_prompt
        except ModuleNotFoundError:
            from harness.train_multiselect import option_labels, make_prompt

        if dtype not in ("bf16", "fp16", "fp32"):
            raise ValueError("dtype must be bf16, fp16 or fp32")
        self.device = torch.device(device)
        self.max_input_tokens = max_input_tokens
        self.make_prompt = make_prompt
        root = str(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.labels = option_labels(self.tokenizer)
        ids = [
            self.tokenizer.encode(s, add_special_tokens=False)[0]
            for s in self.labels
        ]
        cfg = AutoConfig.from_pretrained(root, local_files_only=True)
        cls = (Qwen3_5ForConditionalGeneration
               if cfg.model_type == "qwen3_5" else Qwen3_5ForCausalLM)
        resolved = {"bf16": torch.bfloat16, "fp16": torch.float16,
                    "fp32": torch.float32}[dtype]
        model, loading = cls.from_pretrained(
            root, dtype=resolved, attn_implementation="sdpa",
            local_files_only=True, output_loading_info=True,
        )
        if loading.get("missing_keys") or loading.get("mismatched_keys"):
            raise RuntimeError("incomplete base-model export (missing/mismatched keys)")
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.backbone = getattr(self.model.model, "language_model", self.model.model)
        output_head = self.model.get_output_embeddings()
        self.label_weight = output_head.weight.index_select(
            0, torch.tensor(ids, device=self.device, dtype=torch.long)
        ).float()
        self.hidden_size = int(self.label_weight.shape[1])

    @torch.no_grad()
    def _slot_vectors(self, prompts: list[str]) -> torch.Tensor:
        if not prompts:
            raise ValueError("no prompts")
        encoded = self.tokenizer(
            prompts, padding=True, add_special_tokens=False, return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] > self.max_input_tokens:
            raise ValueError("prompt exceeds configured context budget")
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        result = self.backbone(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        index = encoded["attention_mask"].sum(-1) - 1
        return result.last_hidden_state[
            torch.arange(len(prompts), device=self.device), index
        ].float()

    def _text_prompt(self, text: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    @torch.no_grad()
    def encode_question(self, row: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns [1,H] context, [1,K] native scores, [1,K] bool mask."""
        if row.get("selection_mode") != "multi":
            raise ValueError("expected SetJev multi-select decision protocol")
        opts = row.get("options")
        if not isinstance(opts, list) or not 2 <= len(opts) <= len(self.labels):
            raise ValueError("expected 2..255 options")
        prompt = self.make_prompt(self.tokenizer, row, self.labels)
        context = self._slot_vectors([prompt])
        scores = torch.nn.functional.linear(context, self.label_weight)[:, :len(opts)]
        mask = torch.ones_like(scores, dtype=torch.bool)
        return context, scores, mask

    @torch.no_grad()
    def encode_observation(self, observed_feedback: str) -> torch.Tensor:
        """Only feedback received after the executed action; no hidden oracle."""
        return self._slot_vectors([
            self._text_prompt("Observed environment feedback after action:\n" + observed_feedback)
        ])

    @torch.no_grad()
    def encode_demonstrations(self, observations: list[str]) -> torch.Tensor:
        """Encode already visible input-output demonstrations as [1,D,H]."""
        if not observations:
            return torch.zeros(1, 0, self.hidden_size, device=self.device)
        return self._slot_vectors([
            self._text_prompt("Known demonstration:\n" + example)
            for example in observations
        ]).unsqueeze(0)

    @torch.no_grad()
    def encode_option_meanings(self, row: dict) -> torch.Tensor:
        """Returns [1,K,H] aligned to the SAME option order as logits.

        Position-only action encoding is insufficient when a changing menu
        reuses letters A/B/C for different actual actions.
        """
        options = row["options"]
        prompts = [
            self._text_prompt("Possible action (semantic meaning only):\n" + item["desc"])
            for item in options
        ]
        return self._slot_vectors(prompts).unsqueeze(0)
