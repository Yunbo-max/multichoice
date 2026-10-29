"""Padding-safe Qwen3.5 visual-action token collation.

Every example's answer cue must immediately follow its own unpadded
multimodal prompt, never another sample's padding. CPU torch only.
"""
from __future__ import annotations


def pack_token_rows(input_ids, attention_mask, *, mm_token_type_ids=None,
                    append_ids=None, pad_token_id=0):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    if not input_ids or len(input_ids) != len(attention_mask):
        raise ValueError("input_ids and attention_mask must have matching nonzero rows")
    if mm_token_type_ids is not None and len(mm_token_type_ids) != len(input_ids):
        raise ValueError("mm token type list length mismatch")
    cue = None
    if append_ids is not None:
        cue = torch.as_tensor(append_ids, dtype=input_ids[0].dtype).flatten()
        if not cue.numel():
            raise ValueError("answer cue must be nonempty")
    ids_out, mask_out, mm_out = [], [], []
    for index, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
        ids = torch.as_tensor(ids).flatten()
        mask = torch.as_tensor(mask).flatten().bool()
        if ids.numel() != mask.numel() or not mask.any():
            raise ValueError(f"malformed or empty input sequence {index}")
        next_ids = ids[mask]
        if cue is not None:
            next_ids = torch.cat((next_ids, cue.to(ids.device, dtype=ids.dtype)))
        ids_out.append(next_ids)
        mask_out.append(torch.ones(next_ids.numel(), dtype=torch.long, device=ids.device))
        if mm_token_type_ids is not None:
            mm = torch.as_tensor(mm_token_type_ids[index]).flatten()
            if mm.numel() != ids.numel():
                raise ValueError(f"mm_token_type_ids length mismatch at index {index}")
            next_mm = mm[mask]
            if cue is not None:
                next_mm = torch.cat((next_mm, torch.zeros_like(cue, dtype=next_mm.dtype,
                                                              device=next_mm.device)))
            mm_out.append(next_mm)
    output = {
        "input_ids": pad_sequence(ids_out, batch_first=True,
                                  padding_value=pad_token_id),
        "attention_mask": pad_sequence(mask_out, batch_first=True, padding_value=0),
    }
    if mm_token_type_ids is not None:
        output["mm_token_type_ids"] = pad_sequence(mm_out, batch_first=True,
                                                  padding_value=0)
    last_valid_positions(output["attention_mask"])
    return output


def append_answer_cue(features, cue_ids, *, pad_token_id=0):
    """Unpad each processor sample, append cue, and right-pad full sequences.

    Accepts existing left or right padding; forwards pixel and grid metadata.
    """
    out = dict(features)
    n = features["input_ids"].shape[0]
    out.update(pack_token_rows(
        [features["input_ids"][i] for i in range(n)],
        [features["attention_mask"][i] for i in range(n)],
        mm_token_type_ids=([features["mm_token_type_ids"][i] for i in range(n)]
                           if "mm_token_type_ids" in features else None),
        append_ids=cue_ids, pad_token_id=pad_token_id,
    ))
    return out


def last_valid_positions(attention_mask):
    """Reject non-right-packed input instead of silently reading padding logits."""
    import torch
    mask = attention_mask.bool()
    if mask.ndim != 2 or mask.shape[1] == 0 or not mask.any(dim=1).all():
        raise ValueError("expected nonempty 2-D text attention mask")
    lengths = mask.long().sum(dim=1)
    expected = torch.arange(mask.shape[1], device=mask.device)[None, :] < lengths[:, None]
    if not torch.equal(mask, expected):
        raise ValueError("tokens are not right-packed; answer position is ambiguous")
    return lengths - 1
