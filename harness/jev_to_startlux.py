"""Convert JEVal rows into StartLux ``/v1/systemone`` requests.

Input is the normalised JEVal file produced by ``jev_to_sii.py`` (id/type/state/question/options/
gold_index).  Output:

  startlux_requests.jsonl   one line per item: {"id", "type", "request": {...}}
                            ``request`` is the exact POST body for ``/v1/systemone``: a string
                            ``state`` and one question named ``q``.
  startlux_sidecar.json     id -> {type, question, options, gold_index, ...}.  ``options`` keeps
                            the original option order/keys so predictions can be mapped back and
                            scored by ``score_jeval.py``.
  startlux_summary.json     conversion statistics and schema-validation results.

Conversions
  choice  criteria = {option key: description or null}, order preserved.
  noul    criteria = {"true": desc, "false": desc} (the API answers P(yes) = P(true));
          both keys are always present, whatever the original order was.
  score   criteria = ordered list of levels (lowest first).  JEVal keys are "0".."n-1".

Descriptions that the SII conversion serialised from JSON ``null`` are sent as ``null`` (the API
then shows the option id alone).  ``None``/""/"null" all become null.

The script also validates every question against the official package's renderer
(``startlux_decide.prompt.from_systemone`` + ``validate_row``) without touching a GPU, and can
estimate prefill tokens (``--token-stats``) including the wide-list knock-out.

Usage:
  python -I jev_to_startlux.py \
      --input ../data/jeval/sii_input.jsonl --out-dir ../data/jeval/startlux --token-stats
"""
import argparse
import json
from collections import Counter
from pathlib import Path


QUESTION_KEY = "q"
# The server adds an internal "None of the above" option by default (none_check=true).  A choice
# list then has to leave a letter free for it, so lists of more than 25 options are knocked out.
MAX_OPTIONS_PER_PASS = 26
WIDE_GROUP = 25
WIDE_KEEP = 3
NONE_CHECK = True
NONE_OPTION = "None of the above"


def none_check_for(row):
    """Whether to ask for the internal "None of the above" check on this item.

    The server refuses a choice question that also has its own option id "None of the above"
    (11 JEVal decider rows: ids such as ``none of the above``).  For those the check is turned
    off; scoring never uses ``none_probability``.  Everything else keeps the release default.
    """
    if row["type"] != "choice":
        return NONE_CHECK       # ignored by noul/score
    import re
    for option in row["options"]:
        norm = re.sub(r"[\s_\-]+", " ", str(option["key"])).strip().lower()
        if norm == NONE_OPTION.lower():
            return False
    return NONE_CHECK


def option_desc(value):
    """An option description, or None when the source had no description."""
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = text.strip()
    if not text or text.lower() == "null":
        return None
    return text


def is_true_key(key):
    return str(key).strip().lower() in ("true", "yes")


def spec_for(row):
    """The /v1/systemone question spec for one normalised JEVal row."""
    qtype = row["type"]
    options = row["options"]
    instructions = row["question"]

    if qtype == "choice":
        criteria = {}
        for option in options:
            key = str(option["key"])
            if key in criteria:
                raise ValueError(f"duplicate option key {key!r}")
            criteria[key] = option_desc(option["desc"])
        return {"type": "choice", "instructions": instructions, "criteria": criteria}

    if qtype == "noul":
        by_key = {str(o["key"]).strip().lower(): option_desc(o["desc"]) for o in options}
        if set(by_key) != {"true", "false"}:
            raise ValueError(f"noul options must be true/false, got {sorted(by_key)}")
        return {"type": "noul", "instructions": instructions,
                "criteria": {"true": by_key["true"], "false": by_key["false"]}}

    if qtype == "score":
        keys = [str(o["key"]) for o in options]
        if keys != [str(i) for i in range(len(keys))]:
            raise ValueError(f"score keys must be 0..n-1 in order, got {keys}")
        if not 2 <= len(options) <= 10:
            raise ValueError(f"score needs 2..10 levels, got {len(options)}")
        return {"type": "score", "instructions": instructions,
                "criteria": [option_desc(o["desc"]) for o in options]}

    raise ValueError(f"unknown question type {qtype!r}")


def sidecar_entry(row, index):
    qtype = row["type"]
    options = [{"key": str(o["key"]), "desc": option_desc(o["desc"])} for o in row["options"]]
    gold = int(row["gold_index"])
    if not 0 <= gold < len(options):
        raise ValueError(f"gold index {gold} outside 0..{len(options) - 1}")
    return {
        "id": row["id"],
        "type": qtype,
        "question_key": QUESTION_KEY,
        "question": row["question"],
        "options": options,
        "n_options": len(options),
        "gold_index": gold,
        "gold_option_key": options[gold]["key"],
        "source": row["source"],
        "source_item_id": row["source_item_id"],
        "request_index": index,
    }


# ------------------------------------------------------------------ validation

def _validate_spec(state, spec, allow_none, from_systemone, validate_row):
    return validate_row(from_systemone(state, spec, allow_none=allow_none))


def validate_request(state, spec, from_systemone, validate_row, none_check=NONE_CHECK):
    """Validate a question the way the pipeline prepares it: the full list, or every knock-out
    group plus a final round when the list is too wide for one pass."""
    qtype = spec.get("type")
    if qtype == "choice":
        criteria = spec["criteria"]
        if len(criteria) > MAX_OPTIONS_PER_PASS - int(none_check):
            from startlux_decide.readout import knockout_groups
            groups = knockout_groups(list(criteria), WIDE_GROUP)
            for group in groups:
                _validate_spec(state, dict(spec, criteria={k: criteria[k] for k in group}), False,
                               from_systemone, validate_row)
            finalists = [k for group in groups for k in group[:WIDE_KEEP]][:MAX_OPTIONS_PER_PASS - 1]
            _validate_spec(state, dict(spec, criteria={k: criteria[k] for k in finalists}), True,
                           from_systemone, validate_row)
            return len(groups) + 1
        _validate_spec(state, spec, bool(none_check), from_systemone, validate_row)
        return 1
    _validate_spec(state, spec, False, from_systemone, validate_row)
    return 1


# ------------------------------------------------------------------ token estimate

def estimate_prompt_tokens(row, spec, tokenizer, from_systemone, messages, plain_prompt, count_tokens,
                           knockout_groups, none_check=NONE_CHECK):
    """Approximate prefill tokens of a question: (total over all passes, largest single pass).

    The state is encoded once; each pass adds the chat-template overhead and its option block,
    which is measured by rendering the same row with a one-token state.  This avoids encoding a
    long state once per knock-out round (a 78K-token state can be read 12 times for a 255-option
    list).  The result is within a couple of tokens per pass of the exact count.
    """
    state = spec["state"]
    state_tokens = count_tokens(tokenizer, state)

    def one_pass(criteria, allow_none):
        probe = dict(spec, state="x", criteria=criteria)
        text = plain_prompt(tokenizer, messages(from_systemone(probe, _spec_only(probe), allow_none=allow_none)))
        return state_tokens + count_tokens(tokenizer, text) - 1

    if spec["type"] == "choice":
        criteria = spec["criteria"]
        if len(criteria) > MAX_OPTIONS_PER_PASS - int(none_check):
            groups = knockout_groups(list(criteria), WIDE_GROUP)
            per_pass = [one_pass({k: criteria[k] for k in group}, False) for group in groups]
            finalists = [k for group in groups for k in group[:WIDE_KEEP]][:MAX_OPTIONS_PER_PASS - 1]
            per_pass.append(one_pass({k: criteria[k] for k in finalists}, True))
            return sum(per_pass), max(per_pass)
        n = one_pass(criteria, bool(none_check))
        return n, n
    n = one_pass(spec["criteria"], False)
    return n, n


def _spec_only(probe):
    """The probe passed to from_systemone already carries type/instructions/criteria."""
    return {k: probe[k] for k in ("type", "instructions", "criteria")}


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/jeval/sii_input.jsonl")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/jeval/startlux")
    parser.add_argument("--token-stats", action="store_true",
                        help="estimate prefill tokens with the model tokenizer (CPU only)")
    parser.add_argument("--model", default=None,
                        help="model folder for --token-stats (default: models/StartLux-4B-AutoThink)")
    parser.add_argument("--validate-only", action="store_true",
                        help="validate an existing startlux_requests.jsonl instead of writing one")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from startlux_decide.prompt import (count_tokens, from_systemone, messages, plain_prompt,
                                            validate_row)
        from startlux_decide.readout import knockout_groups
    except Exception as error:  # noqa: BLE001
        raise SystemExit(f"cannot import startlux_decide (install it in the vLLM environment): {error}")

    if args.validate_only:
        path = args.out_dir / "startlux_requests.jsonl"
        entries = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        requests = [(e["request"]["state"], e["request"]["questions"][QUESTION_KEY],
                     bool(e["request"].get("none_check", NONE_CHECK))) for e in entries]
        ids = [e["id"] for e in entries]
    else:
        requests, ids = [], []
        for row in rows:
            requests.append((row["state"], spec_for(row), none_check_for(row)))
            ids.append(row["id"])

    errors, passes = [], Counter()
    for ident, (state, spec, none_check) in zip(ids, requests):
        try:
            passes[validate_request(state, spec, from_systemone, validate_row, none_check)] += 1
        except Exception as error:  # noqa: BLE001
            errors.append({"id": ident, "error": str(error)})
    if errors:
        print(json.dumps({"invalid_requests": len(errors), "errors": errors[:10]}, ensure_ascii=False, indent=2))
        raise SystemExit(f"{len(errors)} request(s) failed schema validation")

    token_stats = None
    if args.token_stats:
        import os
        from startlux_decide.modelcfg import load_tokenizer
        root = Path(__file__).resolve().parents[1]
        model = args.model or str(root / "models/StartLux-4B-AutoThink")
        tokenizer = load_tokenizer(model)
        per_item, total, max_single = {}, 0, 0
        for ident, (state, spec, none_check) in zip(ids, requests):
            full = dict(spec, state=state)
            n, peak = estimate_prompt_tokens(full, full, tokenizer, from_systemone, messages, plain_prompt,
                                             count_tokens, knockout_groups, none_check)
            per_item[ident] = n
            total += n
            max_single = max(max_single, peak)
        token_stats = {"items": len(per_item), "total_prompt_tokens_estimate": total,
                       "max_total_prompt_tokens_per_item_estimate": max(per_item.values()),
                       "max_single_pass_tokens_estimate": max_single,
                       "mean_prompt_tokens_estimate": total / len(per_item),
                       "model": model}

    if args.validate_only:
        print(json.dumps({"validated": len(requests), "passes_histogram": dict(sorted(passes.items())),
                          "token_stats": token_stats}, ensure_ascii=False, indent=2))
        return

    requests_path = args.out_dir / "startlux_requests.jsonl"
    sidecar_path = args.out_dir / "startlux_sidecar.json"
    sidecar = {}
    with requests_path.open("w", encoding="utf-8") as stream:
        for index, (row, (state, spec, none_check)) in enumerate(zip(rows, requests)):
            entry = sidecar_entry(row, index)
            entry["none_check"] = none_check
            sidecar[row["id"]] = entry
            line = {"id": row["id"], "type": row["type"],
                    "request": {"state": state, "questions": {QUESTION_KEY: spec},
                                # explicit so the run does not depend on serve flags
                                "none_check": none_check, "allow_none": False}}
            stream.write(json.dumps(line, ensure_ascii=False) + "\n")
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    summary = {
        "rows_in": len(rows),
        "rows_out": len(sidecar),
        "types": dict(Counter(r["type"] for r in rows)),
        "option_counts": dict(sorted(Counter(len(r["options"]) for r in rows).items())),
        "max_options": max(len(r["options"]) for r in rows),
        "wide_lists": sum(r["type"] == "choice" and len(r["options"]) > 25 for r in rows),
        "passes_histogram": dict(sorted(passes.items())),
        "schema_validation": "ok",
        "none_check": NONE_CHECK,
        "none_check_off_items": sum(1 for r in rows if r["type"] == "choice"
                                    and not none_check_for(r)),
        "token_stats": token_stats,
        "outputs": {"requests": str(requests_path), "sidecar": str(sidecar_path)},
    }
    summary_path = args.out_dir / "startlux_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
