"""
Evaluate base or fine-tuned model on the DriveLM val split.

Metrics
-------
parse_rate       — % of samples where the model outputs valid JSON
causal_grounded  — % of valid outputs where every causal_factor references
                   a real perception object or prediction subject
planning_match   — approximate % where meta_action aligns with the ground
                   truth planning answer (keyword extraction, not exact match)

Usage
-----
  python eval.py                  # base model, 200 samples
  python eval.py 500              # base model, 500 samples
  python eval.py --ft             # fine-tuned model (checkpoints/lora-final)
  python eval.py --ft 500
"""

import sys
import json
from pathlib import Path

from train.dataset import (
    load_drivelm, train_val_split,
    _load_img, _build_temporal_index, N_FRAMES, FPS,
)
from prompt import build_messages
from model import infer, infer_ft
from output import parse as parse_output

# ── config ────────────────────────────────────────────────────────────────────
USE_FT       = "--ft" in sys.argv
N_SAMPLES    = next((int(a) for a in sys.argv[1:] if a.isdigit()), 200)
ADAPTER_PATH = "checkpoints/lora-final"

# ── planning keyword → meta_action vocab ─────────────────────────────────────
_PLAN_MAP = {
    "stop":               ["stop", "halt", "red light", "come to a stop"],
    "brake":              ["brake", "slow down", "decelerate", "slow"],
    "yield":              ["yield", "give way"],
    "maintain_speed":     ["continue", "maintain", "keep going", "proceed", "move forward"],
    "accelerate":         ["accelerate", "speed up"],
    "turn_left":          ["turn left"],
    "turn_right":         ["turn right"],
    "lane_change_left":   ["change lane to the left", "merge left", "move to the left lane"],
    "lane_change_right":  ["change lane to the right", "merge right", "move to the right lane"],
}


def _gt_action(conversations: list):
    """Extract a meta_action string from the most planning-relevant GT answer, or None."""
    for turn in reversed(conversations):
        if turn["from"] != "gpt":
            continue
        text = turn["value"].lower()
        for action, keywords in _PLAN_MAP.items():
            if any(kw in text for kw in keywords):
                return action
    return None


def _causal_grounded(data: dict) -> bool:
    valid = {p["object"] for p in data.get("perception", [])}
    valid |= {p["subject"] for p in data.get("prediction", [])}
    factors = data.get("planning", {}).get("causal_factors", [])
    return bool(factors) and all(f in valid for f in factors)


# ── load data ─────────────────────────────────────────────────────────────────
print("Loading DriveLM val split...")
full_ds   = load_drivelm("train")
_, val_ds = train_val_split(full_ds)

hf_ds    = full_ds._ds
prev_map = full_ds._prev_map

n = min(N_SAMPLES, len(val_ds))
val_idx = val_ds._idx[:n]

tag = "ft" if USE_FT else "base"
print(f"Evaluating {n} samples  |  model: {tag}\n")

# ── eval loop ─────────────────────────────────────────────────────────────────
counts  = {"parse": 0, "causal": 0, "plan_match": 0, "plan_total": 0}
records = []

for step, idx in enumerate(val_idx):
    sample = hf_ds[idx]

    # Build temporal frame sequence
    prev_imgs = [_load_img(hf_ds[p]["image"]) for p in prev_map.get(idx, [])]
    cur_img   = _load_img(sample["image"])
    frames    = prev_imgs + [cur_img]
    while len(frames) < N_FRAMES:
        frames = [frames[0]] + frames

    msgs = build_messages(frames)
    raw  = infer_ft(msgs, ADAPTER_PATH) if USE_FT else infer(msgs)

    rec = {"idx": idx, "ok": False}

    try:
        data       = parse_output(raw)
        rec["ok"]  = True
        rec["data"] = data
        counts["parse"] += 1

        if _causal_grounded(data):
            counts["causal"] += 1

        gt = _gt_action(sample.get("conversations", []))
        if gt is not None:
            counts["plan_total"] += 1
            pred_action = data.get("planning", {}).get("meta_action")
            matched = (pred_action == gt)
            if matched:
                counts["plan_match"] += 1
            rec.update({"gt_action": gt, "pred_action": pred_action, "plan_match": matched})

    except Exception as e:
        rec["error"] = str(e)
        rec["raw"]   = raw

    records.append(rec)

    if (step + 1) % 20 == 0 or (step + 1) == n:
        pr = counts["parse"] / (step + 1)
        print(f"  {step + 1:>4}/{n}  parse={pr:.1%}")

# ── summary ───────────────────────────────────────────────────────────────────
parsed = max(counts["parse"], 1)
pt     = max(counts["plan_total"], 1)

summary = {
    "model":           tag,
    "n_samples":       n,
    "parse_rate":      round(counts["parse"] / n, 4),
    "causal_grounded": round(counts["causal"] / parsed, 4),
    "planning_match":  round(counts["plan_match"] / pt, 4),
    "plan_total":      counts["plan_total"],
}

print(f"\n{'='*42}")
print(f"  Model           : {tag}")
print(f"  Samples         : {n}")
print(f"  Parse rate      : {summary['parse_rate']:.1%}")
print(f"  Causal grounded : {summary['causal_grounded']:.1%}")
print(f"  Planning match  : {summary['planning_match']:.1%}  ({counts['plan_total']} samples with GT label)")
print(f"{'='*42}\n")

out_path = Path(f"eval_results_{tag}.json")
out_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
print(f"Saved -> {out_path}")
