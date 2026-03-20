"""
Evaluate base or fine-tuned model on DriveLM val set.

Metrics
-------
parse_rate       — % of samples producing valid JSON
causal_grounded  — % of valid outputs where every causal_factor references
                   a real perception object or prediction subject

Note: planning_match is not available on val (answers are held out).

Usage
-----
  python eval.py                  # base model, 200 samples
  python eval.py 500              # base model, 500 samples
  python eval.py --ft             # fine-tuned (checkpoints/lora-final)
  python eval.py --ft 500

Requirements
------------
  drivelm_imgs_val/ directory must exist (run once to get it):
    python -c "
    from huggingface_hub import hf_hub_download; import zipfile
    z = hf_hub_download('OpenDriveLab/DriveLM', 'drivelm_nus_imgs_val.zip',
        repo_type='dataset', token='YOUR_HF_TOKEN')
    zipfile.ZipFile(z).extractall('drivelm_imgs_val')
    "
"""

import sys
import os
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

from prompt import build_messages
from model import infer, infer_ft
from output import parse as parse_output

# ── config ────────────────────────────────────────────────────────────────────
USE_FT       = "--ft" in sys.argv
N_SAMPLES    = next((int(a) for a in sys.argv[1:] if a.isdigit()), 200)
ADAPTER_PATH = "checkpoints/lora-final"
IMG_DIR      = Path("drivelm_imgs_val/val_data/CAM_FRONT")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")  # set via: export HF_TOKEN=hf_...
N_FRAMES     = 3


# ── data helpers ──────────────────────────────────────────────────────────────
def _load_val_data() -> list:
    """Load val JSON and return a flat list of samples with temporal context."""
    json_path = hf_hub_download(
        repo_id="OpenDriveLab/DriveLM",
        filename="v1_1_val_nus_q_only.json",
        repo_type="dataset",
        token=HF_TOKEN,
    )
    with open(json_path) as f:
        data = json.load(f)

    samples = []
    for scene_token, scene in data.items():
        frame_tokens = list(scene["key_frames"].keys())
        for j, sample_token in enumerate(frame_tokens):
            frame = scene["key_frames"][sample_token]
            prev_tokens = frame_tokens[max(0, j - (N_FRAMES - 1)) : j]
            samples.append({
                "scene_token":   scene_token,
                "sample_token":  sample_token,
                "prev_tokens":   prev_tokens,
                "frame_lookup":  {t: scene["key_frames"][t]["image_paths"]["CAM_FRONT"]
                                  for t in prev_tokens + [sample_token]},
            })
    random.seed(42)
    random.shuffle(samples)
    return samples


def _load_frame(raw_path: str) -> Image.Image:
    fname = Path(raw_path).name
    return Image.open(IMG_DIR / fname).convert("RGB")


def _build_frames(sample: dict) -> list:
    frames = [_load_frame(sample["frame_lookup"][t]) for t in sample["prev_tokens"]]
    frames.append(_load_frame(sample["frame_lookup"][sample["sample_token"]]))
    while len(frames) < N_FRAMES:
        frames = [frames[0]] + frames
    return frames


# ── metric helpers ────────────────────────────────────────────────────────────
def _causal_grounded(data: dict) -> bool:
    valid = {p["object"] for p in data.get("perception", [])}
    valid |= {p["subject"] for p in data.get("prediction", [])}
    factors = data.get("planning", {}).get("causal_factors", [])
    return bool(factors) and all(f in valid for f in factors)


# ── main ──────────────────────────────────────────────────────────────────────
if not IMG_DIR.exists():
    print(f"ERROR: {IMG_DIR} not found. Download val images first (see docstring).")
    sys.exit(1)

print("Loading DriveLM val data...")
all_samples = _load_val_data()
n = min(N_SAMPLES, len(all_samples))
samples = all_samples[:n]

tag = "ft" if USE_FT else "base"
print(f"Evaluating {n} samples  |  model: {tag}\n")

counts  = {"parse": 0, "causal": 0}
records = []

for step, s in enumerate(samples):
    frames = _build_frames(s)
    msgs   = build_messages(frames)
    raw    = infer_ft(msgs, ADAPTER_PATH) if USE_FT else infer(msgs)

    rec = {"scene": s["scene_token"], "sample": s["sample_token"], "ok": False}

    try:
        data        = parse_output(raw)
        rec["ok"]   = True
        rec["data"] = data
        counts["parse"] += 1
        if _causal_grounded(data):
            counts["causal"] += 1
    except Exception as e:
        rec["error"] = str(e)
        rec["raw"]   = raw

    records.append(rec)

    if (step + 1) % 20 == 0 or (step + 1) == n:
        pr = counts["parse"] / (step + 1)
        print(f"  {step + 1:>4}/{n}  parse={pr:.1%}")

# ── summary ───────────────────────────────────────────────────────────────────
parsed = max(counts["parse"], 1)
summary = {
    "model":           tag,
    "n_samples":       n,
    "parse_rate":      round(counts["parse"] / n, 4),
    "causal_grounded": round(counts["causal"] / parsed, 4),
}

print(f"\n{'='*40}")
print(f"  Model           : {tag}")
print(f"  Samples         : {n}")
print(f"  Parse rate      : {summary['parse_rate']:.1%}")
print(f"  Causal grounded : {summary['causal_grounded']:.1%}")
print(f"{'='*40}\n")

out_path = Path(f"eval_results_{tag}.json")
out_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
print(f"Saved -> {out_path}")
