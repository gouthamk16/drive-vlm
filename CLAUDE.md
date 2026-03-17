# DriveVLM — Project Context

A CLI tool that takes a single dashcam image and returns structured driving scene analysis using a fine-tuned Vision-Language Model. Output follows the DriveLM reasoning schema: perception → prediction → planning, with causal factors linking prediction to action (inspired by NVIDIA Alpamayo-R1).

**Not** a web app, real-time system, ROS node, or research framework. CLI only.

---

## Model

- **Primary:** `Qwen/Qwen2.5-VL-7B-Instruct` via Unsloth, 4-bit NF4
- **Fallback:** `Qwen/Qwen2.5-VL-3B-Instruct` (drop-in swap via `MODEL_ID` in `config.py`)
- `min_pixels = 256 * 28 * 28`, `max_pixels = 1280 * 28 * 28` — never change without OOM testing
- Vision encoder is **always frozen** — do not unfreeze it

---

## Module Map

| File | Does exactly one thing |
|---|---|
| `config.py` | Constants and paths only — no logic, no imports from this project |
| `model.py` | Load model + run inference → return raw string. Qwen2.5-VL path only. |
| `prompt.py` | System prompt + message builder. Schema lives here. |
| `output.py` | Parse raw string → structured dict. Render to terminal or JSON. |
| `cli.py` | Wire the above four together via argparse. No business logic here. |
| `train/dataset.py` | Load DriveLM from HuggingFace → format to Qwen2.5-VL chat format |
| `train/finetune.py` | QLoRA training loop. Nothing else. |

If logic doesn't fit cleanly in one of these, question whether it needs to exist.

---

## Output Schema

This is the source of truth. The system prompt, output parser, and fine-tuning targets all mirror this exactly:

```json
{
  "perception": [
    {"object": "string", "location": "string", "state": "string"}
  ],
  "prediction": [
    {"subject": "string", "action": "string", "confidence": "high|medium|low"}
  ],
  "planning": {
    "action": "string",
    "reason": "string",
    "causal_factors": ["string"]
  }
}
```

`causal_factors` must reference entities present in `perception` or `prediction`. This is how consistency is scored.

---

## Current Phase

**Phase 1 — Prompt-Only** (target: `v0.3-cli`)

In scope: `config.py`, `model.py`, `prompt.py`, `output.py`, `cli.py`
Out of scope: anything in `train/`

Update this section when phases change.

---

## CLI

```bash
python cli.py --image <path>                        # pretty-print
python cli.py --image <path> --json                 # raw JSON
python cli.py --image <path> --mode ft --adapter <path>   # fine-tuned
python cli.py --image <path> --compare --adapter <path>   # side-by-side
```

---

## Testing Protocol

No pytest. Every feature is tested by running the CLI against `test.jpg` before committing:

```bash
python cli.py --image test.jpg
python cli.py --image test.jpg --json | python -m json.tool
```

`test.jpg` is committed to the repo root. Use it exclusively for validation runs. If the above two commands succeed cleanly, the feature is done.

---

## Commit Discipline

- One atomic commit per micro-feature
- Linear history on `main`, no branches
- Format: `type: short description` (`init`, `feat`, `fix`, `refactor`, `test`)
- Milestone tags: `v0.1-skeleton`, `v0.2-prompt`, `v0.3-cli`, `v0.4-data`, `v0.5-qlora`, `v0.6-finetuned`
- Run `drivevlm-review` skill at each milestone before tagging

---

## Hard Rules

- No README files unless explicitly requested
- No `utils.py` or catch-all helper modules
- No one-line wrapper functions
- No features added "for future use"
- No comments explaining what the code does — only why, when non-obvious
- Do not touch `train/` files during Phase 1–3 milestones
- Do not change pixel caps without testing for OOM first

---

## QLoRA Settings (Phase 3 reference)

```
rank=16, lora_alpha=32
target_modules: Unsloth defaults — do not override
batch=1, grad_accum=16  →  effective batch = 16
max_seq=2048
lr=2e-4, scheduler=cosine, warmup_steps=50
use_gradient_checkpointing="unsloth"
```

---

## Prior Art

- **DriveLM** (OpenDriveLab, ECCV 2024 Oral) — dataset and schema we build on
- **OmniDrive-R1** — Qwen2.5-VL-7B on DriveLM, 80.35% benchmark score
- **NVIDIA Alpamayo-R1** (NeurIPS 2025) — 10B VLA with Chain of Causation + flow-matching diffusion. Source of `causal_factors` concept. They used RL; we use SFT only.
