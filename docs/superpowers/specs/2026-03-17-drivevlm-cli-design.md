# DriveVLM-CLI — Design Spec
**Date:** 2026-03-17
**Status:** Approved

---

## 1. What This Is

A command-line tool that takes a single dashcam image and returns structured driving scene analysis using a Vision-Language Model. Output follows the DriveLM reasoning schema: perception → prediction → planning.

**What it is not:**
- Not real-time or video-based
- Not a ROS node or embedded system component
- Not a web app or API server
- Not a research framework — it is a focused CLI tool

---

## 2. Goals

1. Run a 4-bit quantized Qwen2.5-VL-7B-Instruct model locally on an RTX 4060 (8GB VRAM)
2. Output structured JSON and/or rich terminal output from a single image
3. Provide a clear comparison baseline between prompt-only and fine-tuned inference
4. Be maintainable, minimal, and reviewable by a senior engineer without explanation

---

## 3. Architecture

### Directory layout

```
drivevlm/
├── CLAUDE.md
├── .gitignore
├── requirements.txt
├── test.jpg             # sample dashcam image committed to repo for validation runs
├── config.py            # model id, pixel caps, paths — no logic
├── model.py             # model load + inference only
├── prompt.py            # system prompt + message builder
├── output.py            # JSON schema + rich terminal renderer
├── cli.py               # argparse entry point
├── train/
│   ├── dataset.py       # DriveLM loader + chat formatter
│   └── finetune.py      # QLoRA training loop
└── docs/
    └── superpowers/specs/
```

### Module responsibilities (one-liner each)

| File | Single responsibility |
|---|---|
| `config.py` | All constants and paths — nothing else imported here; exposes `MODEL_ID` string for model selection |
| `model.py` | Load Qwen2.5-VL via Unsloth from config, run inference, return raw string — Qwen2.5-VL loading path only; InternVL2.5 fallback would require a separate loader |
| `prompt.py` | Build the DriveLM-schema system prompt and format image+query into chat messages |
| `output.py` | Parse raw model output into structured dict; render to terminal or JSON string |
| `cli.py` | Parse args, wire the above four together, print result |
| `train/dataset.py` | Load DriveLM from HuggingFace, format into Qwen2.5-VL chat format |
| `train/finetune.py` | QLoRA training loop using Unsloth — nothing else |

### Data flow

```
cli.py
  → load model (model.py, using config.py)
  → build messages (prompt.py)
  → run inference (model.py)
  → parse + render output (output.py)
  → print to terminal or stdout as JSON
```

---

## 4. Model

**Primary:** `Qwen2.5-VL-7B-Instruct` loaded via Unsloth in 4-bit NF4
**Fallback chain:** Qwen2.5-VL-3B → InternVL2.5-4B

To switch models, change `MODEL_ID` in `config.py`. The Qwen2.5-VL-3B fallback is a direct drop-in (same loader). InternVL2.5-4B would require a new loader function in `model.py`.

**Key inference constraints:**
- `min_pixels = 256 * 28 * 28`
- `max_pixels = 1280 * 28 * 28` (caps visual tokens to ~1280 to avoid OOM)
- `load_in_4bit = True`
- Vision encoder frozen during QLoRA fine-tuning

**Why Qwen2.5-VL-7B:** Best-in-class benchmarks at this size (MMMU 58.6, DocVQA 95.7), used in OmniDrive-R1 which achieves 80.35% on DriveLM, native structured coordinate output, Apache 2.0 license, best Unsloth QLoRA support of any candidate.

---

## 5. Output Schema

The DriveLM reasoning schema, mirrored exactly in both the system prompt and the output parser:

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
    "reason": "string"
  }
}
```

**CLI behavior:**
- Default: rich terminal output (colored sections, human-readable)
- `--json` flag: raw JSON to stdout (machine-readable, pipeable)
- `--compare` flag (Phase 4): runs both `--mode prompt` and `--mode ft` on the same image and prints output side-by-side; requires `--adapter`

---

## 6. CLI Interface

```
python cli.py --image <path> [--json] [--mode prompt|ft] [--adapter <path>] [--compare]
```

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Path to dashcam image |
| `--json` | false | Output raw JSON instead of pretty-print |
| `--mode` | `prompt` | `prompt` = base model + system prompt; `ft` = base + LoRA adapter |
| `--adapter` | None | Path to LoRA adapter dir (required when `--mode ft` or `--compare`) |
| `--compare` | false | Run both prompt and ft modes; print side-by-side; requires `--adapter` |

---

## 7. Phases

### Phase 1 — Prompt-Only (`v0.1` → `v0.3`)
- Base model + DriveLM-schema system prompt
- No fine-tuning
- Deliverable: working CLI, both output modes, milestone tag `v0.3-cli`

### Phase 2 — Data Pipeline (`v0.4`)
- DriveLM dataset from HuggingFace (~300K QA pairs across 696 nuScenes scenes)
- Formatted into Qwen2.5-VL chat format
- Train/val split
- Before writing `finetune.py`, sample 20 formatted examples and measure token lengths to confirm `max_seq` setting
- Deliverable: `dataset.py` that can stream batches, milestone tag `v0.4-data`

### Phase 3 — QLoRA Fine-tuning (`v0.5`)

**QLoRA settings:**
- `rank = 16`, `lora_alpha = 32`
- `target_modules`: Unsloth defaults — do not override (covers all attention projection layers)
- `load_in_4bit = True`, vision encoder frozen
- `use_gradient_checkpointing = "unsloth"`
- `batch = 1`, `grad_accum = 16` → effective batch size = 16
- `max_seq = 2048` (covers 95th percentile of DriveLM chat-formatted samples; 512 is too short for schema + system prompt + image tokens)
- `lr = 2e-4`, `scheduler = cosine`, `warmup_steps = 50`
- DriveLM has ~300K QA pairs; one epoch ≈ 18,750 optimizer steps at effective batch 16

**Steps:**
- Smoke test: 10 steps, confirm no OOM
- Full training run, checkpoint save every epoch
- Deliverable: saved LoRA adapter, milestone tag `v0.5-qlora`

### Phase 4 — Fine-tuned Inference (`v0.6`)
- Load adapter via `--mode ft --adapter`
- `--compare` mode: side-by-side prompt-only vs fine-tuned on same image
- Final code review pass
- Deliverable: `v0.6-finetuned`

---

## 8. Git Discipline

- Linear history on `main`, no branches
- One atomic commit per micro-feature
- Milestone tags: `v0.1-skeleton`, `v0.2-prompt`, `v0.3-cli`, `v0.4-data`, `v0.5-qlora`, `v0.6-finetuned`
- Commit message format: `type: short description` (types: `init`, `feat`, `fix`, `refactor`, `test`)

---

## 9. Testing Protocol

No automated test suite. Every feature is validated by running the CLI against a real image before committing. `test.jpg` is a sample dashcam image committed to the repo root and used exclusively for validation runs.

```bash
python cli.py --image test.jpg
python cli.py --image test.jpg --json | python -m json.tool
```

At each milestone tag, a code review agent checks the diff against CLAUDE.md standards.

---

## 10. CLAUDE.md Contents (Project-Level)

The project CLAUDE.md will encode:
1. Project purpose and non-goals (one paragraph)
2. Model facts: Qwen2.5-VL-7B, 4-bit NF4, pixel caps, fallback chain
3. Module map: one-liner per file
4. Output schema: the DriveLM JSON structure
5. Phase awareness: which phase we're in, what's in scope
6. Testing protocol: run CLI against real image, not pytest
7. Commit discipline: atomic, tagged, linear
8. Hard rules: no README, no utility files, no over-abstraction

---

## 11. Custom Skills

### `drivevlm-test`
Runs `python cli.py --image test.jpg --json` and pipes through `python -m json.tool`. Prints pass/fail + structured output. Used after every feature commit.

### `drivevlm-review`
At each milestone tag, dispatches a code-reviewer agent with the diff since the last tag. Checks against CLAUDE.md standards. Returns structured verdict before proceeding to next milestone.

---

## 12. Dependencies

```
torch>=2.1.0
unsloth
transformers>=4.45.0
qwen-vl-utils
datasets          # DriveLM from HuggingFace
rich              # terminal rendering
pillow
```

No web framework, no FastAPI, no Gradio — CLI only per spec.
