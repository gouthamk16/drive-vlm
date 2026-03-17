# DriveVLM-CLI Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that takes a dashcam image and returns structured perception → prediction → planning output using Qwen2.5-VL-7B-Instruct, first prompt-only then with QLoRA fine-tuning on DriveLM.

**Architecture:** Five focused modules wired by a single argparse CLI. Phase 1 (prompt-only) covers all five modules. Phase 2 adds `train/dataset.py`. Phase 3 adds `train/finetune.py`. Phase 4 extends `model.py` and `output.py` for adapter loading and side-by-side comparison.

**Tech Stack:** Python 3.10+, PyTorch 2.1+, Unsloth (model load + QLoRA), Transformers 4.45+, qwen-vl-utils, HuggingFace datasets, Rich (terminal output)

**Testing protocol:** No pytest. Every feature is validated by running the CLI against `test.jpg` before committing. A commit only happens when both of the following succeed cleanly:
```bash
python cli.py --image test.jpg
python cli.py --image test.jpg --json | python -m json.tool
```

---

## File Map

| File | Created/Modified | Responsibility |
|---|---|---|
| `.gitignore` | Create | Ignore `__pycache__`, `.env`, `checkpoints/`, `*.ckpt` |
| `requirements.txt` | Create | All dependencies pinned |
| `test.jpg` | Create (download) | Sample dashcam image for all validation runs |
| `config.py` | Create | All constants: MODEL_ID, pixel caps, paths |
| `model.py` | Create | Load Qwen2.5-VL via Unsloth, run inference, return raw string |
| `prompt.py` | Create | DriveLM-schema system prompt + Qwen2.5-VL message builder |
| `output.py` | Create | Parse raw string → structured dict; Rich terminal + JSON render |
| `cli.py` | Create | argparse entry point wiring the four above |
| `train/dataset.py` | Create | Load DriveLM from HuggingFace, format to Qwen2.5-VL chat format |
| `train/finetune.py` | Create | QLoRA training loop via Unsloth |
| `model.py` (Phase 4) | Modify | Add LoRA adapter load path |
| `output.py` (Phase 4) | Modify | Add `--compare` side-by-side render |

---

## Phase 1 — Prompt-Only CLI

---

### Task 1: Project Skeleton

**Milestone:** `v0.1-skeleton`

**Files:**
- Create: `drivevlm/.gitignore`
- Create: `drivevlm/requirements.txt`
- Create: `drivevlm/config.py` (empty stub)
- Create: `drivevlm/model.py` (empty stub)
- Create: `drivevlm/prompt.py` (empty stub)
- Create: `drivevlm/output.py` (empty stub)
- Create: `drivevlm/cli.py` (empty stub)
- Create: `drivevlm/train/__init__.py`
- Create: `drivevlm/train/dataset.py` (empty stub)
- Create: `drivevlm/train/finetune.py` (empty stub)
- Create: `drivevlm/test.jpg` (download sample)

- [ ] **Step 1: Create `.gitignore`**

```
__pycache__/
*.pyc
.env
checkpoints/
*.ckpt
*.safetensors
wandb/
.DS_Store
```

- [ ] **Step 2: Create `requirements.txt`**

```
torch>=2.1.0
unsloth
transformers>=4.45.0
trl>=0.11.0
qwen-vl-utils
datasets
rich
pillow
```

- [ ] **Step 3: Create empty module stubs**

Each file should contain only a single comment for now:

`config.py`:
```python
# constants and paths only — no logic
```

`model.py`:
```python
# model load + inference — returns raw string
```

`prompt.py`:
```python
# system prompt + message builder
```

`output.py`:
```python
# parse raw string → dict; render to terminal or JSON
```

`cli.py`:
```python
# argparse entry point
```

`train/__init__.py`: empty file

`train/dataset.py`:
```python
# DriveLM loader + Qwen2.5-VL chat formatter
```

`train/finetune.py`:
```python
# QLoRA training loop
```

- [ ] **Step 4: Download a test dashcam image**

```bash
curl -L "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Dashcam_image_example.jpg/1280px-Dashcam_image_example.jpg" -o test.jpg
```

If that URL is unavailable, use any dashcam image saved as `test.jpg` in the project root. The image must show a road scene from a driver's perspective.

- [ ] **Step 5: Commit**

```bash
git add .gitignore requirements.txt config.py model.py prompt.py output.py cli.py train/ test.jpg
git commit -m "init: project skeleton, stubs, test image"
```

Expected: 10 files changed, clean working tree.

---

### Task 2: Config

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Write `config.py`**

```python
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

LORA_ADAPTER_PATH = None  # set at runtime via CLI --adapter flag
```

- [ ] **Step 2: Verify import**

```bash
python -c "import config; print(config.MODEL_ID)"
```

Expected output: `Qwen/Qwen2.5-VL-7B-Instruct`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: config — model id and pixel cap constants"
```

---

### Task 3: Model Load + Inference

**Files:**
- Modify: `model.py`

Note: Unsloth must be installed and CUDA available. The first load downloads ~15GB. Expected load time: 3–5 minutes on first run.

- [ ] **Step 1: Write `model.py`**

```python
from unsloth import FastVisionModel
from qwen_vl_utils import process_vision_info
import config

_model = None
_processor = None


def load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor
    model, processor = FastVisionModel.from_pretrained(
        config.MODEL_ID,
        load_in_4bit=True,
        use_gradient_checkpointing=False,
    )
    FastVisionModel.for_inference(model)
    _model, _processor = model, processor
    return model, processor


def infer(messages: list) -> str:
    model, processor = load_model()
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=512)
    # strip input tokens from output
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]
```

- [ ] **Step 2: Verify model loads**

```bash
python -c "from model import load_model; load_model(); print('model loaded OK')"
```

Expected: model downloads (if first time), then prints `model loaded OK`. No OOM errors.

- [ ] **Step 3: Commit**

```bash
git add model.py
git commit -m "feat: model.py — Qwen2.5-VL-7B load + inference via Unsloth"
```

---

### Task 4: System Prompt + Message Builder

**Milestone:** `v0.2-prompt`

**Files:**
- Modify: `prompt.py`

- [ ] **Step 1: Write `prompt.py`**

```python
from PIL import Image

SYSTEM_PROMPT = """You are a driving scene analyst. Given a dashcam image, respond in strict JSON with this exact structure:

{
  "perception": [
    {"object": "<what>", "location": "<where in frame: left/center/right/far-left/far-right + near/mid/far>", "state": "<what it is doing>"}
  ],
  "prediction": [
    {"subject": "<object from perception>", "action": "<what it will likely do next>", "confidence": "<high|medium|low>"}
  ],
  "planning": {
    "action": "<what the ego vehicle should do: e.g. maintain speed, brake, change lane left>",
    "reason": "<one sentence explanation>",
    "causal_factors": ["<specific perceived or predicted element that drove this decision>"]
  }
}

Rules:
- Output valid JSON only. No markdown, no explanation outside the JSON.
- perception must list every traffic-relevant object visible.
- causal_factors must reference objects or predictions already listed above.
- If the scene is clear, perception may be an empty list and planning.action should be "maintain speed"."""


def build_messages(image_path: str) -> list:
    img = Image.open(image_path).convert("RGB")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Analyse this driving scene."},
            ],
        },
    ]
```

- [ ] **Step 2: Verify message builds without error**

```bash
python -c "from prompt import build_messages; m = build_messages('test.jpg'); print('messages OK, roles:', [x['role'] for x in m])"
```

Expected: `messages OK, roles: ['system', 'user']`

- [ ] **Step 3: Run end-to-end inference (no output parsing yet)**

```bash
python -c "
from prompt import build_messages
from model import infer
msgs = build_messages('test.jpg')
raw = infer(msgs)
print(raw[:500])
"
```

Expected: a JSON-like string starting with `{`. The model may not produce perfect JSON yet — that is fine, we are just confirming inference runs.

- [ ] **Step 4: Commit**

```bash
git add prompt.py
git commit -m "feat: prompt.py — DriveLM-schema system prompt + message builder"
```

---

### Task 5: Output Parser + Renderer

**Files:**
- Modify: `output.py`

- [ ] **Step 1: Write `output.py`**

```python
import json
import re
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def parse(raw: str) -> dict:
    # strip markdown code fences if model wraps output
    cleaned = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned)
    return json.loads(cleaned)


def render_rich(data: dict):
    # Perception
    console.rule("[bold cyan]Perception")
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("Object"), t.add_column("Location"), t.add_column("State")
    for p in data.get("perception", []):
        t.add_row(p["object"], p["location"], p["state"])
    console.print(t)

    # Prediction
    console.rule("[bold yellow]Prediction")
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("Subject"), t.add_column("Action"), t.add_column("Confidence")
    for p in data.get("prediction", []):
        t.add_row(p["subject"], p["action"], p["confidence"])
    console.print(t)

    # Planning
    console.rule("[bold green]Planning")
    plan = data.get("planning", {})
    console.print(f"[bold]Action:[/bold] {plan.get('action', '')}")
    console.print(f"[bold]Reason:[/bold] {plan.get('reason', '')}")
    factors = plan.get("causal_factors", [])
    if factors:
        console.print(f"[bold]Causal factors:[/bold] {', '.join(factors)}")


def render_json(data: dict) -> str:
    return json.dumps(data, indent=2)
```

- [ ] **Step 2: Verify parsing on model output**

```bash
python -c "
from prompt import build_messages
from model import infer
from output import parse, render_rich
msgs = build_messages('test.jpg')
raw = infer(msgs)
data = parse(raw)
render_rich(data)
print('parse OK')
"
```

Expected: rich table output with perception/prediction/planning sections, then `parse OK`. If `json.loads` raises, the model output is malformed — note it but proceed; the CLI will handle this gracefully in the next task.

- [ ] **Step 3: Commit**

```bash
git add output.py
git commit -m "feat: output.py — JSON parser + rich terminal renderer"
```

---

### Task 6: CLI Entry Point

**Milestone:** `v0.3-cli`

**Files:**
- Modify: `cli.py`

- [ ] **Step 1: Write `cli.py`**

```python
import argparse
import sys
from prompt import build_messages
from model import infer
from output import parse, render_rich, render_json


def main():
    ap = argparse.ArgumentParser(description="DriveVLM — driving scene analyser")
    ap.add_argument("--image", required=True, help="Path to dashcam image")
    ap.add_argument("--json", action="store_true", help="Output raw JSON")
    ap.add_argument("--mode", choices=["prompt", "ft"], default="prompt",
                    help="prompt=base model, ft=load LoRA adapter")
    ap.add_argument("--adapter", default=None,
                    help="Path to LoRA adapter dir (required for --mode ft)")
    ap.add_argument("--compare", action="store_true",
                    help="Run both modes side-by-side (requires --adapter)")
    args = ap.parse_args()

    if args.mode == "ft" and not args.adapter:
        print("error: --mode ft requires --adapter", file=sys.stderr)
        sys.exit(1)
    if args.compare and not args.adapter:
        print("error: --compare requires --adapter", file=sys.stderr)
        sys.exit(1)

    msgs = build_messages(args.image)
    raw = infer(msgs)

    try:
        data = parse(raw)
    except Exception as e:
        print(f"warning: could not parse model output as JSON: {e}", file=sys.stderr)
        print(raw)
        sys.exit(1)

    if args.json:
        print(render_json(data))
    else:
        render_rich(data)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Validation run — pretty-print**

```bash
python cli.py --image test.jpg
```

Expected: rich terminal output with three labeled sections (Perception, Prediction, Planning). No errors.

- [ ] **Step 3: Validation run — JSON mode**

```bash
python cli.py --image test.jpg --json | python -m json.tool
```

Expected: pretty-printed valid JSON with `perception`, `prediction`, `planning` keys. `python -m json.tool` must exit 0.

- [ ] **Step 4: Tag milestone**

```bash
git add cli.py
git commit -m "feat: cli.py — argparse entry point, --json and --mode flags"
git tag v0.3-cli
```

- [ ] **Step 5: Run drivevlm-review skill**

Dispatch the `drivevlm-review` skill (or `superpowers:requesting-code-review`) against the diff since project init. Confirm no CLAUDE.md violations before proceeding to Phase 2.

---

## Phase 2 — Data Pipeline

---

### Task 7: DriveLM Dataset Loader

**Milestone:** `v0.4-data`

**Files:**
- Modify: `train/dataset.py`

Note: DriveLM is at `OpenDriveLab/DriveLM` on HuggingFace. It has a `DriveLM_nuScenes` config. Requires HuggingFace account login (`huggingface-cli login`) and dataset access approval if gated.

- [ ] **Step 1: Write `train/dataset.py`**

```python
from datasets import load_dataset
from PIL import Image
import io


DRIVELM_REPO = "OpenDriveLab/DriveLM"
DRIVELM_CONFIG = "DriveLM_nuScenes"


def _format_sample(sample) -> dict:
    """Convert one DriveLM sample to Qwen2.5-VL chat format."""
    img = sample["image"]
    if isinstance(img, bytes):
        img = Image.open(io.BytesIO(img)).convert("RGB")

    conversations = sample.get("conversations", [])
    # DriveLM conversations alternate human/gpt turns
    messages = []
    for i, turn in enumerate(conversations):
        role = "user" if turn["from"] == "human" else "assistant"
        content = turn["value"]
        if i == 0:
            # attach image to first user turn
            messages.append({
                "role": role,
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": content},
                ],
            })
        else:
            messages.append({"role": role, "content": content})
    return {"messages": messages}


def load_drivelm(split: str = "train"):
    """Returns HuggingFace dataset split with 'messages' column."""
    ds = load_dataset(DRIVELM_REPO, DRIVELM_CONFIG, split=split, trust_remote_code=True)
    return ds.map(_format_sample, remove_columns=ds.column_names)


def train_val_split(ds, val_ratio: float = 0.05):
    split = ds.train_test_split(test_size=val_ratio, seed=42)
    return split["train"], split["test"]
```

- [ ] **Step 2: Load 10 samples and inspect**

```bash
python -c "
from train.dataset import load_drivelm
ds = load_drivelm('train')
for i, s in enumerate(ds.select(range(10))):
    msgs = s['messages']
    print(f'sample {i}: {len(msgs)} turns, first user content types: {[c[\"type\"] for c in msgs[0][\"content\"] if isinstance(msgs[0][\"content\"], list)]}')
"
```

Expected: 10 lines, each showing 2+ turns with `image` and `text` content types in turn 0.

- [ ] **Step 3: Measure token lengths on 20 samples**

```bash
python -c "
from train.dataset import load_drivelm
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
ds = load_drivelm('train')
lengths = []
for s in ds.select(range(20)):
    text = ' '.join([
        c if isinstance(c, str) else ' '.join(x['text'] for x in c if x['type'] == 'text')
        for m in s['messages'] for c in [m['content']]
    ])
    lengths.append(len(tok.encode(text)))
print('min:', min(lengths), 'max:', max(lengths), 'mean:', sum(lengths)//len(lengths))
"
```

Expected: max should be well under 2048. Confirm `max_seq=2048` is sufficient.

- [ ] **Step 4: Commit and tag**

```bash
git add train/dataset.py
git commit -m "feat: dataset.py — DriveLM loader + Qwen2.5-VL chat formatter"
git tag v0.4-data
```

---

## Phase 3 — QLoRA Fine-tuning

---

### Task 8: QLoRA Training Loop

**Milestone:** `v0.5-qlora`

**Files:**
- Modify: `train/finetune.py`

- [ ] **Step 1: Write `train/finetune.py`**

```python
import torch
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from train.dataset import load_drivelm, train_val_split
import config

QLORA = dict(
    r=16,
    lora_alpha=32,
    # target_modules: Unsloth defaults — do not override
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

TRAIN_CFG = dict(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    max_seq_length=2048,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    num_train_epochs=1,
    fp16=not is_bf16_supported(),
    bf16=is_bf16_supported(),
    logging_steps=10,
    save_strategy="epoch",
    output_dir="checkpoints",
    report_to="none",
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
)


def train(smoke_test: bool = False):
    model, processor = FastVisionModel.from_pretrained(
        config.MODEL_ID, load_in_4bit=True
    )
    model = FastVisionModel.get_peft_model(model, **QLORA)

    ds = load_drivelm("train")
    train_ds, val_ds = train_val_split(ds)

    if smoke_test:
        train_ds = train_ds.select(range(16))
        TRAIN_CFG["max_steps"] = 10
        TRAIN_CFG["save_strategy"] = "no"

    cfg = SFTConfig(**TRAIN_CFG)
    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=cfg,
    )
    trainer.train()

    if not smoke_test:
        model.save_pretrained("checkpoints/lora-final")
        processor.save_pretrained("checkpoints/lora-final")
        print("saved to checkpoints/lora-final")


if __name__ == "__main__":
    import sys
    train(smoke_test="--smoke" in sys.argv)
```

- [ ] **Step 2: Smoke test (10 steps, confirm no OOM)**

```bash
python train/finetune.py --smoke
```

Expected: 10 training steps complete, loss logged, no CUDA OOM. Typical VRAM usage should stay under 7.5GB. If OOM occurs, reduce `max_seq_length` to 1024 first.

- [ ] **Step 3: Commit smoke-test passing state**

```bash
git add train/finetune.py
git commit -m "feat: finetune.py — QLoRA training loop, smoke test passing"
```

- [ ] **Step 4: Full training run**

```bash
python train/finetune.py
```

Expected: ~18,750 optimizer steps, checkpoint saved to `checkpoints/lora-final/`. This will run for several hours on an RTX 4060. Monitor loss — should decrease steadily.

- [ ] **Step 5: Tag milestone**

```bash
git commit -m "test: full QLoRA training run complete, adapter saved" --allow-empty
git tag v0.5-qlora
```

---

## Phase 4 — Fine-tuned Inference

---

### Task 9: Adapter Load + Compare Mode

**Milestone:** `v0.6-finetuned`

**Files:**
- Modify: `model.py` — add adapter loading path
- Modify: `output.py` — add compare renderer
- Modify: `cli.py` — wire `--compare` through

- [ ] **Step 1: Extend `model.py` with adapter loading**

Add the `load_ft_model` function below the existing `load_model` and `infer` functions:

```python
_ft_model = None
_ft_processor = None


def load_ft_model(adapter_path: str):
    global _ft_model, _ft_processor
    if _ft_model is not None:
        return _ft_model, _ft_processor
    from peft import PeftModel
    base_model, processor = load_model()
    ft_model = PeftModel.from_pretrained(base_model, adapter_path)
    FastVisionModel.for_inference(ft_model)
    _ft_model, _ft_processor = ft_model, processor
    return ft_model, processor


def infer_ft(messages: list, adapter_path: str) -> str:
    model, processor = load_ft_model(adapter_path)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=512)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]
```

- [ ] **Step 2: Add `render_compare` to `output.py`**

Add at the bottom of `output.py`:

```python
def render_compare(base_data: dict, ft_data: dict):
    from rich.columns import Columns
    from rich.panel import Panel

    console.rule("[bold]Prompt-only vs Fine-tuned comparison")

    base_json = render_json(base_data)
    ft_json = render_json(ft_data)

    console.print(Columns([
        Panel(base_json, title="[cyan]Prompt-only", border_style="cyan"),
        Panel(ft_json, title="[green]Fine-tuned", border_style="green"),
    ]))
```

- [ ] **Step 3: Update `cli.py` to wire `--compare` and `--mode ft`**

Replace the inference section of `main()`:

```python
    msgs = build_messages(args.image)

    if args.compare:
        from model import infer, infer_ft
        raw_base = infer(msgs)
        raw_ft = infer_ft(msgs, args.adapter)
        try:
            base_data = parse(raw_base)
            ft_data = parse(raw_ft)
        except Exception as e:
            print(f"error parsing output: {e}", file=sys.stderr)
            sys.exit(1)
        from output import render_compare
        render_compare(base_data, ft_data)
        return

    if args.mode == "ft":
        from model import infer_ft
        raw = infer_ft(msgs, args.adapter)
    else:
        from model import infer
        raw = infer(msgs)

    try:
        data = parse(raw)
    except Exception as e:
        print(f"warning: could not parse model output as JSON: {e}", file=sys.stderr)
        print(raw)
        sys.exit(1)

    if args.json:
        print(render_json(data))
    else:
        render_rich(data)
```

- [ ] **Step 4: Validation — fine-tuned mode**

```bash
python cli.py --image test.jpg --mode ft --adapter checkpoints/lora-final
python cli.py --image test.jpg --mode ft --adapter checkpoints/lora-final --json | python -m json.tool
```

Expected: structured output from fine-tuned model. JSON must parse cleanly.

- [ ] **Step 5: Validation — compare mode**

```bash
python cli.py --image test.jpg --compare --adapter checkpoints/lora-final
```

Expected: side-by-side Rich panel output showing prompt-only (left) vs fine-tuned (right).

- [ ] **Step 6: Reasoning-action consistency check**

Verify that `causal_factors` in the fine-tuned model's output references entities actually present in `perception` or `prediction`. Run on 5 images and inspect manually:

```bash
python cli.py --image test.jpg --mode ft --adapter checkpoints/lora-final --json
```

For each output, confirm every string in `planning.causal_factors` names an object that appears in `perception[*].object` or `prediction[*].subject`. This is the Alpamayo-R1 Chain of Causation consistency check — it is the key qualitative eval for this project.

- [ ] **Step 7: Final code review pass**

Run `superpowers:requesting-code-review` (or `drivevlm-review` skill) with diff since `v0.5-qlora`. Confirm all modules still have single responsibilities and no CLAUDE.md violations were introduced.

- [ ] **Step 8: Tag final milestone**

```bash
git add model.py output.py cli.py
git commit -m "feat: adapter loading, --mode ft, --compare side-by-side"
git tag v0.6-finetuned
```

---

## Milestone Summary

| Tag | What works at this point |
|---|---|
| `v0.1-skeleton` | Repo structure, stubs, test.jpg |
| `v0.2-prompt` | System prompt + raw inference confirmed |
| `v0.3-cli` | `python cli.py --image test.jpg` — full pretty-print + JSON |
| `v0.4-data` | DriveLM loads and formats to Qwen2.5-VL chat format |
| `v0.5-qlora` | LoRA adapter trained and saved |
| `v0.6-finetuned` | `--mode ft` and `--compare` working |

---

## Quick Reference: Validation Commands

Run these two after every feature. Both must succeed before committing:

```bash
python cli.py --image test.jpg
python cli.py --image test.jpg --json | python -m json.tool
```
