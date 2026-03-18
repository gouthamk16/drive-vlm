# DriveVLM

Driving scene analysis using a fine-tuned Vision-Language Model (Qwen2.5-VL).

Given a single dashcam image, outputs structured perception → prediction → planning reasoning in JSON.

## Setup

```bash
py -3.12 -m venv .venv
.venv/Scripts/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers bitsandbytes accelerate qwen-vl-utils datasets rich pillow
```

## Usage

```bash
python main.py <image_path>
```

## Model

Base: `Qwen/Qwen2.5-VL-3B-Instruct` (4-bit NF4, runs on 8GB VRAM)

Fine-tuned on [DriveLM](https://github.com/OpenDriveLab/DriveLM) (nuScenes) via QLoRA.

## Output schema

```json
{
  "scene":      { "weather", "time", "road", "ego_lane" },
  "perception": [ { "object", "direction", "distance", "state", "risk" } ],
  "prediction": [ { "subject", "action", "confidence", "influence" } ],
  "planning":   { "meta_action", "reason", "causal_factors" }
}
```
