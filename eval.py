"""
Compare 3B vs 7B-bnb-4bit on 8 diverse driving scenes.
Usage: python eval.py
"""

import json, os, sys, time, traceback
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

from prompt import SYSTEM_PROMPT

IMAGES = [
    ("highway_day",       "eval_images/highway_day.png"),
    ("urban_intersect",   "eval_images/urban_intersection.png"),
    ("urban_pedestrians", "eval_images/urban_pedestrians.png"),
    ("urban_multilane",   "eval_images/urban_multilane.png"),
    ("night_1",           "eval_images/night_1.png"),
    ("night_2",           "eval_images/night_2.png"),
    ("highway_sample",    "eval_images/highway_sample.jpg"),
    ("suburban_day",      "eval_images/suburban_day.png"),
]

MODELS = {
    "3B": "Qwen/Qwen2.5-VL-3B-Instruct",
    "7B": "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
}

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28


def load(model_id):
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    # 7B pre-quantized model has quantization_config embedded — no need to pass cfg
    kwargs = {"device_map": "auto"}
    if "7B" not in model_id:
        kwargs["quantization_config"] = cfg
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    return model, processor


def run(model, processor, image_path):
    img = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Analyse this driving scene."},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to("cuda")
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=1024, temperature=0.1,
                         do_sample=True, repetition_penalty=1.1)
    elapsed = time.time() - t0
    raw = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0]
    return raw, round(elapsed, 1)


def parse(raw):
    import re
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def score(data):
    """Return a dict of quality signals."""
    s = {}
    s["json_valid"] = True
    s["n_objects"]  = len(data.get("perception", []))
    s["n_preds"]    = len(data.get("prediction", []))
    plan            = data.get("planning", {})
    s["has_reason"] = bool(plan.get("reason", "").strip())
    s["has_action"] = bool(plan.get("meta_action", "").strip())
    s["has_scene"]  = bool(data.get("scene", {}))

    # causal integrity: every factor must appear in perception objects or prediction subjects
    perc_objs  = {p["object"] for p in data.get("perception", [])}
    pred_subjs = {p["subject"] for p in data.get("prediction", [])}
    valid_refs = perc_objs | pred_subjs
    factors    = plan.get("causal_factors", [])
    s["causal_valid"] = all(f in valid_refs for f in factors) and bool(factors)

    # influence quality: count predictions that have a non-empty influence sentence
    s["influence_filled"] = sum(
        1 for p in data.get("prediction", []) if p.get("influence", "").strip()
    )
    return s


def print_result(label, img_name, data, raw, elapsed, err=None):
    if err:
        print(f"  [{label}] PARSE FAIL ({elapsed}s): {err}")
        print(f"         raw[:200]: {raw[:200]}")
        return
    s = score(data)
    plan = data.get("planning", {})
    scene = data.get("scene", {})
    print(f"  [{label}] {elapsed}s | scene={scene.get('road','?')}/{scene.get('time','?')} "
          f"| objects={s['n_objects']} preds={s['n_preds']} "
          f"| action={plan.get('meta_action','?')} "
          f"| causal={'ok' if s['causal_valid'] else 'BROKEN'} "
          f"| influence={s['influence_filled']}/{s['n_preds']}")
    print(f"         reason: {plan.get('reason','')[:120]}")


results = {}

for model_key, model_id in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Loading {model_key}: {model_id}")
    print('='*60)
    try:
        model, processor = load(model_id)
    except Exception as e:
        print(f"LOAD FAILED: {e}")
        continue

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM after load: {vram:.2f}GB")
    results[model_key] = {}

    for img_name, img_path in IMAGES:
        print(f"\n  Image: {img_name}")
        try:
            raw, elapsed = run(model, processor, img_path)
            try:
                data = parse(raw)
                results[model_key][img_name] = {"data": data, "elapsed": elapsed}
                print_result(model_key, img_name, data, raw, elapsed)
            except Exception as e:
                results[model_key][img_name] = {"data": None, "elapsed": elapsed, "error": str(e)}
                print_result(model_key, img_name, None, raw, elapsed, err=str(e))
        except Exception as e:
            print(f"  [{model_key}] INFERENCE FAIL: {e}")
            traceback.print_exc()

    del model, processor
    torch.cuda.empty_cache()

# --- Summary ---
print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)
print(f"{'Image':<22} {'3B objects':>10} {'7B objects':>10} {'3B action':<20} {'7B action':<20} {'3B causal':<10} {'7B causal':<10}")
print("-"*100)
for img_name, _ in IMAGES:
    r3 = results.get("3B", {}).get(img_name, {})
    r7 = results.get("7B", {}).get(img_name, {})
    d3 = r3.get("data") or {}
    d7 = r7.get("data") or {}
    s3 = score(d3) if d3 else {}
    s7 = score(d7) if d7 else {}
    a3 = (d3.get("planning") or {}).get("meta_action", "FAIL")
    a7 = (d7.get("planning") or {}).get("meta_action", "FAIL")
    print(f"{img_name:<22} {s3.get('n_objects','?'):>10} {s7.get('n_objects','?'):>10} "
          f"{a3:<20} {a7:<20} "
          f"{'ok' if s3.get('causal_valid') else 'BROKEN':<10} "
          f"{'ok' if s7.get('causal_valid') else 'BROKEN':<10}")

# Save full results
with open("eval_results.json", "w") as f:
    json.dump({
        k: {img: {"elapsed": v.get("elapsed"), "score": score(v["data"]) if v.get("data") else None,
                  "planning": (v.get("data") or {}).get("planning"),
                  "scene": (v.get("data") or {}).get("scene"),
                  "n_perception": len((v.get("data") or {}).get("perception", []))}
            for img, v in imgs.items()}
        for k, imgs in results.items()
    }, f, indent=2)
print("\nFull results saved to eval_results.json")
