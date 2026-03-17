from datasets import load_dataset
from PIL import Image
import io

DRIVELM_REPO = "OpenDriveLab/DriveLM"
DRIVELM_CONFIG = "DriveLM_nuScenes"


def _format_sample(sample) -> dict:
    img = sample["image"]
    if isinstance(img, bytes):
        img = Image.open(io.BytesIO(img)).convert("RGB")

    messages = []
    for i, turn in enumerate(sample.get("conversations", [])):
        role = "user" if turn["from"] == "human" else "assistant"
        if i == 0:
            messages.append({
                "role": role,
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": turn["value"]},
                ],
            })
        else:
            messages.append({"role": role, "content": turn["value"]})
    return {"messages": messages}


def load_drivelm(split: str = "train"):
    ds = load_dataset(DRIVELM_REPO, DRIVELM_CONFIG, split=split, trust_remote_code=True)
    return ds.map(_format_sample, remove_columns=ds.column_names)


def train_val_split(ds, val_ratio: float = 0.05):
    split = ds.train_test_split(test_size=val_ratio, seed=42)
    return split["train"], split["test"]
