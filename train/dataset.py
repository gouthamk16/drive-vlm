from datasets import load_dataset
from PIL import Image, ImageDraw
import io, random

DRIVELM_REPO = "OpenDriveLab/DriveLM"
DRIVELM_CONFIG = "DriveLM_nuScenes"


def _to_messages(sample) -> dict:
    """Convert a raw DriveLM sample to {messages: [...]} chat format.
    Kept separate from the dataset pipeline — called lazily at collation time
    to avoid PyArrow serialisation issues with mixed-type content fields."""
    img = sample["image"]
    if isinstance(img, bytes):
        img = Image.open(io.BytesIO(img)).convert("RGB")
    elif not isinstance(img, Image.Image):
        img = Image.fromarray(img).convert("RGB")

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


class LazyDataset:
    """Wraps a raw HuggingFace dataset and converts samples on-the-fly.
    Avoids Arrow serialisation errors from mixed-type message content."""

    def __init__(self, hf_dataset):
        self._ds = hf_dataset

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return _to_messages(self._ds[idx])


def load_drivelm(split: str = "train") -> LazyDataset:
    ds = load_dataset(DRIVELM_REPO, DRIVELM_CONFIG, split=split)
    return LazyDataset(ds)


def train_val_split(ds: LazyDataset, val_ratio: float = 0.05):
    n = len(ds)
    val_n = max(1, int(n * val_ratio))
    # deterministic split: last val_ratio fraction as val
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    train_idx, val_idx = indices[val_n:], indices[:val_n]

    class _Subset:
        def __init__(self, ds, idx): self._ds, self._idx = ds, idx
        def __len__(self): return len(self._idx)
        def __getitem__(self, i): return self._ds[self._idx[i]]

    return _Subset(ds, train_idx), _Subset(ds, val_idx)


def synthetic_dataset(n: int = 32) -> LazyDataset:
    """Synthetic data for pipeline validation without HF auth."""
    qa_pairs = [
        ("What should the ego vehicle do?",
         '{"scene":{"weather":"clear","time":"day","road":"intersection","ego_lane":"center"},'
         '"perception":[{"object":"car","direction":"front","distance":"near","state":"stopped","risk":"medium"}],'
         '"prediction":[{"subject":"car","action":"remain stopped","confidence":"high","influence":"blocks forward path"}],'
         '"planning":{"meta_action":"stop","reason":"car blocking intersection","causal_factors":["car"]}}'),
        ("Analyse this driving scene.",
         '{"scene":{"weather":"clear","time":"day","road":"highway","ego_lane":"center"},'
         '"perception":[{"object":"truck","direction":"front-left","distance":"mid","state":"moving_straight","risk":"low"}],'
         '"prediction":[{"subject":"truck","action":"continue straight","confidence":"high","influence":"no immediate effect"}],'
         '"planning":{"meta_action":"maintain_speed","reason":"truck posing no immediate risk","causal_factors":["truck"]}}'),
    ]

    class _SyntheticDataset:
        def __init__(self, n):
            self._n = n
            self._pairs = qa_pairs

        def __len__(self):
            return self._n

        def __getitem__(self, idx):
            random.seed(idx)
            img = Image.new("RGB", (224, 224), color=(random.randint(80, 180),) * 3)
            ImageDraw.Draw(img).rectangle([50, 80, 174, 144], fill=(50, 50, 50))
            q, a = self._pairs[idx % len(self._pairs)]
            return {"messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": q},
                ]},
                {"role": "assistant", "content": a},
            ]}

    return _SyntheticDataset(n)
