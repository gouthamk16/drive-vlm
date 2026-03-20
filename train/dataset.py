from datasets import load_dataset
from PIL import Image, ImageDraw
import io, random

DRIVELM_REPO = "OpenDriveLab/DriveLM"
DRIVELM_CONFIG = "DriveLM_nuScenes"


def _load_img(raw) -> Image.Image:
    if isinstance(raw, bytes):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    return Image.fromarray(raw).convert("RGB")


def _to_messages(sample, prev_imgs=None) -> dict:
    """Convert a raw DriveLM sample to {messages: [...]} chat format.
    prev_imgs: list of PIL Images (T-n ... T-1), if available triggers video block."""
    img = _load_img(sample["image"])

    if prev_imgs:
        visual = {"type": "video", "video": prev_imgs + [img], "fps": 2.0}
    else:
        visual = {"type": "image", "image": img}

    messages = []
    for i, turn in enumerate(sample.get("conversations", [])):
        role = "user" if turn["from"] == "human" else "assistant"
        if i == 0:
            messages.append({
                "role": role,
                "content": [visual, {"type": "text", "text": turn["value"]}],
            })
        else:
            messages.append({"role": role, "content": turn["value"]})
    return {"messages": messages}


def _build_temporal_index(hf_ds) -> dict:
    """Build {idx -> [prev_idx, ...]} from scene metadata if available.
    DriveLM samples share a scene_token; frames within a scene are ordered
    by their position in the dataset. Returns empty dict if no usable field."""
    cols = getattr(hf_ds, "column_names", [])
    scene_field = next((f for f in ["scene_token", "id"] if f in cols), None)
    if scene_field is None:
        return {}

    # Batch column access is fast via Arrow
    vals = hf_ds[scene_field]
    groups = {}
    for i, val in enumerate(vals):
        # For 'id' fields like "scene123_frame456", take the scene prefix
        key = str(val).split("_")[0] if scene_field == "id" else str(val)
        groups.setdefault(key, []).append(i)

    prev_map = {}
    for indices in groups.values():
        for j, idx in enumerate(indices):
            if j > 0:
                # Up to 2 previous frames, in chronological order
                prev_map[idx] = indices[max(0, j - 2) : j]
    return prev_map


class LazyDataset:
    """Wraps a raw HuggingFace dataset and converts samples on-the-fly.
    Avoids Arrow serialisation errors from mixed-type message content.
    Uses temporal context (prev frames) when scene metadata is available."""

    def __init__(self, hf_dataset):
        self._ds = hf_dataset
        self._prev_map = _build_temporal_index(hf_dataset)
        if self._prev_map:
            print(f"Temporal index built: {len(self._prev_map)}/{len(hf_dataset)} samples have prev frames")

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        prev_imgs = None
        if idx in self._prev_map:
            prev_imgs = [_load_img(self._ds[p]["image"]) for p in self._prev_map[idx]]
        return _to_messages(self._ds[idx], prev_imgs)


def load_drivelm(split: str = "train") -> LazyDataset:
    ds = load_dataset(DRIVELM_REPO, DRIVELM_CONFIG, split=split)
    return LazyDataset(ds)


def train_val_split(ds: LazyDataset, val_ratio: float = 0.05):
    n = len(ds)
    val_n = max(1, int(n * val_ratio))
    indices = list(range(n))
    random.seed(42)
    random.shuffle(indices)
    train_idx, val_idx = indices[val_n:], indices[:val_n]

    class _Subset:
        def __init__(self, ds, idx): self._ds, self._idx = ds, idx
        def __len__(self): return len(self._idx)
        def __getitem__(self, i): return self._ds[self._idx[i]]

    return _Subset(ds, train_idx), _Subset(ds, val_idx)


def synthetic_dataset(n: int = 32) -> "LazyDataset":
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
