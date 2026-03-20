from datasets import load_dataset
from PIL import Image, ImageDraw
import io, random

DRIVELM_REPO   = "OpenDriveLab/DriveLM"
DRIVELM_CONFIG = "default"
N_FRAMES       = 3    # frames per training sample: [T-2, T-1, T]
FPS            = 2.0  # nuScenes keyframe rate


def _load_img(raw) -> Image.Image:
    if isinstance(raw, bytes):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    return Image.fromarray(raw).convert("RGB")


def _to_messages(sample, prev_imgs: list) -> dict:
    """Build a {messages: [...]} dict with a video content block.
    prev_imgs: list of PIL Images [T-2, T-1, ...]. If empty, current frame
    is duplicated to fill N_FRAMES — standard temporal padding for first frames."""
    img = _load_img(sample["image"])
    frames = prev_imgs + [img]

    # Pad to N_FRAMES by repeating the oldest frame at the front
    while len(frames) < N_FRAMES:
        frames = [frames[0]] + frames

    visual = {"type": "video", "video": frames, "fps": FPS}

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
    """Build {idx -> [prev_idx, ...]} using scene metadata.

    DriveLM samples share a scene_token field; frames within a scene appear
    in chronological order. If no usable field exists, returns {} and the
    dataset falls back to duplicate-padding every sample."""
    cols = getattr(hf_ds, "column_names", [])
    scene_field = next((f for f in ["scene_token", "id"] if f in cols), None)
    if scene_field is None:
        return {}

    vals = hf_ds[scene_field]   # fast Arrow column access
    groups = {}
    for i, val in enumerate(vals):
        # For 'id' fields like "scene123_frame456", the scene prefix is before '_'
        key = str(val).split("_")[0] if scene_field == "id" else str(val)
        groups.setdefault(key, []).append(i)

    prev_map = {}
    for indices in groups.values():
        for j, idx in enumerate(indices):
            # Store up to (N_FRAMES - 1) previous indices in chronological order
            prev_map[idx] = indices[max(0, j - (N_FRAMES - 1)) : j]
    return prev_map


class LazyDataset:
    """Wraps a HuggingFace dataset; converts samples on-the-fly to video sequences.
    Always produces N_FRAMES-frame video blocks — no single-frame path."""

    def __init__(self, hf_dataset):
        self._ds = hf_dataset
        self._prev_map = _build_temporal_index(hf_dataset)
        if self._prev_map:
            n = sum(1 for v in self._prev_map.values() if v)
            print(f"Temporal index: {n}/{len(hf_dataset)} samples have real prev frames "
                  f"({len(hf_dataset) - n} will be padded)")
        else:
            print("No scene metadata found — all samples will be duplicate-padded")

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        prev_imgs = [_load_img(self._ds[p]["image"]) for p in self._prev_map.get(idx, [])]
        return _to_messages(self._ds[idx], prev_imgs)


def load_drivelm(split: str = "train") -> LazyDataset:
    ds = load_dataset(DRIVELM_REPO, DRIVELM_CONFIG, split=split)
    return LazyDataset(ds)


def train_val_split(ds, val_ratio: float = 0.05):
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


def synthetic_dataset(n: int = 32):
    """Synthetic N_FRAMES-frame video sequences for pipeline validation."""
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
        def __init__(self, n): self._n = n

        def __len__(self): return self._n

        def __getitem__(self, idx):
            random.seed(idx)
            # Slight brightness shift per frame simulates motion
            frames = []
            for t in range(N_FRAMES):
                base = random.randint(80, 180)
                img = Image.new("RGB", (224, 224), color=(base + t * 5,) * 3)
                ImageDraw.Draw(img).rectangle([50 + t * 3, 80, 174 + t * 3, 144], fill=(50, 50, 50))
                frames.append(img)
            q, a = qa_pairs[idx % len(qa_pairs)]
            return {"messages": [
                {"role": "user", "content": [
                    {"type": "video", "video": frames, "fps": FPS},
                    {"type": "text", "text": q},
                ]},
                {"role": "assistant", "content": a},
            ]}

    return _SyntheticDataset(n)
