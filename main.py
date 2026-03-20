import sys
import json
from pathlib import Path

from pipeline import extract_frames, sliding_windows, N_FRAMES
from prompt import build_messages
from model import infer
from output import parse, render

if len(sys.argv) < 2:
    print("usage: python main.py <video_path> [sample_fps]")
    sys.exit(1)

video_path = sys.argv[1]
sample_fps = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

frames = extract_frames(video_path, sample_fps)
windows = list(sliding_windows(frames, n=N_FRAMES))
print(f"{len(windows)} windows to process\n")

results = []
for i, window in enumerate(windows):
    print(f"=== window {i + 1}/{len(windows)} ===")
    raw = infer(build_messages(window))
    try:
        data = parse(raw)
        render(data)
        results.append({"window": i + 1, "ok": True, "data": data})
    except Exception as e:
        print(f"parse error: {e}\nraw: {raw}\n")
        results.append({"window": i + 1, "ok": False, "raw": raw})

out_path = Path(video_path).stem + "_results.json"
Path(out_path).write_text(json.dumps(results, indent=2))
print(f"Results saved to {out_path}")
