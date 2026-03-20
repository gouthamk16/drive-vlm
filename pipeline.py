"""
Video processing pipeline.

Extracts frames from a dashcam video at a fixed sample rate,
then yields sliding windows of N frames for inference.
"""

import cv2
from PIL import Image

SAMPLE_FPS = 2.0   # frames to sample per second (matches nuScenes keyframe rate)
N_FRAMES   = 3     # frames per inference window: [T-2, T-1, T]


def extract_frames(video_path: str, sample_fps: float = SAMPLE_FPS) -> list:
    """Extract frames from a video at sample_fps. Returns list of PIL Images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(video_fps / sample_fps))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_fps:.1f}fps, {total} frames -> sampling every {step} frames (~{sample_fps}Hz)")

    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames


def sliding_windows(frames: list, n: int = N_FRAMES):
    """Yield [oldest, ..., current] windows of length n over a frame list."""
    for i in range(n - 1, len(frames)):
        yield frames[i - n + 1 : i + 1]
