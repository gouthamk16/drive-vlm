import sys
import json
from prompt import build_messages
from model import infer
from output import parse, render

frames = sys.argv[1:]
if not frames:
    print("usage: python main.py <frame1> [frame2 frame3 ...]  (oldest first)")
    sys.exit(1)

msgs = build_messages(frames)
raw = infer(msgs)

try:
    data = parse(raw)
except Exception as e:
    print(f"parse error: {e}\nraw output:\n{raw}")
    sys.exit(1)

render(data)
