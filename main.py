import sys
from prompt import build_messages
from model import infer
from output import parse, render

frames = sys.argv[1:]
if len(frames) < 2:
    print("usage: python main.py <frame_t-2> <frame_t-1> <frame_t>  (oldest first, min 2 frames)")
    sys.exit(1)

msgs = build_messages(frames)
raw = infer(msgs)

try:
    data = parse(raw)
except Exception as e:
    print(f"parse error: {e}\nraw output:\n{raw}")
    sys.exit(1)

render(data)
