import sys
import json
from prompt import build_messages
from model import infer
from output import parse, render

if len(sys.argv) < 2:
    print("usage: python main.py <image_path>")
    sys.exit(1)

msgs = build_messages(sys.argv[1])
raw = infer(msgs)

try:
    data = parse(raw)
except Exception as e:
    print(f"parse error: {e}\nraw output:\n{raw}")
    sys.exit(1)

render(data)
