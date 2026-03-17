import argparse
import sys
from prompt import build_messages
from output import parse, render_rich, render_json


def main():
    ap = argparse.ArgumentParser(description="DriveVLM — driving scene analyser")
    ap.add_argument("--image", required=True, help="Path to dashcam image")
    ap.add_argument("--json", action="store_true", help="Output raw JSON")
    ap.add_argument("--mode", choices=["prompt", "ft"], default="prompt",
                    help="prompt=base model, ft=load LoRA adapter")
    ap.add_argument("--adapter", default=None,
                    help="Path to LoRA adapter dir (required for --mode ft)")
    ap.add_argument("--compare", action="store_true",
                    help="Run both modes side-by-side (requires --adapter)")
    args = ap.parse_args()

    if args.mode == "ft" and not args.adapter:
        print("error: --mode ft requires --adapter", file=sys.stderr)
        sys.exit(1)
    if args.compare and not args.adapter:
        print("error: --compare requires --adapter", file=sys.stderr)
        sys.exit(1)

    msgs = build_messages(args.image)

    if args.compare:
        from model import infer, infer_ft
        raw_base = infer(msgs)
        raw_ft = infer_ft(msgs, args.adapter)
        try:
            base_data = parse(raw_base)
            ft_data = parse(raw_ft)
        except Exception as e:
            print(f"error parsing output: {e}", file=sys.stderr)
            sys.exit(1)
        from output import render_compare
        render_compare(base_data, ft_data)
        return

    if args.mode == "ft":
        from model import infer_ft
        raw = infer_ft(msgs, args.adapter)
    else:
        from model import infer
        raw = infer(msgs)

    try:
        data = parse(raw)
    except Exception as e:
        print(f"warning: could not parse model output as JSON: {e}", file=sys.stderr)
        print(raw)
        sys.exit(1)

    if args.json:
        print(render_json(data))
    else:
        render_rich(data)


if __name__ == "__main__":
    main()
