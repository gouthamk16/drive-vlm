import json
import re


def parse(raw: str) -> dict:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def render(data: dict):
    scene = data.get("scene", {})
    if scene:
        print(f"\nScene: {scene.get('weather')} | {scene.get('time')} | {scene.get('road')} | ego in {scene.get('ego_lane')} lane")

    print("\nPerception:")
    for p in data.get("perception", []):
        risk = p.get("risk", "")
        loc = f"{p.get('direction','?')} {p.get('distance','?')}"
        print(f"  [{risk}] {p['object']} | {loc} | {p['state']}")

    print("\nPrediction:")
    for p in data.get("prediction", []):
        print(f"  {p.get('subject','')} -> {p.get('action','')} ({p.get('confidence','')})")
        influence = p.get("influence", "")
        if influence:
            print(f"      {influence}")

    plan = data.get("planning", {})
    print("\nPlanning:")
    print(f"  {plan.get('meta_action', '').upper()} | {plan.get('reason', '')}")
    factors = plan.get("causal_factors", [])
    if factors:
        print(f"  caused by: {', '.join(factors)}")
    print()


def compare(base: dict, ft: dict):
    for section in ["scene", "perception", "prediction", "planning"]:
        print(f"\n--- {section} ---")
        print(f"  base: {json.dumps(base.get(section), indent=2)}")
        print(f"    ft: {json.dumps(ft.get(section), indent=2)}")
