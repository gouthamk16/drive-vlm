import json
import re
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def parse(raw: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned)
    return json.loads(cleaned)


def render_rich(data: dict):
    console.rule("[bold cyan]Perception")
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("Object"), t.add_column("Location"), t.add_column("State")
    for p in data.get("perception", []):
        t.add_row(p["object"], p["location"], p["state"])
    console.print(t)

    console.rule("[bold yellow]Prediction")
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("Subject"), t.add_column("Action"), t.add_column("Confidence")
    for p in data.get("prediction", []):
        t.add_row(p["subject"], p["action"], p["confidence"])
    console.print(t)

    console.rule("[bold green]Planning")
    plan = data.get("planning", {})
    console.print(f"[bold]Action:[/bold] {plan.get('action', '')}")
    console.print(f"[bold]Reason:[/bold] {plan.get('reason', '')}")
    factors = plan.get("causal_factors", [])
    if factors:
        console.print(f"[bold]Causal factors:[/bold] {', '.join(factors)}")


def render_compare(base_data: dict, ft_data: dict):
    from rich.columns import Columns
    from rich.panel import Panel

    console.rule("[bold]Prompt-only vs Fine-tuned comparison")
    console.print(Columns([
        Panel(json.dumps(base_data, indent=2), title="[cyan]Prompt-only", border_style="cyan"),
        Panel(json.dumps(ft_data, indent=2), title="[green]Fine-tuned", border_style="green"),
    ]))
