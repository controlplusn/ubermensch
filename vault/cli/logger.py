from __future__ import annotations
 
import time
from rich.console import Console
from rich.text import Text
from rich.theme import Theme
 
_theme = Theme(
    {
        "step":  "bold cyan",
        "done":  "bold green",
        "warn":  "bold yellow",
        "fail":  "bold red",
        "dim":   "dim white",
        "label": "bold white",
    }
)

console = Console(theme=_theme, highlight=False)
 
# Track timing per named step
_timers: dict[str, float] = {}


def step(tag: str, message: str) -> None:
    _timers[tag] = time.perf_counter()
    _print_line("●", "step", tag, message)

def log(message: str) -> None:
    console.print(f"  [dim]│[/dim]  {message}")

def done(tag: str, message: str) -> None:
    elapsed = time.perf_counter() - _timers.pop(tag, time.perf_counter())
    _print_line("✓", "done", tag, f"{message}  [dim]({elapsed:.2f}s)[/dim]")

def warn(tag: str, message: str) -> None:
    _print_line("⚠", "warn", tag, message)

def fail(tag: str, message: str) -> None:
    _print_line("✗", "fail", tag, message)

def section(title: str) -> None:
    console.rule(f"[dim]{title}[/dim]", style="dim")
 
def blank() -> None:
    console.print()
 
 
def _print_line(icon: str, style: str, tag: str, message: str) -> None:
    tag_text = Text(f" {tag:<10}", style="label")
    msg_text = Text.from_markup(message)
    line = Text()
    line.append(f"{icon} ", style=style)
    line.append_text(tag_text)
    line.append("  ")
    line.append_text(msg_text)
    console.print(line)