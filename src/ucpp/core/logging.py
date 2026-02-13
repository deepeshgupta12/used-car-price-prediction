from __future__ import annotations

from rich.console import Console

console = Console()


def info(msg: str) -> None:
    console.print(f"[bold]INFO[/bold] {msg}")


def warn(msg: str) -> None:
    console.print(f"[yellow]WARN[/yellow] {msg}")


def error(msg: str) -> None:
    console.print(f"[red]ERROR[/red] {msg}")
