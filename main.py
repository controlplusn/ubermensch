from __future__ import annotations

import vault
import typer

from pathlib import Path
from typing import Optional

from rich.columns import Columns
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from vault.cli.logger import blank, log, section, step, done
from vault.config.discovery import get_or_prompt_vault, save_config
from vault.core.ingester import ingest_vault
from vault.cli.logger import blank, log, section, step, done
from vault.config.discovery import CONFIG_PATH


app = typer.Typer(
    name="vault",
    help="Local-first agentic knowledge system for Obsidian vaults.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


# Vault init

@app.command()
def init(
    path: Optional[Path] = typer.Option(
        None,
        "--path", "-p",
        help="Override vault path (skips auto-detection)",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Re-index even if nothing changed",
    ),
) -> None:

    if path:
        section("Vault init (manual path)")
        step("PATH", f"Using provided path: {path}")
        vault_path = path.expanduser().resolve()
        if not vault_path.exists():
            console.print(f"[bold red]Error:[/bold red] Path not found: {vault_path}")
            raise typer.Exit(1)
        log(f"Resolved: {vault_path}")
        done("PATH", "Path accepted")
        save_config(vault_path)
    else:
        vault_path = get_or_prompt_vault()

    if force:
        from vault.cli.logger import warn
        warn("CACHE", "--force flag set: clearing hash cache for full re-index")
        cache = Path.home() / ".vault" / "cache" / "hashes.json"
        if cache.exists():
            cache.unlink()
            from vault.cli.logger import log as _log
            _log("Hash cache cleared")

    notes = ingest_vault(vault_path)

    blank()
    _print_summary(notes, vault_path)


# Vault status

@app.command()
def status() -> None:
    from vault.cli.logger import blank, log, section, step, done, warn
    from vault.config.discovery import get_or_prompt_vault
    from vault.core.ingester import injest_vault

    section("Vault status")
    vault_path = get_or_prompt_vault()

    step("INDEX", "Loading vault index")
    log("Parsing notes and resolving links (cached notes are instant)")
    notes = ingest_vault(vault_path)

    blank()
    _print_summary(notes, vault_path)


# Vault config

@app.command(name="config")
def config_cmd(
    reset: bool = typer.Option(False, "--reset", help="Clear saved config and re-run setup"),
) -> None:

    section("Config")

    if reset:
        step("RESET", "Clearing saved config")
        if CONFIG_PATH.exists():
            CONFIG_PATH.unlink()
            log(f"Deleted: {CONFIG_PATH}")
            done("RESET", "Config cleared — next launch will run setup again")
        else:
            log("No config file found — nothing to clear")
        return

    if not CONFIG_PATH.exists():
        console.print("[dim]No config saved yet. Run [bold]vault init[/bold] to set up.[/dim]")
        return

    import yaml
    cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    step("READ", f"Config file: {CONFIG_PATH}")
    for key, value in cfg.items():
        log(f"  {key}: [bold]{value}[/bold]")
    done("READ", "Config loaded")


# Shared display helpers

def _print_summary(notes: list, vault_path: Path) -> None:
    if not notes:
        console.print(Panel("[yellow]No notes indexed.[/yellow]", title="Vault"))
        return

    total_notes = len(notes)
    total_words = sum(len(n.content.split()) for n in notes)
    total_links = sum(len(n.wikilinks) for n in notes)
    total_bl = sum(len(n.backlinks) for n in notes)
    total_tags = sum(len(n.tags) for n in notes)
    orphans = sum(1 for n in notes if not n.wikilinks and not n.backlinks)

    # Tag frequency map
    tag_freq: dict[str, int] = {}
    for n in notes:
        for t in n.tags:
            tag_freq[t] = tag_freq.get(t, 0) + 1

    top_tags = sorted(tag_freq.items(), key=lambda x: -x[1])[:12]

    # Top linked notes (most backlinks)
    top_linked = sorted(notes, key=lambda n: len(n.backlinks), reverse=True)[:5]

    # Header panel
    header = Text()
    header.append(f"{vault_path.name}", style="bold cyan")
    header.append(f"  {vault_path}\n", style="dim")
    console.print(Panel(header, box=box.ROUNDED, border_style="cyan", padding=(0, 1)))

    # State grid
    stats_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    stats_table.add_column(style="dim", no_wrap=True)
    stats_table.add_column(style="bold white", no_wrap=True)

    stats_table.add_row("Notes", str(total_notes))
    stats_table.add_row("Total words", f"{total_words:,}")
    stats_table.add_row("Wikilinks", str(total_links))
    stats_table.add_row("Backlinks", str(total_bl))
    stats_table.add_row("Tagged notes", str(sum(1 for n in notes if n.tags)))
    stats_table.add_row("Total tags", str(total_tags))
    stats_table.add_row("Orphan notes", str(orphans))
    if total_notes:
        density = total_links / total_notes
        stats_table.add_row("Link density",  f"{density:.1f} links/note")

    console.print(Padding(stats_table, (0, 0, 0, 2)))

    # Top tags
    if top_tags:
        console.print("\n  [dim]Top tags[/dim]")
        tag_items = [
            Text.from_markup(f"  [cyan]#{tag}[/cyan] [dim]{count}[/dim]")
            for tag, count in top_tags
        ]
        console.print(Padding(Columns(tag_items, equal=False, expand=False), (0, 0, 0, 2)))

    # Most linked notes
    if any(n.backlinks for n in top_linked):
        console.print("\n  [dim]Most referenced notes[/dim]")
        link_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        link_table.add_column(style="bold white")
        link_table.add_column(style="dim", no_wrap=True)
        link_table.add_column(style="dim")

        for n in top_linked:
            if n.backlinks:
                link_table.add_row(
                    n.title,
                    f"{len(n.backlinks)} backlinks",
                    f"{len(n.wikilinks)} outbound",
                )
        console.print(Padding(link_table, (0, 0, 0, 2)))

    # Orphan note hint
    if orphans > 0:
        console.print(
            f"\n  [dim]💡 {orphans} orphan note(s) have no links. "
            f"Run [bold]vault graph suggest[/bold] to discover connections.[/dim]"
        )

    console.print()


# __init__ stubs for submodules

def _ensure_inits() -> None:
    """Create __init__.py files for all subpackages at import time."""
    base = Path(vault.__file__).parent
    for sub in ("cli", "config", "core", "storage"):
        init = base / sub / "__init__.py"
        init.parent.mkdir(exist_ok=True)
        if not init.exists():
            init.write_text("")


_ensure_inits()


if __name__ == "__main__":
    app()