from __future__ import annotations

import vault
import typer
import os
import yaml

from pathlib import Path
from typing import Optional

from rich.columns import Columns
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

from vault.cli.logger import blank, log, section, step, done, warn
from vault.config.discovery import get_or_prompt_vault, save_config, CONFIG_PATH
from vault.core.ingester import ingest_vault
from vault.retrieval.chunker import chunk_notes
from vault.retrieval.embedder import embed_chunks
from vault.retrieval.store import upsert_chunks, store_stats

from vault.agents.ask import run_ask


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

    # Clear cache if --force
    if force:
        from vault.cli.logger import warn
        warn("CACHE", "--force flag set: clearing hash cache for full re-index")
        cache = Path.home() / ".vault" / "cache" / "hashes.json"
        if cache.exists():
            cache.unlink()
            from vault.cli.logger import log as _log
            _log("Hash cache cleared")


    # Step 1: Ingest
    notes = ingest_vault(vault_path)
    if not notes:
        console.print("[yellow]No notes found. Check your vault path.[/yellow]")
        raise typer.Exit(0)
    blank()

    # Step 2: Chunk
    chunks = chunk_notes(notes)
    if not chunks:
        console.print("[yellow]No chunks produced. Notes may be empty.[/yellow]")
        raise typer.Exit(0)
    blank()
 
    # Step 3: Embed
    texts = [c.text for c in chunks]
    embeddings = embed_chunks(texts)
    blank()

    # Step 4: Store
    upsert_chunks(chunks, embeddings)
    blank()

    # Summary
    _print_summary(notes, vault_path, total_chunks=len(chunks))


# Vault ask

@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask against your vault"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Chunks to retrieve per attempt"),
    eval_mode: bool = typer.Option(
        False, "--eval",
        help="Show detailed per-claim faithfulness breakdown",
    ),
    # TODO: test the --no-eval helper
    no_eval: bool = typer.Option(
        False, "--no-eval",
        help="Skip faithfulness scoring entirely (faster)",
    ),
    key: Optional[str] = typer.Option(None, "--key", help="Gemini API key"),
) -> None:
    api_key = key or os.environ.get("GEMINI_API_KEY", "")

    result = run_ask(
        query=question,
        top_k=top_k,
        api_key=api_key,
        show_eval=not no_eval,
        eval_verbose=eval_mode,     # --eval
    )
 
    blank()
    _print_ask_result(result, show_eval=not no_eval, eval_verbose=eval_mode)


# Vault status

@app.command()
def status() -> None:
    section("Vault status")
    vault_path = get_or_prompt_vault()

    step("INDEX", "Loading vault index")
    log("Parsing notes and resolving links (cached notes are instant)")
    notes = ingest_vault(vault_path)

    blank()
    stats = store_stats()
    blank()
    _print_summary(notes, vault_path, total_chunks=stats["total_chunks"])


# Vault config

@app.command(name="config")
def config_cmd(
    reset: bool = typer.Option(False, "--reset", help="Clear saved config"),
    set_key: Optional[str] = typer.Option(None, "--set-key", help="Save Gemini API key"),
) -> None:

    section("Config")

    if set_key:
        step("KEY", "Saving Gemini API key to config")
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        cfg = {}
        if CONFIG_PATH.exists():
            cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        cfg["gemini_api_key"] = set_key
        CONFIG_PATH.write_text(yaml.dump(cfg))
        log(f"Key saved to {CONFIG_PATH}")
        done("KEY", "API key saved — you won't need --key on future runs")
        return

    if reset:
        step("RESET", "Clearing config")
        if CONFIG_PATH.exists():
            CONFIG_PATH.unlink()
            log(f"Deleted: {CONFIG_PATH}")
            done("RESET", "Config cleared — next launch runs setup again")
        else:
            log("No config file found")
        return

    if not CONFIG_PATH.exists():
        console.print("[dim]No config saved. Run [bold]vault init[/bold] first.[/dim]")
        return

    cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    step("READ", f"Config: {CONFIG_PATH}")
    for k, v in cfg.items():
        display = "****" + str(v)[-4:] if "key" in k.lower() else str(v)
        log(f"  {k}: [bold]{display}[/bold]")
    done("READ", "Config loaded") 


# Display helpers

def _print_ask_result(result, show_eval: bool = True, eval_verbose: bool = False) -> None:
    # Retry notice
    if result.attempts > 1:
        console.print(
            f"  [dim]ℹ Retrieved after {result.attempts} attempt(s) "
            f"(re-retrieval loop expanded the query for better coverage)[/dim]\n"
        )

    # Sources
    console.print(Rule("[dim]Sources retrieved[/dim]", style="dim"))
    src_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    src_table.add_column(style="dim", no_wrap=True)
    src_table.add_column(style="cyan")
    src_table.add_column(style="dim")
    for i, chunk in enumerate(result.retrieved_chunks, 1):
        src_table.add_row(f"{i}.", chunk.note_title, f"relevance {chunk.score:.3f}")
    console.print(src_table)

    # Answer panel
    console.print(Rule("[dim]Answer[/dim]", style="dim"))
    console.print(Padding(result.answer, (1, 2)))

    # Faithfulness Score
    if show_eval and result.eval_result is not None:
        er = result.eval_result
        color = er.confidence_color
        label = er.confidence_label
        console.print(Rule(style="dim"))
        console.print(
            Padding(
                f"[{color}]Faithfulness: {er.score:.1%}[/{color}]"
                f"  [dim]{label}[/dim]"
                f"  [dim]({er.supported_count}/{er.total_count} claims)[/dim]"
                f"  [dim]· {result.model}[/dim]",
                (0, 2),
            )
        )

        # --eval: per-claim table
        if eval_verbose and er.claims:
            console.print()
            console.print(Rule("[dim]Claim-level breakdown (--eval)[/dim]", style="dim"))
            claim_table = Table(
                box=box.ROUNDED,
                show_header=True,
                header_style="dim",
                padding=(0, 1),
                expand=True,
            )
            claim_table.add_column("#",        style="dim",  width=3,  no_wrap=True)
            claim_table.add_column("Claim",    style="white", ratio=5)
            claim_table.add_column("Overlap",  style="dim",  width=9,  no_wrap=True)
            claim_table.add_column("Verdict",  width=14,     no_wrap=True)
            claim_table.add_column("Matched words", style="dim", ratio=2)
 
            for i, cr in enumerate(er.claims, 1):
                verdict = (
                    "[green]✓ supported[/green]"
                    if cr.supported
                    else "[red]✗ unsupported[/red]"
                )
                matched_str = ", ".join(cr.matched[:6]) + ("…" if len(cr.matched) > 6 else "")
                claim_table.add_row(
                    str(i),
                    cr.text[:120] + ("…" if len(cr.text) > 120 else ""),
                    f"{cr.overlap:.0%}",
                    verdict,
                    matched_str or "[dim]none[/dim]",
                )
 
            console.print(Padding(claim_table, (0, 2)))
 
            if er.skipped:
                console.print(
                    f"\n  [dim]  {er.skipped} meta-sentence(s) excluded from scoring "
                    f"(e.g. 'I couldn't find...')[/dim]"
                )
 
    console.print()


def _print_summary(notes: list, vault_path: Path, total_chunks: int = 0) -> None:
    if not notes:
        console.print(Panel("[yellow]No notes indexed.[/yellow]"))
        return
    
    total_notes = len(notes)
    total_words = sum(len(n.content.split()) for n in notes)
    total_links = sum(len(n.wikilinks) for n in notes)
    total_bl = sum(len(n.backlinks) for n in notes)
    total_tags = sum(len(n.tags) for n in notes)
    orphans = sum(1 for n in notes if not n.wikilinks and not n.backlinks)

    tag_freq: dict[str, int] = {}
    for n in notes:
        for t in n.tags:
            tag_freq[t] = tag_freq.get(t, 0) + 1
    top_tags = sorted(tag_freq.items(), key=lambda x: -x[1])[:12]
    top_linked  = sorted(notes, key=lambda n: len(n.backlinks), reverse=True)[:5]

    # Header
    header = Text()
    header.append(f"{vault_path.name}", style="bold cyan")
    header.append(f"  {vault_path}\n", style="dim")
    console.print(Panel(header, box=box.ROUNDED, border_style="cyan", padding=(0, 1)))
 
    # Stats
    stats_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    stats_table.add_column(style="dim", no_wrap=True)
    stats_table.add_column(style="bold white", no_wrap=True)
    stats_table.add_row("Notes",         str(total_notes))
    stats_table.add_row("Words",         f"{total_words:,}")
    stats_table.add_row("Chunks indexed", str(total_chunks) if total_chunks else "[dim]run vault init[/dim]")
    stats_table.add_row("Wikilinks",     str(total_links))
    stats_table.add_row("Backlinks",     str(total_bl))
    stats_table.add_row("Total tags",    str(total_tags))
    stats_table.add_row("Orphan notes",  str(orphans))
    if total_notes:
        stats_table.add_row("Link density", f"{total_links / total_notes:.1f} links/note")
    console.print(Padding(stats_table, (0, 0, 0, 2)))
 
    # Top tags
    if top_tags:
        console.print("\n  [dim]Top tags[/dim]")
        items = [
            Text.from_markup(f"  [cyan]#{t}[/cyan] [dim]{c}[/dim]")
            for t, c in top_tags
        ]
        console.print(Padding(Columns(items, equal=False, expand=False), (0, 0, 0, 2)))

    # Most referenced
    if any(n.backlinks for n in top_linked):
        console.print("\n  [dim]Most referenced notes[/dim]")
        lt = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        lt.add_column(style="bold white")
        lt.add_column(style="dim")
        lt.add_column(style="dim")
        for n in top_linked:
            if n.backlinks:
                lt.add_row(n.title, f"{len(n.backlinks)} backlinks", f"{len(n.wikilinks)} outbound")
        console.print(Padding(lt, (0, 0, 0, 2)))
 
    # Hints
    if orphans > 0:
        console.print(
            f"\n  [dim]💡 {orphans} orphan note(s) found. "
            "Run [bold]vault graph suggest[/bold] to discover connections.[/dim]"
        )
    if total_chunks > 0:
        console.print(
            f"\n  [dim]✓ {total_chunks} chunks indexed. "
            "Run [bold]vault ask \"your question\"[/bold] to query your vault.[/dim]"
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