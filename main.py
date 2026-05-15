from __future__ import annotations

import sys

# Windows defaults to cp1252; Rich uses Unicode (bullets, box drawing). Without
# UTF-8, Console.print raises UnicodeEncodeError and commands like `status` appear blank.
if sys.platform == "win32":
    for _stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        if _stream is not None and hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass

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
from vault.retrieval.store import upsert_chunks, store_stats, get_collection
from vault.agents.ask import run_ask
from vault.cli.logger import blank, done, log, step, warn
from vault.graph.builder import build_graph, load_graph
from vault.graph.suggester import find_suggestions
from vault.graph.mapper import map_topic
from rich.prompt import Confirm
from vault.agents.writer import write_backlink_pair



app = typer.Typer(
    name="vault",
    help="Local-first agentic knowledge system for Obsidian vaults.",
    add_completion=False,
    rich_markup_mode="rich",
)
graph_app = typer.Typer(help="Knowledge graph operations.")
app.add_typer(graph_app, name="graph")
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


# Vault graph build

@graph_app.command("build")
def graph_build(
    path: Optional[Path] = typer.Option(None, "--path", "-p"),
    no_semantic: bool = typer.Option(False, "--no-semantic", help="Skip semantic edges"),
) -> None:
    vault_path = path.expanduser().resolve() if path else get_or_prompt_vault()
    notes = ingest_vault(vault_path)

    if not notes:
        console.print("[yellow]No notes found.[/yellow]")
        raise typer.Exit(0)
    
    blank()
 
    note_embeddings = None

    if not no_semantic:
        step("EMBEDDINGS", "Loading note embeddings for semantic edges")
        log("Averaging chunk embeddings per note -> one vector per note")

        try:
            collection = get_collection()
            results = collection.get(include=["embeddings", "metadatas"])
            note_emb_acc: dict[str, list] = {}

            for emb, meta in zip(results["embeddings"], results["metadatas"]):
                title = meta.get("note_title", "")

                if title:
                    note_emb_acc.setdefault(title, []).append(emb)

            note_embeddings = {}

            for title, emb_list in note_emb_acc.items():
                n, dim = len(emb_list), len(emb_list[0])
                note_embeddings[title] = [sum(e[i] for e in emb_list) / n for i in range(dim)]

            done("EMBEDDINGS", f"Ready for {len(note_embeddings)} notes")
            blank()
        except Exception as exc:
            warn("EMBEDDINGS", f"Could not load embeddings: {exc}")
            log("Run vault init first, then vault graph build")

            note_embeddings = None

            blank()
 
    build_graph(notes, embeddings=note_embeddings)


# Vault graph suggest

@graph_app.command("suggest")
def graph_suggest(
    top_n: int = typer.Option(15, "--top", "-n", help="Number of suggestions"),
    confirm: bool = typer.Option(False, "--confirm", help="Interactively write to vault"),
    dry_run: bool = typer.Option(True, "--dry-run/--write", help="Preview only (default)"),
) -> None:
    G = load_graph()

    if G is None:
        console.print("[yellow]Run [bold]vault graph build[/bold] first.[/yellow]")
        raise typer.Exit(1)
 
    suggestions = find_suggestions(G, top_n=top_n)
    blank()
 
    if not suggestions:
        console.print("[dim]No backlink suggestions found.[/dim]")
        return
 
    _print_suggestions(suggestions)
 
    if confirm:
        _interactive_confirm(suggestions, G, dry_run=dry_run)

    
# Vault graph map
@graph_app.command("map")
def graph_map(
    topic: str = typer.Argument(..., help="Note title or keyword to map around"),
    depth: int = typer.Option(2, "--depth", "-d", help="BFS hops from seed (1-3)"),
) -> None:
    G = load_graph()

    if G is None:
        console.print("[yellow]Run [bold]vault graph build[/bold] first.[/yellow]")
        raise typer.Exit(1)
 
    nodes = map_topic(G, topic, max_depth=min(depth, 3))
    blank()
 
    if not nodes:
        console.print(f"[yellow]No notes found matching '{topic}'.[/yellow]")
        return
 
    _print_map(nodes, topic)


# Vault status

@app.command()
def status() -> None:
    section("Vault status")
    vault_path = Path(get_or_prompt_vault())

    step("INDEX", "Loading vault index")
    notes = ingest_vault(vault_path)
    done("INDEX", f"{len(notes)} notes loaded")

    stats = store_stats()
    G = load_graph()
    blank()

    _print_summary(notes, vault_path, total_chunks=stats["total_chunks"], graph=G)


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


def _print_suggestions(suggestions) -> None:
    console.print(Rule("[dim]Backlink suggestions[/dim]", style="dim"))
    console.print("  [dim]These notes discuss similar topics but are not linked yet.[/dim]\n")
 
    t = Table(box=box.ROUNDED, show_header=True, header_style="dim",
              padding=(0, 1), expand=True)
    
    t.add_column("#", style="dim",  width=3,  no_wrap=True)
    t.add_column("Note A", style="cyan", ratio=3)
    t.add_column("Note B", style="cyan", ratio=3)
    t.add_column("Score", style="bold", width=7,  no_wrap=True)
    t.add_column("Sim", style="dim",  width=6,  no_wrap=True)
    t.add_column("Shared tags", style="dim",  ratio=2)
 
    for i, s in enumerate(suggestions, 1):
        color = "green" if s.score >= 0.8 else "yellow" if s.score >= 0.7 else "white"
        tags  = ", ".join(f"#{t_}" for t_ in s.shared_tags[:3]) or "[dim]—[/dim]"

        t.add_row(str(i), s.source, s.target,
                  f"[{color}]{s.score:.3f}[/{color}]",
                  f"{s.semantic_score:.2f}", tags)
 
    console.print(Padding(t, (0, 2)))
    console.print(
        "\n  [dim]To interactively write to your vault:[/dim]\n"
        "  [dim]vault graph suggest --confirm --write[/dim]\n"
    )


def _print_map(nodes, topic: str) -> None:
    console.print(Rule(f"[dim]Idea map: {topic}[/dim]", style="dim"))
    depth_colors = {0: "bold cyan", 1: "white", 2: "dim"}
 
    t = Table(box=box.ROUNDED, show_header=True, header_style="dim",
              padding=(0, 1), expand=True)
    t.add_column("Depth", style="dim", width=7,  no_wrap=True)
    t.add_column("Note", ratio=4)
    t.add_column("Connection", style="dim", width=12, no_wrap=True)
    t.add_column("Weight", style="dim", width=8,  no_wrap=True)
    t.add_column("Tags", style="dim", ratio=2)
 
    for node in nodes:
        color = depth_colors.get(node.depth, "dim")
        label = "seed" if node.depth == 0 else f"depth {node.depth}"
        conn = "+".join(node.edge_types) if node.edge_types else "—"
        tags = ", ".join(f"#{tg}" for tg in node.tags[:3]) or "—"
        t.add_row(label,
                  f"[{color}]{node.title}[/{color}]",
                  conn,
                  f"{node.edge_weight:.2f}" if node.depth > 0 else "—",
                  tags)
 
    console.print(Padding(t, (0, 2)))
    max_d = max(n.depth for n in nodes)
    console.print(f"\n  [dim]{len(nodes)} notes connected to '{topic}' within {max_d} hop(s)[/dim]\n")


def _interactive_confirm(suggestions, G, dry_run=True) -> None:
 
    console.print(Rule("[dim]Interactive confirmation[/dim]", style="dim"))
    
    if dry_run:
        console.print("  [yellow]DRY RUN — use --confirm --write to modify files[/yellow]\n")
 
    confirmed = skipped = 0

    for s in suggestions:
        console.print(
            f"\n  [cyan]{s.source}[/cyan]  <-->  [cyan]{s.target}[/cyan]\n"
            f"  [dim]{s.reason}[/dim]"
        )

        src_p = G.nodes[s.source].get("path", "")
        tgt_p = G.nodes[s.target].get("path", "")

        if not src_p or not tgt_p:
            console.print("  [dim]File paths missing — skipping[/dim]")
            skipped += 1
            continue
        if not Confirm.ask(f"  Add [[{s.target}]] <-> [[{s.source}]]?", default=False):
            skipped += 1
            continue

        blank()
        r1, r2 = write_backlink_pair(Path(src_p), s.source, Path(tgt_p), s.target, dry_run=dry_run)

        if r1.success and r2.success:
            confirmed += 1

        blank()
 
    console.print(Rule(style="dim"))
    console.print(
        f"  [green]Confirmed: {confirmed}[/green]  [dim]Skipped: {skipped}[/dim]"
        + ("  [yellow](dry run)[/yellow]" if dry_run else "")
    )
    console.print()


def _print_summary(notes: list, vault_path: Path, total_chunks: int = 0, graph=None) -> None:
    vault_path = Path(vault_path)

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

    if graph is not None:
        stats_table.add_row("Graph nodes", str(graph.number_of_nodes()))
        stats_table.add_row("Graph edges", str(graph.number_of_edges()))
    
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
    hints = []
    
    if orphans > 0:
        hints.append(f"💡 {orphans} orphan note(s) — run [bold]vault graph suggest[/bold]")
    if total_chunks > 0:
        hints.append("✓ Try [bold]vault ask \"your question\"[/bold]")
    if graph is not None:
        hints.append("✓ Try [bold]vault graph map \"topic\"[/bold]")
    for h in hints:
        console.print(f"\n  [dim]{h}[/dim]")
    console.print()


if __name__ == "__main__":
    app()