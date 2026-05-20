from __future__ import annotations

import typer
import os
import yaml
import subprocess
import sys

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

from vault.cli.logger import blank, log, section, step, done, warn, set_verbose

from vault.config.discovery import get_or_prompt_vault, save_config, CONFIG_PATH

from vault.core.ingester import ingest_vault

from vault.retrieval.chunker import chunk_notes
from vault.retrieval.embedder import embed_chunks
from vault.retrieval.store import upsert_chunks, store_stats, get_collection

from vault.agents.ask import run_ask
from vault.agents.writer import write_backlink_pair
from vault.agents.loop import run_loop

from rich.prompt import Confirm

from vault.graph.builder import build_graph, load_graph
from vault.graph.suggester import find_suggestions
from vault.graph.mapper import map_topic

from vault.cli.doctor import run_doctor



app = typer.Typer(
    name="vault",
    help="Local-first agentic knowledge system for Obsidian vaults.",
    add_completion=False,
    rich_markup_mode="rich",
)
graph_app = typer.Typer(help="Knowledge graph operations.")
agent_app = typer.Typer(help="Interactive agent loop.")
app.add_typer(graph_app, name="graph")
app.add_typer(agent_app, name="agent")

console = Console()


# Global verbose flag

@app.callback()
def main_callback(
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show internal process logs (default: hidden)",
    ),
) -> None:
    # Local-first agentic knowledge system for Obsidian vaults
    set_verbose(verbose)



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
        warn("CACHE", "--force flag set: clearing hash cache for full re-index")
        cache = Path.home() / ".vault" / "cache" / "hashes.json"

        if cache.exists():
            cache.unlink()
            log("Hash cache cleared")


    # 1: Ingest
    notes = ingest_vault(vault_path)
    if not notes:
        console.print("[yellow]No notes found. Check your vault path.[/yellow]")
        raise typer.Exit(0)
    blank()

    # 2: Chunk
    chunks = chunk_notes(notes)
    if not chunks:
        console.print("[yellow]No chunks produced. Notes may be empty.[/yellow]")
        raise typer.Exit(0)
    blank()
 
    # 3: Embed
    texts = [c.text for c in chunks]
    embeddings = embed_chunks(texts)
    blank()

    # 4: Store
    upsert_chunks(chunks, embeddings)
    blank()

    # 5. Build graph
    try:
        collection = get_collection()
        results = collection.get(include=["embeddings", "metadatas"])
        note_emb_acc: dict = {}

        for emb, meta in zip(results["embeddings"], results["metadatas"]):
            title = meta.get("note_title", "")

            if title:
                note_emb_acc.setdefault(title, []).append(emb)

        note_embeddings = {
            t: [sum(e[i] for e in el) / len(el) for i in range(len(el[0]))]
            for t, el in note_emb_acc.items()
        }
    except Exception:
        note_embeddings = None
 
    build_graph(notes, embeddings=note_embeddings)
    blank()

    # Summary
    _print_summary(notes, vault_path, total_chunks=len(chunks))



# Vault ask

@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    eval_mode: bool = typer.Option(False, "--eval", help="Per-claim breakdown"),
    no_eval: bool = typer.Option(False, "--no-eval", help="Skip faithfulness scoring"),
    key: Optional[str] = typer.Option(None, "--key", help="Gemini API key"),
    llm: str = typer.Option("gemini", "--llm", help="gemini or ollama"),
    model: str = typer.Option("", "--model", help="Model override"),
) -> None:
 
    result = run_ask(
        query=question,
        top_k=top_k,
        api_key=key or os.environ.get("GEMINI_API_KEY", ""),
        show_eval=not no_eval,
        eval_verbose=eval_mode,
        backend_name=llm,
        model=model,
    )
    blank()
    _print_ask_result(result, show_eval=not no_eval, eval_verbose=eval_mode)


# Vault agent run

@agent_app.command("run")
def agent_run(
    llm: str = typer.Option("gemini", "--llm",   help="gemini or ollama"),
    model: str = typer.Option("", "--model", help="Model name override"),
    key: Optional[str] = typer.Option(None, "--key",   help="Gemini API key"),
    auto: bool = typer.Option(False, "--auto",  help="Skip confirmations"),
    no_eval: bool = typer.Option(False, "--no-eval", help="Disable faithfulness scoring"),
) -> None:

    vault_path = Path(get_or_prompt_vault())
    api_key    = key or os.environ.get("GEMINI_API_KEY", "")
 
    run_loop(
        vault_path=vault_path,
        llm_backend=llm,
        llm_model=model,
        api_key=api_key or os.environ.get("GEMINI_API_KEY", ""),
        auto=auto,
        show_eval=not no_eval,
    )



# Vault doctor

@app.command()
def doctor() -> None:
    run_doctor()


# Vault publish
@app.command()
def publish(
    dry_run: bool = typer.Option(False, "--dry-run", help="Check without publishing"),
    test: bool = typer.Option(False, "--test",    help="Publish to TestPyPI"),
) -> None:
    console.print(Rule("[bold]Vault publish[/bold]", style="cyan"))
    console.print()

    console.print("  [dim]Running health checks before publish...[/dim]\n")
    ok = run_doctor()

    if not ok:
        console.print("[red]Fix failing checks before publishing.[/red]")
        raise typer.Exit(1)
 
    if dry_run:
        console.print("  [yellow]DRY RUN — build and check only, not uploading[/yellow]\n")
 
    # Check build tools
    console.print("  Checking build tools...")
    for pkg in ("build", "twine"):
        try:
            __import__(pkg)
        except ImportError:
            console.print(f"  [red]Missing: {pkg}[/red]  Run: pip install {pkg}")
            raise typer.Exit(1)
        
    # Build
    console.print("\n  [bold]Building package...[/bold]")
    result = subprocess.run(
        [sys.executable, "-m", "build"],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        console.print(f"  [red]Build failed:[/red]\n{result.stderr}")
        raise typer.Exit(1)
    console.print("  [green]Build successful[/green]")

    if dry_run:
        # Validate with twine check
        result2 = subprocess.run(
            [sys.executable, "-m", "twine", "check", "dist/*"],
            capture_output=True, text=True,
        )
        console.print(result2.stdout)
        console.print("  [green]Dry run complete — package is valid[/green]")
        return
    
    # Upload
    repo_flag = ["--repository", "testpypi"] if test else []
    dest = "TestPyPI" if test else "PyPI"
    console.print(f"\n  [bold]Uploading to {dest}...[/bold]")
 
    result3 = subprocess.run(
        [sys.executable, "-m", "twine", "upload"] + repo_flag + ["dist/*"],
    )
    if result3.returncode == 0:
        pkg_name = "vault-kb"
        if test:
            console.print(f"\n  [green]Published to TestPyPI[/green]")
            console.print(f"  Install: pip install -i https://test.pypi.org/simple/ {pkg_name}")
        else:
            console.print(f"\n  [green]Published to PyPI[/green]")
            console.print(f"  Install: pip install {pkg_name}")



# Vault graph build

@graph_app.command("build")
def graph_build(
    path: Optional[Path] = typer.Option(None, "--path", "-p"),
    no_semantic: bool = typer.Option(False, "--no-semantic"),
) -> None:
    vault_path = path.expanduser().resolve() if path else get_or_prompt_vault()
    notes = ingest_vault(vault_path)

    if not notes:
        console.print("[yellow]No notes found.[/yellow]")
        raise typer.Exit(0)
    
    blank()
 
    note_embeddings = None

    if not no_semantic:
        step("EMBEDDINGS", "Loading embeddings for semantic edges")

        try:
            collection = get_collection()
            results = collection.get(include=["embeddings", "metadatas"])
            acc: dict = {}

            for emb, meta in zip(results["embeddings"], results["metadatas"]):
                t = meta.get("note_title", "")

                if t:
                    acc.setdefault(t, []).append(emb)
                    
            note_embeddings = {
                t: [sum(e[i] for e in el) / len(el) for i in range(len(el[0]))]
                for t, el in acc.items()
            }

            done("EMBEDDINGS", f"Ready for {len(note_embeddings)} notes")
            blank()
        except Exception as exc:
            warn("EMBEDDINGS", f"Could not load embeddings: {exc}")
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

        cfg = yaml.safe_load(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
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

def _print_ask_result(result, show_eval=True, eval_verbose=False) -> None:
    # Retry notice
    if result.attempts > 1:
        console.print(
            f"  [dim] Retrieved after {result.attempts} attempt(s) "
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
            use_nli = any(getattr(cr, "scorer_used", "") == "nli" for cr in er.claims)
            
            claim_table = Table(
                box=box.ROUNDED,
                show_header=True,
                header_style="dim",
                padding=(0, 1),
                expand=True,
            )
            claim_table.add_column("#", style="dim",  width=3,  no_wrap=True)
            claim_table.add_column("Claim", style="white", ratio=5)
            
            if use_nli:
                claim_table.add_column("NLI label",  width=15, no_wrap=True)
                claim_table.add_column("Entail",     width=8,  no_wrap=True)
            else:
                claim_table.add_column("Overlap",    width=9,  no_wrap=True)
                claim_table.add_column("Verdict",    width=14, no_wrap=True)

            claim_table.add_column("Scorer", style="dim", width=9, no_wrap=True)
 
            for i, cr in enumerate(er.claims, 1):
                if use_nli:
                    nli_icon = {
                        "entailment":    "[green]✓ entailed[/green]",
                        "contradiction": "[red]✗ contradicted[/red]",
                        "neutral":       "[yellow]~ neutral[/yellow]",
                    }.get(getattr(cr, "nli_label", ""), "?")
                    claim_table.add_row(
                        str(i),
                        cr.text[:100] + ("…" if len(cr.text) > 100 else ""),
                        nli_icon,
                        f"{getattr(cr, 'nli_entailment', 0):.2f}",
                        getattr(cr, "scorer_used", ""),
                    )
                else:
                    verdict = "[green]✓ supported[/green]" if cr.supported else "[red]✗ unsupported[/red]"
                    claim_table.add_row(
                        str(i),
                        cr.text[:100] + ("…" if len(cr.text) > 100 else ""),
                        f"{cr.overlap:.0%}",
                        verdict,
                        getattr(cr, "scorer_used", "keyword"),
                    )
            console.print(Padding(claim_table, (0, 2)))
 
            if hasattr(er, "scorer_used"):
                console.print(f"\n  [dim]Scorer: {er.scorer_used}[/dim]")
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


def _print_summary(notes, vault_path, total_chunks=0, graph=None) -> None:
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
    top_linked = sorted(notes, key=lambda n: len(n.backlinks), reverse=True)[:5]
 
    header = Text()
    header.append(f"{vault_path.name}", style="bold cyan")
    header.append(f"  {vault_path}\n", style="dim")
    console.print(Panel(header, box=box.ROUNDED, border_style="cyan", padding=(0, 1)))
 
    st = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    st.add_column(style="dim", no_wrap=True)
    st.add_column(style="bold white", no_wrap=True)
    st.add_row("Notes", str(total_notes))
    st.add_row("Words", f"{total_words:,}")
    st.add_row("Chunks indexed", str(total_chunks) if total_chunks else "[dim]run vault init[/dim]")
    st.add_row("Wikilinks", str(total_links))
    st.add_row("Backlinks", str(total_bl))
    st.add_row("Total tags", str(total_tags))
    st.add_row("Orphan notes", str(orphans))
    if total_notes:
        st.add_row("Link density", f"{total_links / total_notes:.1f} links/note")
    if graph is not None:
        st.add_row("Graph nodes",  str(graph.number_of_nodes()))
        st.add_row("Graph edges",  str(graph.number_of_edges()))
    console.print(Padding(st, (0, 0, 0, 2)))
 
    if top_tags:
        console.print("\n  [dim]Top tags[/dim]")
        items = [Text.from_markup(f"  [cyan]#{tg}[/cyan] [dim]{c}[/dim]") for tg, c in top_tags]
        console.print(Padding(Columns(items, equal=False, expand=False), (0, 0, 0, 2)))
 
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
 
    hints = []
    if orphans > 0:
        hints.append(f"💡 {orphans} orphan note(s) — run [bold]vault graph suggest[/bold]")
    if total_chunks > 0:
        hints.append("✓ Try [bold]vault agent run[/bold] to start the interactive loop")
    if graph is not None:
        hints.append("✓ Try [bold]vault graph map \"topic\"[/bold]")
    for h in hints:
        console.print(f"\n  [dim]{h}[/dim]")
    console.print()


if __name__ == "__main__":
    app()