"""
Loop Architecture:
- REPL (Read-Eval-Print Loop)
    1. Read - reads user input from terminal
    2. Eval - dispatch to the right handler based on input
    3. Print - display the result
    4. Loop - go back to step 1


Commands:
- Inputs starting with "/" -> slash command handler
- Inputs starting with "?" -> treated as a question (same as the plain text)
- Plain text - routed to RAG ask pipeline
- Empty input - ignored, loop continued

Note: 
- this is the only stateful part of the system as the loop maintains 
    a SessionState object that tracks different parts of the system:
        - LLM backend
        - verbose mode toggle 
        - history (/history)
        - current vault path
"""


from __future__ import annotations
 
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
 
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

from vault.cli.display import _print_ask_result, _print_map, _print_suggestions

from vault.cli.logger import blank, done, fail, log, section, step, warn, set_verbose

from vault.retrieval.store import store_stats
from vault.retrieval.embedder import embed_query
from vault.retrieval.store import retrieve

from vault.graph.builder import load_graph, build_graph
from vault.graph.mapper import map_topic
from vault.graph.suggester import find_suggestions

from vault.agents.ask import run_ask
from vault.agents.planner import run_plan
from vault.agents.llm import get_backend, OllamaBackend
from vault.agents.writer import write_synthesis

from vault.eval.faithfulness import FaithfulnessEvaluator

 
console = Console()



# Session State
@dataclass
class SessionState:
    vault_path: Path
    llm_backend: str = "gemini" # "gemini" or "ollama"
    llm_model: str = ""         # empty = use backend default
    verbose: bool = False
    show_eval: bool = True
    api_key: str = ""
    history: list[str] = field(default_factory=list)
    total_asks: int   = 0



# Main loop
def run_loop(
    vault_path: Path,
    llm_backend: str = "gemini",
    llm_model: str = "",
    api_key: str = "",
    auto: bool = False,
    show_eval: bool = True,
) -> None:
    
    state = SessionState(
        vault_path=vault_path,
        llm_backend=llm_backend,
        llm_model=llm_model,
        api_key=api_key or os.environ.get("GEMINI_API_KEY", ""),
        show_eval=show_eval,
    )

    stats = store_stats()
    G = load_graph()
    n_chunks = stats.get("total_chunks", 0)
    n_edges = G.number_of_edges() if G else 0

    _print_banner(vault_path, llm_backend, n_chunks, n_edges)

    # Main REPL
    while True:
        try:
            raw = console.input("\n[bold cyan]>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            _exit_gracefully(state)
            break

        if not raw:
            continue

        if raw.lower() in ("/exit", "/quit", "exit", "quit"):
            _exit_gracefully(state)
            break

        # Dispatch
        try:
            _dispatch(raw, state, auto=auto)
        except KeyboardInterrupt:
            console.print("\n  [dim]Interrupted — type /exit to quit[/dim]")
        except Exception as exc:
            console.print(f"\n  [red]Error: {exc}[/red]")
            if state.verbose:
                import traceback
                console.print(traceback.format_exc())



# Command dispatcher
def _dispatch(raw: str, state: SessionState, auto: bool = False) -> None:
    # Goal: Route input to the correct handler

    if raw.startswith("/"):
        parts = raw[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "help":
            _cmd_help()
        elif cmd == "history":
            _cmd_history(state)
        elif cmd == "clear":
            console.clear()
        elif cmd == "verbose":
            _cmd_verbose(arg, state)
        elif cmd == "eval":
            _cmd_eval(arg, state)
        elif cmd == "llm":
            _cmd_llm(arg, state)
        elif cmd == "plan":
            if not arg:
                console.print("  [dim]Usage: /plan <complex question>[/dim]")
            else:
                _cmd_plan(arg, state)
        elif cmd == "synthesize":
            if not arg:
                console.print("  [dim]Usage: /synthesize <topic>[/dim]")
            else:
                _cmd_synthesize(arg, state, auto=auto)
        elif cmd in ("graph",):
            _cmd_graph(arg, state)
        elif cmd in ("exit", "quit"):
            raise SystemExit(0)
        else:
            console.print(f"  [dim]Unknown command: /{cmd}  — type /help[/dim]")

    # Plain question -> RAG ask
    else:
        _cmd_ask(raw, state)



# Command handlers
def _cmd_ask(question: str, state: SessionState) -> None:
    state.history.append(question)
    state.total_asks += 1

    set_verbose(state.verbose)

    result = run_ask(
        query=question,
        top_k=5,
        api_key=state.api_key,
        show_eval=state.show_eval,
        eval_verbose=False,
        backend_name=state.llm_backend,
        model=state.llm_model,
    )

    blank()
    _print_ask_result(result, show_eval=state.show_eval)


def _cmd_plan(question: str, state: SessionState) -> None:
    # Multi-step planner for complex questions
    set_verbose(state.verbose)
 
    state.history.append(f"/plan {question}")
 
    backend   = get_backend(state.llm_backend, state.api_key, state.llm_model)
    evaluator = FaithfulnessEvaluator()
 
    def retriever_fn(q: str, top_k: int):
        vec = embed_query(q)
        return retrieve(vec, top_k=top_k)
 
    result = run_plan(
        query=question,
        llm_fn=backend.raw,
        retriever_fn=retriever_fn,
        evaluator=evaluator,
        top_k=5,
        verbose=state.verbose,
    )
 
    _print_plan_result(result)


def _cmd_synthesize(topic: str, state: SessionState, auto: bool = False) -> None:
    set_verbose(state.verbose)
    state.history.append(f"/synthesize {topic}")

    section("Synthesis")
    step("SYNTH", f"Synthesizing notes on: [bold]{topic}[/bold]")
    log("Retrieving all relevant notes on this topic")
    log("LLM will generate a unified summary across them")
    log("Result will be saved as a new .md file in your vault")

    # Get broad retrieval
    result = run_ask(
        query=f"Summarize and synthesize everything about {topic}",
        top_k=8,
        api_key=state.api_key,
        show_eval=False,
        eval_verbose=False,
        backend_name=state.llm_backend,
        model=state.llm_model,
    )

    if not result.answer:
        warn("SYNTH", "No content to synthesize")
        return
    
    blank()
    console.print(Rule("[dim]Synthesis preview[/dim]", style="dim"))
    console.print(f"\n  [bold]Title:[/bold] Synthesis — {topic}\n")
    console.print(f"  [bold]Sources:[/bold] {', '.join(result.sources)}\n")
 
    # Truncated preview
    preview = result.answer[:500] + ("..." if len(result.answer) > 500 else "")
    console.print(f"  {preview}\n")

    # Confirm before writing
    if not auto:
        from rich.prompt import Confirm
        if not Confirm.ask("  Write this synthesis to your vault?", default=True):
            console.print("  [dim]Cancelled — synthesis not saved[/dim]")
            return
        
    blank()
    out_path = write_synthesis(
        vault_path=state.vault_path,
        title=f"Synthesis — {topic}",
        content=result.answer,
        source_titles=result.sources,
        tags=[topic.lower().replace(" ", "-")],
        dry_run=False,
    )

    if out_path:
        console.print(f"\n  [green]✓ Saved:[/green] {out_path.name}")
        console.print("  [dim]Open in Obsidian to see the new synthesis note[/dim]\n")


def _cmd_graph(arg: str, state: SessionState) -> None:
    set_verbose(state.verbose)
 
    parts   = arg.split(maxsplit=1)
    subcmd  = parts[0].lower() if parts else ""
    subarg  = parts[1] if len(parts) > 1 else ""

    if subcmd == "map":
        if not subarg:
            console.print("  [dim]Usage: /graph map <topic>[/dim]")
            return
        G = load_graph()
        if G is None:
            console.print("  [yellow]Run vault graph build first.[/yellow]")
            return
        nodes = map_topic(G, subarg, max_depth=2)
        blank()
        if nodes:
            _print_map(nodes, subarg)
        else:
            console.print(f"  [dim]No notes found matching '{subarg}'[/dim]")
 
    elif subcmd == "suggest":
        G = load_graph()
        if G is None:
            console.print("  [yellow]Run vault graph build first.[/yellow]")
            return
        suggestions = find_suggestions(G, top_n=10)
        blank()
        if suggestions:
            _print_suggestions(suggestions)
        else:
            console.print("  [dim]No suggestions found[/dim]")
 
    elif subcmd == "build":
        from vault.core.ingester import ingest_vault
        from vault.retrieval.store import get_collection
        step("GRAPH", "Rebuilding knowledge graph")
        notes = ingest_vault(state.vault_path)
        try:
            collection = get_collection()
            results = collection.get(include=["embeddings", "metadatas"])
            note_emb_acc: dict = {}
            for emb, meta in zip(results["embeddings"], results["metadatas"]):
                title = meta.get("note_title", "")
                if title:
                    note_emb_acc.setdefault(title, []).append(emb)
            note_embeddings = {}
            for title, emb_list in note_emb_acc.items():
                n, dim = len(emb_list), len(emb_list[0])
                note_embeddings[title] = [sum(e[i] for e in emb_list) / n for i in range(dim)]
        except Exception:
            note_embeddings = None
        build_graph(notes, embeddings=note_embeddings)
 
    else:
        console.print("  [dim]Graph commands: /graph map <topic>  /graph suggest  /graph build[/dim]")


def _cmd_llm(arg: str, state: SessionState) -> None:
    parts = arg.lower().split()
    if not parts:
        _show_llm_status(state)
        return
 
    cmd = parts[0]
 
    if cmd == "status":
        _show_llm_status(state)
 
    elif cmd in ("gemini", "google"):
        state.llm_backend = "gemini"
        state.llm_model   = parts[1] if len(parts) > 1 else ""
        console.print(f"  [green]Switched to Gemini[/green]  model: {state.llm_model or 'gemini-2.5-flash-lite'}")
 
    elif cmd == "ollama":
        model = parts[1] if len(parts) > 1 else ""
        state.llm_backend = "ollama"
        state.llm_model   = model
 
        # Check if Ollama is available
        backend = OllamaBackend(model=model or None)
        if backend.is_available():
            models = backend.list_models()
            console.print(f"  [green]Switched to Ollama[/green]  model: {state.llm_model or 'llama3'}")
            if models:
                console.print(f"  [dim]Available: {', '.join(models)}[/dim]")
        else:
            warn("LLM", "Ollama not running — start with: ollama serve")
            console.print("  [dim]Install from: https://ollama.ai[/dim]")
 
    else:
        console.print("  [dim]Usage: /llm status | /llm gemini | /llm ollama [model][/dim]")


def _cmd_verbose(arg: str, state: SessionState) -> None:
    if arg.lower() in ("on", "true", "1", "yes"):
        state.verbose = True
        set_verbose(True)
        console.print("  [green]Verbose mode ON[/green] — internal logs will be shown")
    elif arg.lower() in ("off", "false", "0", "no"):
        state.verbose = False
        set_verbose(False)
        console.print("  [dim]Verbose mode OFF — internal logs hidden[/dim]")
    else:
        status = "[green]ON[/green]" if state.verbose else "[dim]OFF[/dim]"
        console.print(f"  Verbose: {status}   Usage: /verbose on | /verbose off")


def _cmd_eval(arg: str, state: SessionState) -> None:
    if arg.lower() in ("on", "true", "1", "yes"):
        state.show_eval = True
        console.print("  [green]Faithfulness scoring ON[/green]")
    elif arg.lower() in ("off", "false", "0", "no"):
        state.show_eval = False
        console.print("  [dim]Faithfulness scoring OFF[/dim]")
    else:
        status = "[green]ON[/green]" if state.show_eval else "[dim]OFF[/dim]"
        console.print(f"  Eval: {status}   Usage: /eval on | /eval off")


def _cmd_history(state: SessionState) -> None:
    if not state.history:
        console.print("  [dim]No questions asked this session[/dim]")
        return
    console.print(Rule("[dim]Session history[/dim]", style="dim"))
    for i, q in enumerate(state.history, 1):
        console.print(f"  [dim]{i:>2}.[/dim]  {q}")
    console.print()


def _cmd_help() -> None:
    console.print(Rule("[dim]Vault Agent — commands[/dim]", style="dim"))
 
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column(style="cyan",  no_wrap=True)
    t.add_column(style="dim")

    rows = [
        ("Any question", "RAG query with retrieval + faithfulness eval"),
        ("/plan <question>", "Multi-step planner for complex questions"),
        ("/synthesize <topic>", "Generate + save cross-note synthesis to vault"),
        ("", ""),
        ("/graph map <topic>",  "BFS idea cluster around a topic"),
        ("/graph suggest", "Show backlink suggestions"),
        ("/graph build", "Rebuild knowledge graph"),
        ("", ""),
        ("/llm status", "Show current LLM backend"),
        ("/llm gemini", "Switch to Gemini Flash (free API)"),
        ("/llm ollama", "Switch to local Ollama"),
        ("/llm ollama llama3", "Switch to Ollama with specific model"),
        ("", ""),
        ("/verbose on|off", "Show or hide internal process logs"),
        ("/eval on|off", "Show or hide faithfulness scoring"),
        ("/history", "Show questions asked this session"),
        ("/clear", "Clear the screen"),
        ("/help", "Show this help"),
        ("/exit", "Quit the agent"),
    ]
 
    for cmd, desc in rows:
        t.add_row(cmd, desc)
 
    console.print(t)
    console.print()



# Display helpers
def _print_banner(vault_path: Path, llm_backend: str, n_chunks: int, n_edges: int) -> None:
    backend_label = (
        "[green]Gemini Flash[/green]" if llm_backend == "gemini"
        else "[blue]Ollama (local)[/blue]"
    )
    content = (
        f"  [bold]Vault[/bold]    {vault_path.name}  [dim]{vault_path}[/dim]\n"
        f"  [bold]LLM[/bold]      {backend_label}\n"
        f"  [bold]Index[/bold]    {n_chunks} chunks  ·  {n_edges} graph edges\n\n"
        f"  Type a question or [cyan]/help[/cyan] for commands  ·  [cyan]/exit[/cyan] to quit"
    )
    console.print()
    console.print(Panel(
        content,
        title="[bold]Vault Agent[/bold]",
        border_style="cyan",
        padding=(0, 1),
    ))
    console.print()


def _print_plan_result(result) -> None:
    console.print(Rule("[dim]Plan — sub-questions answered[/dim]", style="dim"))
 
    for i, sa in enumerate(result.sub_answers, 1):
        console.print(f"\n  [dim]{i}. {sa.question}[/dim]")
        if sa.sources:
            console.print(f"  [dim]   Sources: {', '.join(sa.sources)}[/dim]")
        console.print(f"  {sa.answer[:300]}{'...' if len(sa.answer) > 300 else ''}")
 
    console.print(Rule("[dim]Final synthesized answer[/dim]", style="dim"))
    console.print(f"\n{result.final_answer}\n")
 
    if result.all_sources:
        console.print(f"  [dim]All sources: {', '.join(result.all_sources)}[/dim]")
 
    color = "green" if result.faithfulness >= 0.7 else "yellow" if result.faithfulness >= 0.4 else "red"
    console.print(f"  [{color}]Faithfulness: {result.faithfulness:.1%}[/{color}]\n")


def _show_llm_status(state: SessionState) -> None:
    console.print(Rule("[dim]LLM status[/dim]", style="dim"))
    console.print(f"  Backend: [bold]{state.llm_backend}[/bold]")
    if state.llm_model:
        console.print(f"  Model:   [bold]{state.llm_model}[/bold]")
    if state.llm_backend == "ollama":
        from vault.agents.llm import OllamaBackend
        b = OllamaBackend()
        available = b.is_available()
        status = "[green]running[/green]" if available else "[red]not running[/red]"
        console.print(f"  Ollama:  {status}")
        if available:
            models = b.list_models()
            console.print(f"  Models:  {', '.join(models) or 'none pulled'}")
    console.print()



# Graceful exit
def _exit_gracefully(state: SessionState) -> None:
    console.print()
    console.print(Panel(
        f"  Session ended  ·  {state.total_asks} question(s) asked\n"
        f"  [dim]Your vault is unchanged unless you used /synthesize or /graph suggest --write[/dim]",
        border_style="dim",
        padding=(0, 1),
    ))
    console.print()