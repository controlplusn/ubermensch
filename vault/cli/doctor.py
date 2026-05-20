from __future__ import annotations
 
import os
import yaml

from pathlib import Path
 
from rich.console import Console
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table
from rich import box

from vault.config.discovery import CONFIG_PATH
from vault.graph.builder import GRAPH_PATH, load_graph
from vault.eval.nli import is_nli_available, NLI_MODEL
from vault.agents.llm import OllamaBackend

console = Console()


def run_doctor() -> bool:
    console.print(Rule("[bold]Vault Doctor — system health check[/bold]", style="cyan"))
    console.print()

    checks: list[tuple[str, str, str, str]] = []

    # 1. Config
    _check_config(checks)
 
    # 2. Index (hash cache)
    _check_index(checks)
 
    # 3. Vector store
    _check_vector(checks)
 
    # 4. Graph
    _check_graph(checks)
 
    # 5. NLI model
    _check_nli(checks)
 
    # 6. Gemini API key
    _check_gemini(checks)
 
    # 7. Ollama
    _check_ollama(checks)
 
    # 8. Dependencies
    _check_deps(checks)
 
    # Render results table
    t = Table(box=box.ROUNDED, show_header=True, header_style="dim",
              padding=(0, 1), expand=True)
    t.add_column("Check",   style="dim",  width=12, no_wrap=True)
    t.add_column("Status",  width=8,  no_wrap=True)
    t.add_column("Detail",  ratio=4)
 
    all_ok   = True
    any_fail = False
 
    for name, status, detail, level in checks:
        icon = {
            "ok":   "[green]✓  pass[/green]",
            "warn": "[yellow]⚠  warn[/yellow]",
            "fail": "[red]✗  fail[/red]",
        }.get(level, "?")
 
        if level == "fail":
            all_ok   = False
            any_fail = True
 
        t.add_row(name, icon, detail)
 
    console.print(Padding(t, (0, 2)))
    console.print()
 
    if any_fail:
        console.print("  [red]Some checks failed — fix the issues above before publishing.[/red]\n")
    elif not all_ok:
        console.print("  [yellow]Some optional components are missing — core features still work.[/yellow]\n")
    else:
        console.print("  [green]All checks passed — vault is ready.[/green]\n")
 
    return all_ok


# Individual checks
def _check_config(checks: list) -> None:
 
    if not CONFIG_PATH.exists():
        checks.append(("Config", "", "No config found — run vault init", "fail"))
        return
 
    try:
        cfg        = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        vault_path = Path(cfg.get("vault_path", ""))
        if vault_path.exists():
            checks.append(("Config", "", f"Vault: {vault_path.name}", "ok"))
        else:
            checks.append(("Config", "", f"Vault path missing: {vault_path}", "fail"))
    except Exception as exc:
        checks.append(("Config", "", f"Config read error: {exc}", "fail"))

def _check_index(checks: list) -> None:
    cache = Path.home() / ".vault" / "cache" / "hashes.json"
    if not cache.exists():
        checks.append(("Index", "", "No hash cache — run vault init", "fail"))
        return
    import json
    try:
        hashes = json.loads(cache.read_text())
        checks.append(("Index", "", f"{len(hashes)} notes cached", "ok"))
    except Exception as exc:
        checks.append(("Index", "", f"Cache read error: {exc}", "warn"))

def _check_vector(checks: list) -> None:
    try:
        from vault.retrieval.store import store_stats
        stats = store_stats()
        n = stats.get("total_chunks", 0)
        if n == 0:
            checks.append(("Vectors", "", "ChromaDB empty — run vault init", "fail"))
        else:
            checks.append(("Vectors", "", f"{n} chunks in ChromaDB", "ok"))
    except Exception as exc:
        checks.append(("Vectors", "", f"ChromaDB error: {exc}", "fail"))

def _check_graph(checks: list) -> None:
    if not GRAPH_PATH.exists():
        checks.append(("Graph", "", "No graph — run vault graph build", "warn"))
        return
    G = load_graph()
    if G is None:
        checks.append(("Graph", "", "Graph file corrupt — rebuild with vault graph build", "fail"))
        return
 
    nodes  = G.number_of_nodes()
    edges  = G.number_of_edges()
    orphans = sum(1 for n, d in G.degree() if d == 0)
    coverage = round((nodes - orphans) / max(nodes, 1) * 100, 1)
    checks.append(("Graph", "", f"{nodes} nodes  {edges} edges  {coverage}% coverage", "ok"))

def _check_nli(checks: list) -> None:
    
    if is_nli_available():
        checks.append(("NLI scorer", "", f"Available: {NLI_MODEL}", "ok"))
    else:
        checks.append((
            "NLI scorer", "",
            "Not available — using keyword fallback\n"
            "    Install: pip install sentence-transformers",
            "warn",
        ))
 
def _check_gemini(checks: list) -> None:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        # Check config file
        if CONFIG_PATH.exists():
            cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
            key = cfg.get("gemini_api_key", "")
 
    if key:
        checks.append(("Gemini key", "", f"Set  (****{key[-4:]})", "ok"))
    else:
        checks.append((
            "Gemini key", "",
            "GEMINI_API_KEY not set\n"
            "    Get free key: https://aistudio.google.com/app/apikey",
            "warn",
        ))

def _check_ollama(checks: list) -> None:
    try:
        b = OllamaBackend()
        if b.is_available():
            models = b.list_models()
            checks.append(("Ollama", "", f"Running  |  models: {', '.join(models) or 'none pulled'}", "ok"))
        else:
            checks.append(("Ollama", "", "Not running (optional — needed for local LLM)", "warn"))
    except Exception:
        checks.append(("Ollama", "", "Not available (optional)", "warn"))

def _check_deps(checks: list) -> None:
    required = [
        ("typer", "typer"),
        ("rich", "rich"),
        ("yaml", "PyYAML"),
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence-transformers"),
        ("networkx", "networkx"),
        ("dotenv", "python-dotenv"),
    ]
    optional = [
        ("google.genai", "google-genai"),
        ("requests", "requests"),
    ]

    missing_required = []
    missing_optional = []
 
    for module, pkg in required:
        try:
            __import__(module)
        except ImportError:
            missing_required.append(pkg)
 
    for module, pkg in optional:
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(pkg)
 
    if missing_required:
        checks.append(("Deps", "", f"Missing required: {', '.join(missing_required)}", "fail"))
    elif missing_optional:
        checks.append(("Deps", "", f"Optional missing: {', '.join(missing_optional)}", "warn"))
    else:
        checks.append(("Deps", "", "All dependencies installed", "ok"))