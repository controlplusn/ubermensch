from __future__ import annotations

import os
from collections import deque
from pathlib import Path

from rich.prompt import Confirm, Prompt
from vault.cli.logger import console

import yaml

from vault.cli.logger import blank, done, fail, log, section, step, warn

# Constants
MAX_DEPTH = 4
CONFIG_PATH = Path.home() / ".vault" / "config.yaml"

SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".cache", "Library",
    "AppData", "snap", "proc", "sys", "dev", ".npm", ".cargo",
    "site-packages", "dist-packages",
}


# Public API
def get_or_prompt_vault() -> Path:
    cfg = _load_config()

    if cfg and "vault_path" in cfg:
        vault_path = Path(cfg["vault_path"])

        section("Vault config")
        step("CONFIG", "Loading saved vault config")
        log(f"Config file: {CONFIG_PATH}")
        log(f"Vault path:  {vault_path}")

        if vault_path.exists():
            done("CONFIG", f"Active vault → {vault_path.name}")
            return vault_path
        else:
            warn("CONFIG", "Saved path no longer exists — running setup again")

    return _run_setup()


def find_obsidian_vaults(start: Path | None = None) -> list[Path]:
    start = start or Path.home()
    candidates: list[tuple[int, Path]] = []   # (depth, path)
    queue: deque[tuple[Path, int]] = deque([(start, 0)])
    visited: set[Path] = set()
    scanned = 0

    step("DISCOVER", f"BFS scan from [bold]{start}[/bold]")
    log("Strategy: breadth-first search (finds shallowest vault first)")
    log(f"Max depth: {MAX_DEPTH}  |  Skipping: {', '.join(sorted(SKIP_DIRS)[:5])}…")

    while queue:
        current, depth = queue.popleft()

        if current in visited or depth > MAX_DEPTH:
            continue
        visited.add(current)

        try:
            entries = list(current.iterdir())
        except PermissionError:
            log(f"  [dim]skip (no permission): {current}[/dim]")
            continue

        scanned += 1
        subdirs = [e for e in entries if e.is_dir()]
        subdir_names = {e.name for e in subdirs}

        if ".obsidian" in subdir_names:
            log(f"  [green]found[/green] {current}  (depth {depth})")
            candidates.append((depth, current))
            # Don't descend further — nested vaults are unusual
            continue

        for subdir in subdirs:
            if subdir.name not in SKIP_DIRS and not subdir.name.startswith("."):
                queue.append((subdir, depth + 1))

    log(f"Scanned {scanned} directories")
    done("DISCOVER", f"Found {len(candidates)} Obsidian vault(s)")

    return [path for _, path in sorted(candidates, key=lambda x: x[0])]


def save_config(vault_path: Path) -> None:
    """Persist the chosen vault path to ~/.vault/config.yaml."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg = _load_config() or {}
    cfg["vault_path"] = str(vault_path)
    CONFIG_PATH.write_text(yaml.dump(cfg))


# Interactive Setup

def _run_setup() -> Path:
    section("First-time setup")


    console.print("\n[bold]Welcome to Vault![/bold]  Let's find your Obsidian vault.\n")

    # Step 1 — auto or manual?
    auto = Confirm.ask(
        "  Auto-detect Obsidian vault in your home directory?",
        default=True,
    )
    blank()

    if auto:
        candidates = find_obsidian_vaults()

        if not candidates:
            warn("DISCOVER", "No Obsidian vaults found automatically")
            log("Tip: make sure your vault has been opened in Obsidian at least once")
            log("     (Obsidian creates a .obsidian/ folder on first open)")
            auto = False   # fall through to manual

    vault_path: Path

    if auto and candidates:
        if len(candidates) == 1:
            vault_path = candidates[0]
            step("SELECT", f"Only one vault found → using [bold]{vault_path.name}[/bold]")
            log(f"Path: {vault_path}")
            done("SELECT", "Vault selected automatically")
        else:
            console.print("\n  [bold]Multiple vaults found:[/bold]\n")
            for i, p in enumerate(candidates, 1):
                console.print(f"    [cyan]{i}[/cyan].  {p.name}   [dim]{p}[/dim]")
            console.print()
            choice = Prompt.ask(
                "  Which vault",
                choices=[str(i) for i in range(1, len(candidates) + 1)],
                default="1",
            )
            vault_path = candidates[int(choice) - 1]
            step("SELECT", f"User selected vault [bold]{vault_path.name}[/bold]")
            log(f"Path: {vault_path}")
            done("SELECT", "Vault selected")
    else:
        # Manual path entry
        raw = Prompt.ask("\n  Paste the full path to your Obsidian vault")
        vault_path = Path(raw.strip()).expanduser().resolve()
        step("VALIDATE", f"Validating manual path: {vault_path}")

        if not vault_path.exists():
            fail("VALIDATE", f"Path does not exist: {vault_path}")
            raise SystemExit(1)
        if not vault_path.is_dir():
            fail("VALIDATE", "Path is a file, not a directory")
            raise SystemExit(1)

        obsidian_marker = vault_path / ".obsidian"

        if not obsidian_marker.exists():
            warn("VALIDATE", "No .obsidian/ folder found — this may not be an Obsidian vault")
            log("Proceeding anyway. If indexing fails, re-run and choose a different path.")
        else:
            log(".obsidian/ marker found ✓")
        done("VALIDATE", "Path accepted")

    # Persist choice
    step("SAVE", f"Saving config to {CONFIG_PATH}")
    save_config(vault_path)
    log("Next launch will skip this setup step")
    done("SAVE", "Config saved")

    blank()
    return vault_path


# Helpers

def _load_config() -> dict | None:
    if CONFIG_PATH.exists():
        try:
            return yaml.safe_load(CONFIG_PATH.read_text()) or {}
        except yaml.YAMLError:
            return None
    return None