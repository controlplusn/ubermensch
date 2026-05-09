from __future__ import annotations
 
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
 
import yaml
 
from vault.cli.logger import blank, done, fail, log, section, step, warn


# Constants
HASH_CACHE_PATH = Path.home() / ".vault" / "cache" / "hashes.json"
 
FRONTMATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
WIKILINK_RE    = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]+)?\]\]")
TAG_RE         = re.compile(r"(?<!\S)#([a-zA-Z][a-zA-Z0-9_/\-]*)")


# Data model
@dataclass
class VaultNote:
    """
    path: absolute path to the .md file
    title: from frontmatter 'title:' key, else filename stem
    content: body text with frontmatter stripped
    wikilinks: list of link targets
    tags: union of body #tags and frontmatter tags list
    backlinks: titles of other notes that link TO this note
    file_hash: MD5 of raw file bytes (used for change detection)
    """

    path: Path
    title: str
    content: str
    frontmatter: dict = field(default_factory=dict)
    wikilinks: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    backlinks: list[str] = field(default_factory=list)
    file_hash: str = ""


# Public API
def ingest_vault(vault_path: Path) -> list[VaultNote]:
    section("Ingestion")
    step("SCAN", f"Scanning vault: [bold]{vault_path}[/bold]")
    log("Strategy: recursive glob — finds notes in all nested subfolders")
    log("Idempotency: MD5 hash per file; unchanged notes are skipped")
 
    md_files = sorted(vault_path.rglob("*.md"))
 
    if not md_files:
        warn("SCAN", "No .md files found in this vault")
        log("Make sure you pointed vault at the right directory")
        return []
 
    log(f"Found {len(md_files)} markdown file(s) across all subfolders")
    done("SCAN", f"{len(md_files)} files to process")
 
    # Load hash cache for idempotency
    step("CACHE", "Loading hash cache")
    log(f"Cache file: {HASH_CACHE_PATH}")
    hash_cache = _load_hash_cache()
    if hash_cache:
        log(f"Cache has {len(hash_cache)} stored hashes from previous run")
    else:
        log("No cache found — this is a fresh index run")
    done("CACHE", f"{len(hash_cache)} cached entries loaded")
 
    # Parse files
    step("PARSE", "Parsing markdown files")
    log("Extracting: frontmatter · wikilinks · tags · content")
 
    notes: list[VaultNote] = []
    skipped = 0
    parsed  = 0
    errors  = 0
    new_hashes: dict[str, str] = {}
 
    for md_file in md_files:
        rel = md_file.relative_to(vault_path)
        key = str(rel)
 
        try:
            raw_bytes = md_file.read_bytes()
            file_hash = _md5(raw_bytes)
            new_hashes[key] = file_hash
 
            if hash_cache.get(key) == file_hash:
                # File unchanged — still need the note object for backlink pass
                note = _parse_bytes(md_file, raw_bytes)
                note.file_hash = file_hash
                notes.append(note)
                skipped += 1
                log(f"  [dim]cache hit[/dim]  {rel}")
            else:
                note = _parse_bytes(md_file, raw_bytes)
                note.file_hash = file_hash
                notes.append(note)
                parsed += 1
                log(f"  [green]parsed[/green]    {rel}  "
                    f"[dim]({len(note.wikilinks)} links · {len(note.tags)} tags)[/dim]")
 
        except Exception as exc:
            errors += 1
            warn("PARSE", f"Could not parse {rel}: {exc}")
 
    done(
        "PARSE",
        f"Parsed {parsed} new/changed · {skipped} unchanged · {errors} errors",
    )
 
    # Resolve backlinks
    step("BACKLINKS", "Resolving backlinks across vault")
    log("For each note, find every other note that links to it via [[wikilink]]")
    _resolve_backlinks(notes)
    total_bl = sum(len(n.backlinks) for n in notes)
    log(f"Total backlink edges resolved: {total_bl}")
    done("BACKLINKS", f"{total_bl} backlink relationships mapped")
 
    # Persist updated hash cache
    step("CACHE", "Updating hash cache")
    _save_hash_cache(new_hashes)
    log(f"Saved {len(new_hashes)} hashes to {HASH_CACHE_PATH}")
    done("CACHE", "Hash cache updated — next run will skip unchanged files")
 
    blank()
    return notes
 
 
def parse_single(path: Path) -> VaultNote:
    return _parse_bytes(path, path.read_bytes())


# Parsing internals
def _parse_bytes(path: Path, raw_bytes: bytes) -> VaultNote:
    raw = raw_bytes.decode("utf-8", errors="replace")
 
    frontmatter: dict = {}
    content = raw
 
    if m := FRONTMATTER_RE.match(raw):
        try:
            frontmatter = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            frontmatter = {}
        content = raw[m.end():]
 
    wikilinks = _extract_wikilinks(content)
    tags = _extract_tags(content, frontmatter)
    title = str(frontmatter.get("title", "") or path.stem)
 
    return VaultNote(
        path        = path,
        title       = title,
        content     = content.strip(),
        frontmatter = frontmatter,
        wikilinks   = wikilinks,
        tags        = tags,
    )
 
 
def _extract_wikilinks(text: str) -> list[str]:
    """
    [[Note Title|display alias]]  →  "Note Title"
    [[Note Title#heading]] →  "Note Title"
    """

    return list(dict.fromkeys(WIKILINK_RE.findall(text)))

def _extract_tags(text: str, frontmatter: dict) -> list[str]:
    inline = TAG_RE.findall(text)
    fm_raw = frontmatter.get("tags", [])
    if isinstance(fm_raw, str):
        fm_raw = [fm_raw]
    return list(dict.fromkeys(inline + [str(t) for t in fm_raw]))

def _resolve_backlinks(notes: list[VaultNote]) -> None:
    title_map: dict[str, VaultNote] = {
        n.title.strip().lower(): n for n in notes
    }
    for note in notes:
        for link in note.wikilinks:
            key = link.strip().lower()
            if key in title_map and title_map[key] is not note:
                target = title_map[key]
                if note.title not in target.backlinks:
                    target.backlinks.append(note.title)

# Hash cache
def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()
 
 
def _load_hash_cache() -> dict[str, str]:
    if HASH_CACHE_PATH.exists():
        try:
            return json.loads(HASH_CACHE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}
 
 
def _save_hash_cache(hashes: dict[str, str]) -> None:
    HASH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    HASH_CACHE_PATH.write_text(json.dumps(hashes, indent=2))