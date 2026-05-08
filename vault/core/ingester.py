from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class VaultNote:
    path: Path
    title: str
    content: str
    frontmatter: dict = field(default_factory=dict)
    wikilinks: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    backlinks: list[str] = field(default_factory=list)


WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]+)?\]\]")
TAG_RE = re.compile(r"(?<!\S)#([a-zA-Z][a-zA-Z0-9_/-]*)")
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)

def parse_note(path: Path) -> VaultNote:
    raw = path.read_text(encoding="utf-8")
    frontmatter, content = {}, raw

    if m := FRONTMATTER_RE.match(raw):
        try:
            frontmatter = yaml.safe_load(m.group(1)) or {}
        except yaml.YAMLError:
            pass

        content = raw[m.end():]

    wikilinks = WIKILINK_RE.findall(content)
    tags = TAG_RE.findall(content)

    if fm_tags := frontmatter.get("tags", []):
        tags = list(set(tags + (fm_tags if isinstance(fm_tags, list) else [fm_tags])))

    return VaultNote(path=path, title=frontmatter.get("title", path.stem),
                     content=content.strip(), frontmatter=frontmatter,
                     wikilinks=wikilinks, tags=tags)


def ingest_vault(vault_path: Path) -> list[VaultNote]:
    notes = [parse_note(p) for p in vault_path.rglob("*.md")]
    title_map = {n.title: n for n in notes}
    for note in notes:
        for link in note.wikilinks:
            if link in title_map:
                title_map[link].backlinks.append(note.title)
    return notes