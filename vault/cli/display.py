from __future__ import annotations

from rich.console import Console
from rich.padding import Padding
from rich.rule import Rule
from rich.table import Table
from rich import box

console = Console()


def _print_ask_result(result, show_eval=True, eval_verbose=False) -> None:
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
            claim_table.add_column("#", style="dim",  width=3,  no_wrap=True)
            claim_table.add_column("Claim", style="white", ratio=5)
            claim_table.add_column("Overlap", style="dim",  width=9,  no_wrap=True)
            claim_table.add_column("Verdict", width=14,     no_wrap=True)
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