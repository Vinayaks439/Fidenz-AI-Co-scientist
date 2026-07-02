"""Standalone CLI for Layer 4 - Agentic LaTeX Paper Stitcher.

Assembles a reproducible manuscript from a completed run's validation artifacts:

    aicoscientist-paper --run-id <run_id>
"""

from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.panel import Panel

from .config import get_settings

console = Console()


def run(args: argparse.Namespace) -> int:
    from .layer4_paper import PaperDataError, stitch_paper

    settings = get_settings()
    run_dir = settings.artifacts_path / args.run_id

    console.print(
        Panel.fit(
            f"[bold]AI Co-Scientist - Layer 4 Paper Stitcher[/bold]\n"
            f"run id: {args.run_id}\nartifacts: {run_dir}",
            border_style="magenta",
        )
    )

    try:
        with console.status("[bold]Stitching manuscript from artifacts...[/bold]"):
            result = stitch_paper(args.run_id, offline=args.offline)
    except PaperDataError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    console.print(f"[green]Manuscript source:[/green] {result.tex_path}")
    for fig in result.figures:
        console.print(f"[green]Figure:[/green] {fig}")
    if result.pdf_path:
        console.print(f"[green]Compiled PDF:[/green] {result.pdf_path}")
    else:
        console.print(
            "[yellow]No LaTeX toolchain found; wrote .tex source only. "
            "Install tectonic or latexmk to build the PDF.[/yellow]"
        )
    console.print(f"[dim]Verdict carried into the manuscript: {result.verdict}[/dim]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aicoscientist-paper",
        description="AI Co-Scientist - Layer 4 agentic LaTeX manuscript stitcher.",
    )
    parser.add_argument(
        "--run-id", required=True, help="Run id of a completed Layer 1-3 run."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Deterministic section writers (no LLM key needed); figures/tables "
             "still render from the real artifacts.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return run(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
