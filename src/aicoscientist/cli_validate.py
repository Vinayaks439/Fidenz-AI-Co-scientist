"""Standalone CLI for Layer 3 — In-Silico Validation.

Loads a run's approved hypothesis (``artifacts/<run_id>/official_hypothesis.json``),
runs the agentic closed-loop validation, and renders the plan, metrics, and verdict.
"""

from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import get_settings
from .models import Layer3Output, ValidationVerdict

console = Console()

_VERDICT_STYLE = {
    ValidationVerdict.SUPPORTED: "bold green",
    ValidationVerdict.PARTIALLY_SUPPORTED: "yellow",
    ValidationVerdict.REJECTED: "bold red",
    ValidationVerdict.INCONCLUSIVE: "dim",
}


def _render_screening(output: Layer3Output) -> None:
    """Campaign table + recommendation (screening-funnel mode only)."""
    sc = output.screening or {}
    rows = sc.get("rows", [])
    if not rows:
        return
    cfg = sc.get("config", {})
    tbl = Table(
        title=(
            f"Screening campaign — pool {cfg.get('pool_size', len(rows))}, "
            f"shortlist {cfg.get('shortlist_m', '?')}, top-{cfg.get('top_k', '?')} "
            f"(tier {cfg.get('compute_tier', '?')})"
        ),
        expand=True,
    )
    tbl.add_column("Inhibitor")
    tbl.add_column("Stage")
    tbl.add_column("dE_ngs", justify="right")
    tbl.add_column("dE_gs", justify="right")
    tbl.add_column("S", justify="right")
    tbl.add_column("Verdict")
    tbl.add_column("Flags")
    winner = sc.get("winner")
    for r in rows:
        flags = []
        if r.get("prior_extrapolated"):
            flags.append("extrap")
        if r.get("prior_missing"):
            flags.append("no-prior")
        if r.get("prior_source") == "ai-proposed":
            flags.append("ai")
        if r.get("calibration_flag") == "review":
            flags.append("review")
        name = r.get("inhibitor", "?")
        s_txt = "-" if r.get("S_mean") is None else (
            f"{r['S_mean']:.3f}"
            + (f"±{r['S_std']:.3f}" if r.get("S_std") is not None else "")
        )
        style = "bold green" if name == winner else None
        tbl.add_row(
            name,
            r.get("stage", ""),
            "-" if r.get("dE_ngs_mean_eV") is None else f"{r['dE_ngs_mean_eV']:.2f}",
            "-" if r.get("dE_gs_mean_eV") is None else f"{r['dE_gs_mean_eV']:.2f}",
            s_txt,
            r.get("verdict") or ("-" if r.get("status") == "ok" else "failed"),
            ",".join(flags) or "-",
            style=style,
        )
    console.print(tbl)

    rec = sc.get("recommendation") or {}
    if rec.get("winner"):
        body = (
            f"[bold green]{rec['winner']}[/bold green]  "
            f"(confidence {rec.get('confidence', 0):.2f})\n\n"
            f"{rec.get('winner_rationale', '')}"
        )
        if rec.get("runners_up"):
            body += f"\n\nRunners-up: {', '.join(rec['runners_up'])}"
        if rec.get("committed_candidate_outcome"):
            body += f"\nCommitted hypothesis: {rec['committed_candidate_outcome']}"
        if rec.get("risks"):
            body += "\nRisks:\n" + "\n".join(f"  - {r}" for r in rec["risks"][:6])
        console.print(Panel(body, title="Final recommendation", border_style="green"))


def _render(output: Layer3Output) -> None:
    result = output.result
    plan = result.plan

    _render_screening(output)

    plan_tbl = Table.grid(padding=(0, 1))
    plan_tbl.add_column(justify="right", style="dim")
    plan_tbl.add_column()
    plan_tbl.add_row("Domain", plan.domain)
    plan_tbl.add_row("Method", plan.method or "(unspecified)")
    if plan.reasoning_trace:
        plan_tbl.add_row("Reasoning", " -> ".join(plan.reasoning_trace[:4]))
    if plan.assumptions:
        plan_tbl.add_row("Assumptions", "; ".join(plan.assumptions[:3]))
    plan_tbl.add_row("Seed", str(plan.seed))
    console.print(Panel(plan_tbl, title="Validation plan (ReAct)", border_style="cyan"))

    metrics_tbl = Table(title="Quantitative metrics", expand=True)
    metrics_tbl.add_column("Metric")
    metrics_tbl.add_column("Value", justify="right")
    metrics_tbl.add_column("Threshold", justify="right")
    metrics_tbl.add_column("Pass")
    metrics_tbl.add_column("Note", overflow="fold")
    for m in result.metrics:
        passed = "" if m.passed is None else ("yes" if m.passed else "no")
        thr = "" if m.threshold is None else f"{m.threshold:g}"
        metrics_tbl.add_row(m.name, f"{m.value:.4g}", thr, passed, m.note)
    console.print(metrics_tbl)

    style = _VERDICT_STYLE.get(result.verdict, "white")
    console.print(
        Panel(
            f"[{style}]{result.verdict.value.upper()}[/{style}]  "
            f"(confidence {result.confidence:.2f})\n\n{result.narrative}",
            title="Verdict",
            border_style="magenta",
        )
    )

    if output.iterations > 1 or output.reflections:
        loop = Table(title="Closed-loop history (Reflection)", expand=True)
        loop.add_column("Iter", justify="right")
        loop.add_column("Domain")
        loop.add_column("Verdict")
        loop.add_column("Conf", justify="right")
        loop.add_column("Reflection")
        for i, r in enumerate(output.history):
            decision = output.reflections[i].decision if i < len(output.reflections) else "-"
            loop.add_row(
                str(r.plan.iteration),
                r.plan.domain,
                r.verdict.value,
                f"{r.confidence:.2f}",
                decision,
            )
        console.print(loop)


def run(args: argparse.Namespace) -> int:
    from .validation.runner import ValidationDataError, run_validation

    settings = get_settings()
    run_dir = settings.artifacts_path / args.run_id

    console.print(
        Panel.fit(
            f"[bold]AS-ALD Co-Scientist — Layer 3 In-Silico Validation[/bold]\n"
            f"run id: {args.run_id}\n"
            f"mode: {'offline (deterministic)' if args.offline else 'LLM-assisted'}\n"
            f"compute tier: {settings.compute_tier} "
            f"({'MLIP ' + settings.mlip_model if settings.compute_tier >= 1 else 'Tier-0 literature priors'})\n"
            f"max loop iterations: {settings.max_validation_iters}\n"
            + (
                f"screening funnel: pool={settings.screen_pool_size}, "
                f"shortlist={settings.screen_shortlist_m}, "
                f"top-k={settings.screen_top_k}\n"
                if settings.screening_mode.strip().lower() == "funnel"
                else "screening: single-candidate (legacy)\n"
            )
            + f"artifacts: {run_dir}",
            border_style="magenta",
        )
    )

    try:
        with console.status("[bold]Running agentic validation loop...[/bold]"):
            output = run_validation(args.run_id, offline=args.offline)
    except ValidationDataError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    console.print(f"[dim]Hypothesis:[/dim] {output.hypothesis_statement}\n")
    _render(output)

    # LLM (Gemini/OpenAI/...) summary of the surfaces + selection + energetics for
    # Layer 4; deterministic fallback offline. Never blocks the validation verdict.
    from .validation.summarizer import write_validation_summary

    summary_path = write_validation_summary(args.run_id, offline=args.offline)
    if summary_path is not None:
        console.print(f"[dim]Validation summary (for Layer 4): {summary_path}[/dim]")

    console.print(
        f"[dim]Artifacts (validation_results.json, validation_plan.json, "
        f"datasets/, simulation_logs/, updated knowledge_graph.json): {run_dir}[/dim]"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aicoscientist-validate",
        description="AI Co-Scientist — Layer 3 In-Silico Validation (standalone).",
    )
    parser.add_argument(
        "--run-id", required=True, help="Run id of a completed Layer 1-2 run."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Deterministic designer/reflection (no LLM key needed). Engines still run for real.",
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
