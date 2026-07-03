"""Final-recommendation agent for the Layer-3 screening funnel.

After the campaign has computed real numbers for every screened candidate, this agent
writes the closing scientific judgement: which inhibitor to recommend, which are viable
runners-up, and what risks attach to each. It is deliberately NOT allowed to override
the physics: the winner must be the top-ranked candidate by computed selectivity, and
every number it cites must come from the screening table. The LLM adds interpretation
(risk flags, committed-hypothesis outcome, deployment caveats); a deterministic
fallback produces the same schema offline so the funnel never blocks on a key.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are the final-recommendation agent of an AS-ALD (area-selective atomic layer "
    "deposition) co-scientist. A screening campaign has ALREADY computed selectivity "
    "for every candidate inhibitor; your job is judgement, not arithmetic.\n\n"
    "HARD RULES:\n"
    "1. The recommended winner MUST be the candidate ranked first by computed "
    "S_at_target_mean (ties broken by differential_blocking). You may not promote a "
    "lower-ranked candidate for any reason; if you believe the top candidate is risky, "
    "keep it as winner and state the risk.\n"
    "2. Every number you cite must appear verbatim in the screening table you are "
    "given. Never invent, recall, or estimate energetics.\n"
    "3. You MUST flag, per candidate you discuss: extrapolated or missing priors, "
    "calibration validity_flag='review', ai-proposed provenance, failed fidelity "
    "gates, and verdicts below 'supported'.\n"
    "4. State explicitly whether the committed hypothesis inhibitor won, lost (and to "
    "whom), or was absent from the pool.\n"
    "5. Keep the rationale grounded in the mechanism: differential blocking "
    "(chemisorb on NGS, physisorb on GS), volatility/removability, site-matching."
)


class RecommendationReport(BaseModel):
    """The funnel's final answer, persisted as recommendation.json."""

    winner: str = ""
    winner_rationale: str = ""
    runners_up: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    committed_candidate_outcome: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


def _fmt_row(r: dict) -> str:
    flags = []
    if r.get("prior_extrapolated"):
        flags.append("extrapolated-prior")
    if r.get("prior_missing"):
        flags.append("missing-prior")
    if r.get("prior_source") == "ai-proposed":
        flags.append("ai-proposed")
    if r.get("calibration_flag") == "review":
        flags.append("calibration-review")
    return (
        f"- {r['inhibitor']}: S={r.get('S_mean', 'n/a')}+/-{r.get('S_std', 'n/a')}, "
        f"diff_blocking={r.get('differential_blocking', 'n/a')}, "
        f"dE_ngs={r.get('dE_ngs_mean_eV', 'n/a')} eV, "
        f"dE_gs={r.get('dE_gs_mean_eV', 'n/a')} eV, "
        f"verdict={r.get('verdict', 'n/a')}, stage={r.get('stage', 'n/a')}"
        + (f", flags=[{', '.join(flags)}]" if flags else "")
    )


def _deterministic(rows: list[dict], committed: str) -> RecommendationReport:
    """Offline / fallback recommendation: argmax computed S, honest flags."""
    scored = [r for r in rows if r.get("S_mean") is not None]
    scored.sort(key=lambda r: (r.get("S_mean", 0.0),
                               r.get("differential_blocking", 0.0)), reverse=True)
    if not scored:
        return RecommendationReport(
            winner="", winner_rationale="No candidate produced a computed selectivity.",
            committed_candidate_outcome="not evaluated", confidence=0.0,
        )
    win = scored[0]
    risks = []
    for r in scored[:5]:
        if r.get("prior_extrapolated"):
            risks.append(f"{r['inhibitor']}: NGS prior extrapolated from another surface")
        if r.get("prior_missing"):
            risks.append(f"{r['inhibitor']}: no literature energetics prior")
        if r.get("prior_source") == "ai-proposed":
            risks.append(f"{r['inhibitor']}: AI-proposed novel compound (needs Tier-1+)")
        if r.get("calibration_flag") == "review":
            risks.append(f"{r['inhibitor']}: MLIP-vs-literature calibration flagged")
    committed_lc = (committed or "").strip().lower()
    if not committed_lc:
        committed_note = "no committed inhibitor in the hypothesis"
    elif win["inhibitor"].lower() == committed_lc:
        committed_note = f"committed inhibitor '{committed}' won the screen"
    elif any(r["inhibitor"].lower() == committed_lc for r in rows):
        committed_note = (
            f"committed inhibitor '{committed}' was screened but lost to "
            f"'{win['inhibitor']}'"
        )
    else:
        committed_note = f"committed inhibitor '{committed}' was not in the pool"
    s_std = win.get("S_std") or 0.0
    conf = max(0.1, min(0.95, 0.5 + 0.4 * float(win.get("S_mean") or 0.0) - float(s_std)))
    return RecommendationReport(
        winner=win["inhibitor"],
        winner_rationale=(
            f"Highest computed selectivity S={win.get('S_mean')}+/-{win.get('S_std')} "
            f"at target thickness with differential blocking "
            f"{win.get('differential_blocking')} "
            f"(dE_ngs={win.get('dE_ngs_mean_eV')} eV vs dE_gs={win.get('dE_gs_mean_eV')} "
            f"eV); verdict '{win.get('verdict')}'."
        ),
        runners_up=[r["inhibitor"] for r in scored[1:4]],
        risks=risks,
        committed_candidate_outcome=committed_note,
        confidence=round(conf, 2),
    )


class RecommendationAgent:
    def __init__(self, offline: bool = False) -> None:
        self.offline = offline

    def recommend(self, rows: list[dict], committed_inhibitor: str = "",
                  target_selectivity: float = 0.9) -> RecommendationReport:
        base = _deterministic(rows, committed_inhibitor)
        if self.offline or not base.winner:
            return base

        table = "\n".join(_fmt_row(r) for r in rows)
        user = (
            f"Screening table ({len(rows)} candidates; target selectivity "
            f"{target_selectivity:.0%} at target thickness). Rows sorted as screened:\n"
            f"{table}\n\n"
            f"Committed hypothesis inhibitor: '{committed_inhibitor or 'none'}'.\n"
            f"Required winner (top by computed S): '{base.winner}'.\n"
            "Write the final RecommendationReport."
        )
        try:
            from ..llm import structured_call

            rec = structured_call(RecommendationReport, _SYSTEM, user)
            # Physics guard: the LLM may only narrate, never change the ranking.
            if rec.winner.strip().lower() != base.winner.strip().lower():
                logger.warning(
                    "recommendation agent tried to override the computed winner "
                    "('%s' vs '%s'); keeping the computed one", rec.winner, base.winner,
                )
                rec.winner = base.winner
                rec.risks = list(rec.risks) + [
                    "LLM narrative disagreed with the computed ranking; the computed "
                    "winner was enforced."
                ]
            if not rec.runners_up:
                rec.runners_up = base.runners_up
            if not rec.committed_candidate_outcome:
                rec.committed_candidate_outcome = base.committed_candidate_outcome
            return rec
        except Exception as exc:  # noqa: BLE001
            logger.warning("recommendation agent LLM call failed (%s); deterministic "
                           "fallback", exc)
            return base
