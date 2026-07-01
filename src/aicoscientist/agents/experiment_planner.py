"""AI experiment planner for the Layer-3 closed loop (ADR-009 extension).

Instead of a fixed schedule (test the rank-``iteration`` inhibitor at the global compute
tier), this agent lets the LLM act as the co-scientist: each closed-loop iteration it reads
the deep research (Layer-1 KG concepts + the committed hypothesis) and the previous
reflection, then decides

* **which inhibitor** (from the ranked candidate library) to put on the bench this
  iteration, and
* **at what compute tier** -- a cheap Tier-0 literature-prior *screen* vs. an expensive
  Tier-1 *real foundation-MLIP* confirmation.

The intended policy (encoded in the prompt, not hard-coded): explore candidates cheaply at
Tier-0, and spend the expensive real-MLIP compute only on the iteration(s) worth
confirming -- typically the committed-hypothesis inhibitor or a candidate that just passed
a cheap screen. This is what makes "real numbers on only some iterations" an *earned*
decision rather than a fixed rule.

Two paths, same schema:

* **LLM** (real API key): asks the configured model to choose ``inhibitor`` + ``compute_tier``.
* **Offline / fallback**: deterministic -- rank-``iteration`` inhibitor at the global tier,
  escalating to the max tier on the final iteration so a real confirmation still happens.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class IterationPlan(BaseModel):
    """The AI planner's decision for a single closed-loop iteration."""

    inhibitor: str = Field(description="name of the inhibitor to test this iteration")
    compute_tier: int = Field(
        default=0,
        description="0 = cheap literature-prior screen, 1 = real foundation-MLIP, 2 = +xTB",
    )
    rationale: str = Field(default="", description="why this inhibitor + tier this iteration")


class ExperimentPlanner:
    """Chooses the inhibitor + compute tier for each closed-loop iteration."""

    def __init__(self, offline: bool = False) -> None:
        self.offline = offline

    def plan(
        self,
        spec,
        ranked: list[tuple[str, dict]],
        iteration: int,
        max_iters: int,
        max_tier: int,
        default_tier: int,
        concept_names: list[str] | None = None,
        prior_critique=None,
    ) -> IterationPlan:
        """Return the (inhibitor, compute_tier) decision for ``iteration``.

        ``ranked`` is the designer's ranked ``[(name, props), ...]`` candidate library;
        the chosen inhibitor is always validated back against it. ``max_tier`` caps what
        can be requested (e.g. the AI-planner ceiling and/or MLIP availability).
        """
        names = [name for name, _ in ranked]
        if not names:
            return IterationPlan(inhibitor=spec.inhibitor, compute_tier=0,
                                 rationale="no ranked candidates; using committed inhibitor")
        max_tier = max(0, min(int(max_tier), 2))
        concept_names = concept_names or []

        if not self.offline:
            try:
                plan = self._plan_llm(spec, ranked, iteration, max_iters, max_tier,
                                      concept_names, prior_critique)
                return self._validate(plan, ranked, max_tier)
            except Exception as exc:  # noqa: BLE001
                logger.warning("AI planner LLM failed (%s); using deterministic schedule", exc)
        return self._plan_offline(ranked, iteration, max_iters, max_tier, default_tier)

    # ──────────────────────── LLM path ────────────────────────

    def _plan_llm(self, spec, ranked, iteration, max_iters, max_tier, concept_names,
                  prior_critique) -> IterationPlan:
        from ..llm import structured_call

        catalog = [
            {
                "name": name,
                "functional_group": props.get("functional_group", "n/a"),
                "dE_ngs_eV": props.get("dE_ngs"),
                "dE_gs_eV": props.get("dE_gs"),
                "provenance": props.get("provenance", "library"),
                "extrapolated": bool(props.get("extrapolated", False)),
            }
            for name, props in ranked[:15]
        ]
        critique_txt = "none (first iteration)"
        if prior_critique is not None:
            adjustments = "; ".join(getattr(prior_critique, "suggested_adjustments", []) or [])
            critique_txt = (
                f"decision={getattr(prior_critique, 'decision', 'n/a')}; "
                f"critique={getattr(prior_critique, 'critique', '')}; "
                f"suggested={adjustments}"
            )

        system = (
            "You are an AS-ALD (area-selective ALD) co-scientist planning an in-silico "
            "validation campaign as a bounded closed loop. Each iteration you choose ONE "
            "inhibitor to test and the COMPUTE TIER to test it at:\n"
            "  - Tier 0 = cheap literature-prior screen (fast, but only re-states assumed "
            "adsorption energies; can never truly confirm a novel candidate).\n"
            "  - Tier 1 = expensive REAL foundation-MLIP calculation (computes adsorption "
            "energy from the actual 3D structure on real slabs; this is a genuine result).\n"
            "Real compute is a scarce budget. Strategy: SCREEN candidates cheaply at Tier 0, "
            "and spend Tier 1 only to CONFIRM a candidate that is promising or is the "
            "committed hypothesis -- especially AI-proposed/extrapolated candidates, which "
            "CANNOT be trusted on priors alone and must be validated on real slabs. Ground "
            "every choice in the mechanistic concepts and the previous reflection. Return "
            f"compute_tier as an integer in [0, {max_tier}]."
        )
        user = (
            f"Committed hypothesis inhibitor: '{spec.inhibitor}' on NGS "
            f"{spec.non_growth_surface}, growing {spec.target_film} on GS "
            f"{spec.growth_surface} to >= {spec.target_selectivity:.0%} selectivity.\n"
            f"Closed-loop iteration {iteration} of max {max_iters - 1} (0-indexed).\n"
            f"Previous reflection: {critique_txt}\n"
            f"Mechanistic KG concepts (deep research): {concept_names[:40]}\n"
            f"Ranked candidate library (choose inhibitor.name from this list): {catalog}\n"
            f"Decide the inhibitor and compute_tier for THIS iteration."
        )
        return structured_call(IterationPlan, system, user)

    # ──────────────────────── deterministic fallback ────────────────────────

    def _plan_offline(self, ranked, iteration, max_iters, max_tier, default_tier) -> IterationPlan:
        idx = min(iteration, len(ranked) - 1)
        name, props = ranked[idx]
        # Escalate to a real confirmation on the final iteration of the budget.
        is_final = iteration >= max_iters - 1
        tier = min(max_tier, default_tier) if not is_final else max_tier
        # AI-proposed / extrapolated candidates are meaningless on priors -> force real tier.
        if props.get("extrapolated") or props.get("provenance") == "ai-proposed":
            tier = max_tier
        return IterationPlan(
            inhibitor=name,
            compute_tier=tier,
            rationale=(
                f"deterministic schedule: rank-{idx + 1} candidate; "
                + ("real-MLIP confirmation (final iteration / extrapolated candidate)"
                   if tier >= 1 else "cheap Tier-0 screen")
            ),
        )

    # ──────────────────────── validation ────────────────────────

    def _validate(self, plan: IterationPlan, ranked, max_tier) -> IterationPlan:
        """Snap the LLM's inhibitor to a real candidate and clamp the tier."""
        by_lower = {name.lower(): name for name, _ in ranked}
        chosen = by_lower.get((plan.inhibitor or "").strip().lower())
        if chosen is None:
            chosen = ranked[0][0]
            logger.warning("planner picked unknown inhibitor %r; using top-ranked '%s'",
                           plan.inhibitor, chosen)
        tier = max(0, min(int(plan.compute_tier), max_tier))
        return IterationPlan(inhibitor=chosen, compute_tier=tier, rationale=plan.rationale)
