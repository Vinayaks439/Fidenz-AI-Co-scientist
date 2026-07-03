"""Reflection agent (per arXiv:2510.27130).

After a validation runs, the Reflection agent critiques the result and decides whether
to accept it or refine the experiment (closed-loop). It looks at the verdict, the
confidence, and how decisively the success criteria were met. With an LLM it produces a
qualitative critique + concrete adjustments; without one it uses deterministic thresholds.
"""

from __future__ import annotations

import logging

from ..llm import structured_call
from ..models import Reflection, ValidationResult, ValidationVerdict

logger = logging.getLogger(__name__)

_REFLECT_SYSTEM = (
    "You are a Reflection agent reviewing an in-silico validation of a scientific "
    "hypothesis. Judge whether the EVIDENCE is trustworthy and complete, or whether "
    "the experiment should be refined. Decide 'accept' or 'refine', give a brief "
    "critique, and list concrete suggested adjustments if refining.\n\n"
    "Scope discipline:\n"
    "- Judge evidence QUALITY only: ensemble size / spread, calibration flags, "
    "convergence, whether metrics map to the declared success criteria. In "
    "screening-funnel mode the candidate space was already explored by a batch "
    "screen, so 'refine' means re-running the SAME winner with better statistics "
    "(e.g. a larger surface ensemble) - never suggest switching molecules.\n"
    "- A clear negative result on solid evidence is an ACCEPT (a trustworthy "
    "rejection), not a reason to refine.\n"
    "- Refining costs real compute: only refine when a concrete deficiency "
    "(wide ensemble spread, flagged calibration, unconverged search) would "
    "plausibly change or solidify the verdict."
)

_CONFIDENCE_FLOOR = 0.6


class ReflectionAgent:
    def __init__(self, offline: bool = False) -> None:
        self.offline = offline

    def review(self, result: ValidationResult, iteration: int, max_iters: int) -> Reflection:
        # Never refine past the budget regardless of agent opinion.
        if iteration + 1 >= max_iters:
            return Reflection(
                decision="accept",
                critique="Iteration budget reached; accepting latest result.",
            )

        if self.offline:
            return self._review_heuristic(result)

        return self._review_llm(result)

    def _review_llm(self, result: ValidationResult) -> Reflection:
        metrics_txt = "\n".join(
            f"- {m.name}={m.value:.4g} (threshold={m.threshold}, passed={m.passed})"
            for m in result.metrics
        )
        user = (
            f"Hypothesis: {result.hypothesis_statement}\n"
            f"Domain/method: {result.plan.domain} / {result.plan.method}\n"
            f"Verdict: {result.verdict.value}; confidence: {result.confidence}\n"
            f"Metrics:\n{metrics_txt}\n\n"
            "Should we accept or refine?"
        )
        try:
            return structured_call(Reflection, _REFLECT_SYSTEM, user)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM reflection failed: %s", exc)
            return self._review_heuristic(result)

    def _review_heuristic(self, result: ValidationResult) -> Reflection:
        from ..config import get_settings

        inconclusive = result.verdict == ValidationVerdict.INCONCLUSIVE
        partial = result.verdict == ValidationVerdict.PARTIALLY_SUPPORTED
        rejected = result.verdict == ValidationVerdict.REJECTED
        low_conf = result.confidence < _CONFIDENCE_FLOOR
        funnel = get_settings().screening_mode.strip().lower() == "funnel"

        # Legacy single mode: a rejected committed candidate -> explore an alternative
        # inhibitor/precursor pair; the designer advances to the next-ranked pair.
        # Funnel mode: the candidate space was already screened, so a decisive
        # rejection of the pool's best candidate is a trustworthy negative -> accept.
        if rejected:
            if funnel:
                if low_conf:
                    return Reflection(
                        decision="refine",
                        critique=(
                            f"Winner rejected at low confidence "
                            f"({result.confidence:.2f}); re-run with a larger surface "
                            "ensemble to solidify the negative before reporting it."
                        ),
                        suggested_adjustments=[
                            "Increase the surface ensemble for tighter S statistics",
                        ],
                    )
                return Reflection(
                    decision="accept",
                    critique=(
                        "Screening winner rejected decisively "
                        f"(confidence {result.confidence:.2f}): the pool's best "
                        "candidate does not meet the target -- a trustworthy "
                        "negative result."
                    ),
                )
            return Reflection(
                decision="refine",
                critique=(
                    f"Committed candidate rejected (S below target at confidence "
                    f"{result.confidence:.2f}); exploring the next-ranked inhibitor/precursor."
                ),
                suggested_adjustments=[
                    "Advance to the next-ranked candidate pair from the selection agent",
                    "Prefer a candidate with weaker growth-surface (GS) physisorption",
                ],
            )

        if inconclusive or (partial and low_conf):
            adjustments = ["Increase synthetic sample size / simulation resolution"]
            if partial:
                adjustments.append("Tighten or re-balance success criteria")
            if inconclusive:
                adjustments.append("Ensure metrics map to the declared success criteria")
            return Reflection(
                decision="refine",
                critique=(
                    f"Verdict '{result.verdict.value}' at confidence "
                    f"{result.confidence:.2f} is not decisive enough."
                ),
                suggested_adjustments=adjustments,
            )
        return Reflection(
            decision="accept",
            critique=(
                f"Verdict '{result.verdict.value}' at confidence "
                f"{result.confidence:.2f} is decisive; accepting."
            ),
        )
