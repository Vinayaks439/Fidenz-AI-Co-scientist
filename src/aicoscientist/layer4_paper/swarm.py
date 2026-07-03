"""LangGraph swarm of section-writer agents for the Layer-4 manuscript (ADR-007).

One writer agent per IEEE section, all fanned out in parallel from START and merged by
a dict reducer -- the same swarm pattern Layer 1 uses for literature mining. Each agent
receives the complete artifact payload (numbers, fidelity reports, designer reasoning
trace, the Layer-3 LLM validation summary, and the architecture digest) plus a
section-specific brief, and must emit LaTeX prose grounded ONLY in that payload.

Guardrails per agent:
* output is validated (length, no preamble/markdown, braces balanced); any failure
  degrades to the deterministic long-form fallback in ``sections.py``,
* offline mode skips the LLM entirely and uses the fallbacks, so the manuscript is
  always reproducible without a key.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, TypedDict

from . import sections

logger = logging.getLogger(__name__)


# (key, IEEE title, target words, section-specific brief)
SECTION_SPECS: list[tuple[str, str, int, str]] = [
    ("abstract", "Abstract", 170,
     "Single paragraph, no citations, no \\ref. State the problem (AS-ALD selectivity "
     "prediction), the system (four-layer agentic co-scientist), the experiment "
     "(surfaces built+gated, pair tested, engine/tier), the key numbers (dE_ads on NGS "
     "and GS in eV, differential blocking, S at the target thickness in nm), and the "
     "verdict. If the calibration flag is 'review', say the energetics are flagged. "
     "If payload['screening'] is non-null, this was a screening CAMPAIGN: state the "
     "pool size, how many candidates were computed, and that the reported molecule is "
     "the recommended winner of the funnel."),
    ("keywords", "Keywords", 15,
     "6-9 comma-separated IEEE keywords. No LaTeX commands, one line."),
    ("introduction", "Introduction", 320,
     "Motivate AS-ALD for self-aligned nanofabrication (3D-NAND cell isolation); "
     "explain the inhibitor mechanism (chemisorb on NGS, physisorb+purge on GS, "
     "nucleation delay = selectivity, the S(N) metric); explain the two computational "
     "failure modes (crystalline-slab error on amorphous surfaces; site-matching, not "
     "strongest-binder); state the committed hypothesis verbatim from the payload; "
     "close with an explicit numbered contributions list (\\begin{itemize})."),
    ("architecture", "System Architecture", 320,
     "Describe the four-layer funnel in \\subsection blocks (Layer 1 deep-research KG "
     "swarm; Layer 2 human-in-the-loop commitment; Layer 3 validation loop = surface "
     "builder + selection designer + tiered reactivity engine + Reflection refine "
     "loop + AI experiment planner choosing tier per iteration; Layer 4 this "
     "manuscript swarm). Emphasize the in-silico testing design: fidelity gate before "
     "compute, tiered compute, literature anchoring with validity flags, bounded "
     "reflection. Reference Table~\\ref{tab:provenance}."),
    ("methods_surfaces", "Methods: Amorphous Surface Builder", 360,
     "Deliverable 1. Describe seed->cleave->saturate (passivation rule table) -> "
     "bridge anneal -> fidelity gate -> ensemble. Quote the measured per-site "
     "densities and bands from the payload's surface_fidelity (nm^-2), name the slab "
     "source from provenance, and reference Table~\\ref{tab:fidelity}, "
     "Fig.~\\ref{fig:sites}, and Fig.~\\ref{fig:slabs} (the rendered atomic models)."),
    ("methods_selection", "Methods: Agentic Inhibitor/Precursor Selection", 360,
     "Deliverable 2. Describe the three-step site-matched screening protocol, the "
     "ranking axes (differential adsorption, volatility, removability, site-match, "
     "steric footprint), and the prior merging (KG-mined / manual criteria file / "
     "built-in). If payload['screening'] is non-null, describe the screening FUNNEL "
     "from payload['screening']['config']: pool assembly (library + AI-proposed "
     "novel candidates), honest Tier-0 prior rank (no committed-candidate pin), "
     "MLIP batch screen of the shortlist on identical seed-shared slab ensembles, "
     "full-fidelity top-k re-run, and the recommendation agent that may narrate but "
     "never override the computed ranking. Reproduce the designer's reasoning trace "
     "from the payload as an itemized list. Reference Fig.~\\ref{fig:molecule}."),
    ("methods_protocol", "Methods: In-Silico Testing Protocol", 300,
     "The five-step ADR-009 protocol. Define dEr and Ea with display equations "
     "(\\begin{equation}), the purge argument, the tiered compute (Tier 0 priors / "
     "Tier 1 MLIP multi-site x orientation x height search with frozen-slab "
     "reference / Tier 2 xTB spot-check), the blocking-coverage sum, differential "
     "blocking -> nucleation delay -> S(N) with the S equation, and the "
     "calibration/validity-flag discipline. Name the exact MLIP and device from "
     "provenance. Reference Table~\\ref{tab:calibration}."),
    ("results", "Results", 420,
     "Report EVERY number from the payload with units: dE_ads NGS/GS (eV, with std), "
     "blocking coverages, differential blocking, S at target thickness (nm) with "
     "std vs the target, verdict. Walk the reader through "
     "Table~\\ref{tab:adsorption}, Fig.~\\ref{fig:energetics}, "
     "Fig.~\\ref{fig:growth}, Fig.~\\ref{fig:selectivity}, and "
     "Table~\\ref{tab:calibration}. If payload['screening'] is non-null, FIRST "
     "present the campaign comparatively (Table~\\ref{tab:screening}, "
     "Fig.~\\ref{fig:screening}): how the winner ranked against the runners-up on "
     "computed S and differential blocking, the committed-candidate outcome from "
     "payload['screening']['recommendation'], and any per-candidate flags "
     "(extrapolated/missing priors, ai-proposed, calibration review) -- then give the "
     "winner's deep-dive. Never present the winner as if it were the only molecule "
     "tested. If the calibration flag is 'review' or energies "
     "sit at the clamp bounds (-3.0/+1.0 eV), state plainly that the energetics are "
     "fallback values and the verdict is a pipeline demonstration, not a "
     "quantitative claim."),
    ("discussion", "Discussion", 240,
     "Interpret: differential blocking (not raw Langmuir coverage) as the selectivity "
     "driver; the purge argument; site-matching vs strongest-binder; why the fidelity "
     "gate is the load-bearing rigor element; what the ensemble convention buys; what "
     "the calibration flag means for trust. Be honest about weaknesses."),
    ("limitations", "Limitations", 150,
     "Enumerate: 0 K energetics + ~0.25 eV entropy at process temperature; MLIP "
     "barrier underestimation (lower bounds); procedural slab vs true melt-quench "
     "AIMD; unmodeled byproducts/lateral interactions; calibration-flag caveat; "
     "ensemble size from provenance."),
    ("conclusion", "Conclusion", 170,
     "Verdict with its numbers, what the autonomous pipeline demonstrated end-to-end, "
     "and the reproducibility guarantee. No new information."),
    ("reproducibility", "Reproducibility and Provenance", 140,
     "List the artifact files and what each contains; reference "
     "Table~\\ref{tab:provenance}; include the exact reproduction command in a "
     "verbatim block using the compute tier and device from provenance."),
]

_SYSTEM_TEMPLATE = (
    "You are the '{title}' section-writer agent in a LangGraph swarm autonomously "
    "authoring an IEEE journal manuscript (IEEEtran, two-column) for a study run with the "
    "Fidenz AI Co-scientist (an autonomous in-silico co-scientist, by Pavan Kumar L and "
    "Vinayak S) on area-selective atomic layer deposition. Refer to the system by name as "
    "the 'Fidenz AI Co-scientist'.\n\n"
    "SECTION BRIEF: {brief}\n\n"
    "TARGET LENGTH: about {words} words of substantive technical prose. Long, "
    "detailed, publication-grade IEEE register -- no filler, no marketing.\n\n"
    "HARD RULES:\n"
    "1. Ground every quantitative statement ONLY in the JSON payload (and the "
    "validation summary inside it). NEVER invent, estimate, or recall numbers from "
    "memory. If a value is absent, write 'not recorded'.\n"
    "2. Output RAW LaTeX body text only: no \\documentclass, no \\section header "
    "(the assembler adds it), no markdown fences, no comments.\n"
    "3. Escape special characters (& % # _) and set all math in $...$ or equation "
    "environments. Units: eV for energies, nm for thicknesses, nm^-2 for site "
    "densities, K for temperatures.\n"
    "4. You may reference ONLY these floats (they exist): Fig.~\\ref{{fig:slabs}}, "
    "Fig.~\\ref{{fig:sites}}, Fig.~\\ref{{fig:molecule}}, "
    "Fig.~\\ref{{fig:energetics}}, Fig.~\\ref{{fig:growth}}, "
    "Fig.~\\ref{{fig:selectivity}}, Table~\\ref{{tab:fidelity}}, "
    "Table~\\ref{{tab:adsorption}}, Table~\\ref{{tab:calibration}}, "
    "Table~\\ref{{tab:provenance}} -- plus Fig.~\\ref{{fig:screening}} and "
    "Table~\\ref{{tab:screening}} ONLY when payload['screening'] is non-null -- "
    "and \\cite only keys listed in payload['citation_keys'].\n"
    "5. Scientific honesty is non-negotiable: flagged calibrations, clamped "
    "energies, and small ensembles must be stated, not smoothed over."
)


class SwarmState(TypedDict):
    payload: dict
    sections: Annotated[dict, operator.or_]


def _valid_latex(text: str, min_chars: int) -> bool:
    if not text or len(text.strip()) < min_chars:
        return False
    bad = ("\\documentclass", "\\begin{document}", "```")
    if any(b in text for b in bad):
        return False
    return abs(text.count("{") - text.count("}")) <= 2


def _strip_wrappers(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = max(parts, key=len).strip()
        if text.startswith(("latex", "tex")):
            text = text.split("\n", 1)[-1]
    return text.strip()


_MATH_SEGMENT = None


def _escape_bare_underscores(text: str) -> str:
    """Escape ``_`` outside math segments (the most common LLM LaTeX error).

    LLM writers reliably produce identifiers like ``dE_ngs`` in prose, which LaTeX
    rejects with 'Missing $ inserted'. Math segments ($...$, \\[...\\], equation-like
    environments) keep their subscripts untouched.
    """
    import re

    global _MATH_SEGMENT
    if _MATH_SEGMENT is None:
        envs = "equation|align|eqnarray|displaymath|gather|multline|math"
        _MATH_SEGMENT = re.compile(
            r"(\$\$.*?\$\$|\$[^$]*\$|\\\[.*?\\\]|"
            rf"\\begin\{{(?:{envs})\*?\}}.*?\\end\{{(?:{envs})\*?\}})",
            re.DOTALL,
        )
    parts = _MATH_SEGMENT.split(text)
    return "".join(
        part if _MATH_SEGMENT.fullmatch(part or "")
        else re.sub(r"(?<!\\)_", r"\\_", part or "")
        for part in parts
    )


def _write_one(key: str, title: str, words: int, brief: str, payload: dict) -> str:
    """One writer agent: LLM attempt with validation, deterministic fallback."""
    import json

    try:
        from ..llm import get_llm

        resp = get_llm().invoke([
            ("system", _SYSTEM_TEMPLATE.format(title=title, brief=brief, words=words)),
            ("human", "ARTIFACT PAYLOAD (sole source of truth):\n"
                      + json.dumps(payload, indent=1, default=str)),
        ])
        text = resp.content if hasattr(resp, "content") else str(resp)
        if isinstance(text, list):
            text = "\n".join(b.get("text", "") if isinstance(b, dict) else str(b)
                             for b in text)
        text = _escape_bare_underscores(_strip_wrappers(text))
        min_chars = 60 if key == "keywords" else 400
        if _valid_latex(text, min_chars):
            return text
        logger.warning("writer '%s' produced invalid/short LaTeX; using fallback", key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("writer '%s' LLM call failed (%s); using fallback", key, exc)
    return sections.fallback(key, payload)


def _make_node(key: str, title: str, words: int, brief: str):
    def node(state: SwarmState) -> dict:
        return {"sections": {key: _write_one(key, title, words, brief,
                                             state["payload"])}}

    node.__name__ = f"write_{key}"
    return node


def run_swarm(payload: dict, offline: bool = False) -> dict[str, str]:
    """Write all sections; parallel LangGraph swarm online, direct fallbacks offline."""
    if offline:
        return {key: sections.fallback(key, payload)
                for key, *_ in SECTION_SPECS}

    try:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(SwarmState)
        for key, title, words, brief in SECTION_SPECS:
            graph.add_node(f"write_{key}", _make_node(key, title, words, brief))
            graph.add_edge(START, f"write_{key}")   # parallel fan-out
            graph.add_edge(f"write_{key}", END)
        app = graph.compile()
        out = app.invoke({"payload": payload, "sections": {}})
        got = out.get("sections", {})
    except Exception as exc:  # noqa: BLE001
        logger.warning("manuscript swarm failed (%s); deterministic fallback", exc)
        got = {}

    # Guarantee completeness whatever the swarm returned.
    return {key: got.get(key) or sections.fallback(key, payload)
            for key, *_ in SECTION_SPECS}
