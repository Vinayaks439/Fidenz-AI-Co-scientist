"""Novel-inhibitor proposer agent (Phase 2, ADR-005).

A ReAct-style generative agent that invents *new* AS-ALD small-molecule inhibitor
candidates (not just ranking a fixed library). It reads the Layer-1 knowledge graph
(mined mechanisms, functional-group <-> site rules, prior verdicts) and emits candidates as
``{name, smiles, functional_group, target_surface, expected_dE_range_eV, rationale,
citations}``.

Two paths, same schema:

* **LLM** (real API key): asks the configured model for chemically sensible novel candidates.
* **Offline**: a deterministic combinatorial generator that attaches known selective head
  groups (carboxylic / phosphonic / sulfonic acid, amine, thiol) to small volatile alkyl
  backbones, so the pipeline demonstrates innovation with no key.

AI-proposed candidates are always tagged ``provenance='ai-proposed'`` and
``extrapolated=True`` downstream so they can never be reported as "supported" on Tier-0
priors alone -- they must be validated on the real slabs (see ``surface_reactivity``).
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Head-group chemistry: SMILES fragment appended to a backbone, and rough NGS/GS dE priors
# (eV) used only to seed the search range; the MLIP recomputes the real values.
_HEAD_GROUPS: dict[str, dict] = {
    "carboxylic acid": {"smiles": "C(=O)O", "dE_ngs": (-1.05, -0.85), "dE_gs": -0.24},
    "phosphonic acid": {"smiles": "P(=O)(O)O", "dE_ngs": (-1.40, -1.10), "dE_gs": -0.30},
    "sulfonic acid": {"smiles": "S(=O)(=O)O", "dE_ngs": (-1.25, -1.00), "dE_gs": -0.28},
    "primary amine": {"smiles": "N", "dE_ngs": (-1.00, -0.70), "dE_gs": -0.45},
    "thiol": {"smiles": "S", "dE_ngs": (-1.15, -0.85), "dE_gs": -0.35},
}
# Volatile backbones (SMILES prefix, descriptor). Ten backbones x five head groups
# gives the offline generator up to 50 distinct candidates, enough to fill the
# largest screening pool (SCREEN_POOL_SIZE max) without an LLM key.
_BACKBONES: list[tuple[str, str, str]] = [
    ("CC", "ethyl", "high"),
    ("CCC", "propyl", "high"),
    ("CC(C)", "isopropyl", "high"),
    ("CC(C)(C)", "tert-butyl", "high"),
    ("CCCC", "butyl", "medium"),
    ("CC(C)C", "isobutyl", "medium"),
    ("CCC(C)", "sec-butyl", "medium"),
    ("CCCCC", "pentyl", "medium"),
    ("CCCCCC", "hexyl", "low"),
    ("C1CCCCC1", "cyclohexyl", "low"),
]


class ProposedInhibitor(BaseModel):
    """A single AI-proposed novel inhibitor candidate."""

    name: str
    smiles: str
    functional_group: str
    target_surface: str = "a-SiN"
    expected_dE_range_eV: list[float] = Field(default_factory=lambda: [-1.1, -0.8])
    dE_gs_eV: float = -0.3
    volatility: str = "high"
    removability: str = "high"
    rationale: str = ""
    citations: list[str] = Field(default_factory=list)


class ProposedInhibitorSet(BaseModel):
    candidates: list[ProposedInhibitor] = Field(default_factory=list)


class InhibitorProposer:
    """Generative agent proposing novel inhibitor candidates."""

    def __init__(self, offline: bool = False) -> None:
        self.offline = offline

    def propose(
        self,
        spec,
        concept_names: list[str] | None = None,
        existing_names: set[str] | None = None,
        n: int = 3,
        citations: list[str] | None = None,
    ) -> list[ProposedInhibitor]:
        concept_names = concept_names or []
        existing = {s.lower() for s in (existing_names or set())}
        if not self.offline:
            try:
                return self._propose_llm(spec, concept_names, existing, n, citations or [])
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM proposer failed (%s); using offline generator", exc)
        return self._propose_offline(spec, existing, n, citations or [])

    # ──────────────────────── LLM path ────────────────────────

    def _propose_llm(self, spec, concept_names, existing, n, citations):
        from ..llm import structured_call

        system = (
            "You are an AS-ALD (area-selective ALD) surface-chemistry expert. Propose NOVEL "
            "small-molecule inhibitor candidates that should CHEMISORB strongly on the "
            "non-growth surface and only PHYSISORB weakly on the growth surface, be volatile "
            "enough to dose from vapor, and be removable after growth.\n\n"
            "HARD RULES (proposals violating them are discarded):\n"
            "1. Valid, parseable SMILES for every candidate; no duplicates of each other "
            "or of the existing list (check synonyms and abbreviations, not just exact "
            "names).\n"
            "2. DIVERSITY is mandatory: spread proposals across DIFFERENT head-group "
            "chemistries (e.g. carboxylic/phosphonic/sulfonic acids, amines, thiols, "
            "silanes) and different backbone sizes. Never return a list dominated by "
            "one functional-group family.\n"
            "3. expected_dE_range_eV must be a plausible chemisorption range for that "
            "head group on the stated NGS, justified in the rationale by an analogous "
            "measured/computed system; dE_gs_eV must be a physisorption-regime value "
            "(> -0.6 eV). Do not fabricate precise numbers - give honest ranges. These "
            "are search priors only: the MLIP recomputes the real energetics.\n"
            "4. volatility/removability must be justified by molecular weight and "
            "anchoring strength in the rationale (e.g. a C18 tail is NOT 'high' "
            "volatility; a phosphonate anchor is NOT 'high' removability).\n"
            "5. Every proposal must be dosable from vapor at typical ALD temperatures "
            "(roughly < 350 g/mol unless justified) and plausibly commercially "
            "obtainable or trivially synthesizable.\n"
            "Ground each proposal in the given mechanistic concepts and cite provided "
            "source ids where relevant."
        )
        user = (
            f"Growth surface (GS): {spec.growth_surface}. Non-growth surface (NGS): "
            f"{spec.non_growth_surface}. Target film: {spec.target_film}. "
            f"Existing candidates (do not repeat): {sorted(existing)}. "
            f"Mechanistic KG concepts: {concept_names[:40]}. "
            f"Available citation ids: {citations[:20]}. "
            f"Propose {n} novel inhibitors."
        )
        result = structured_call(ProposedInhibitorSet, system, user)
        out = [c for c in result.candidates if c.smiles and c.name.lower() not in existing]
        return _validated(out)[:n]

    # ──────────────────────── offline path ────────────────────────

    def _propose_offline(self, spec, existing, n, citations):
        target = spec.non_growth_surface or "a-SiN"
        # Prefer head groups matched to the surface: acids for oxide/nitride, amine as backup.
        group_order = ["carboxylic acid", "phosphonic acid", "sulfonic acid", "thiol",
                       "primary amine"]
        out: list[ProposedInhibitor] = []
        # Backbone-major iteration cycles through the head-group chemistries first, so
        # even a small n yields a chemically DIVERSE set (one candidate per family)
        # instead of n near-identical carboxylic acids.
        for prefix, bb_name, vol in _BACKBONES:
            for group in group_order:
                head = _HEAD_GROUPS[group]
                smiles = prefix + head["smiles"]
                name = f"{bb_name} {group}"
                if name.lower() in existing:
                    continue
                lo, hi = head["dE_ngs"]
                out.append(ProposedInhibitor(
                    name=name,
                    smiles=smiles,
                    functional_group=group,
                    target_surface=target,
                    expected_dE_range_eV=[lo, hi],
                    dE_gs_eV=head["dE_gs"],
                    volatility=vol,
                    removability="high" if vol == "high" else "medium",
                    rationale=(
                        f"{group} head group is expected to chemisorb on {target} while the "
                        f"volatile {bb_name} backbone keeps it dose-able and removable; only "
                        f"weak physisorption expected on {spec.growth_surface}."
                    ),
                    citations=list(citations[:2]),
                ))
                if len(out) >= n:
                    return _validated(out)
        return _validated(out)


def _validated(candidates: list[ProposedInhibitor]) -> list[ProposedInhibitor]:
    """Drop candidates whose SMILES rdkit cannot parse (when rdkit is available)."""
    try:
        from rdkit import Chem
    except Exception:  # noqa: BLE001 -- rdkit absent: accept as-is
        return candidates
    ok: list[ProposedInhibitor] = []
    for c in candidates:
        if Chem.MolFromSmiles(c.smiles) is not None:
            ok.append(c)
        else:
            logger.info("dropping proposal with invalid SMILES: %s (%s)", c.name, c.smiles)
    return ok


def to_library_entries(candidates: list[ProposedInhibitor]) -> dict:
    """Convert proposals into designer-library inhibitor entries (tagged ai-proposed)."""
    entries: dict[str, dict] = {}
    for c in candidates:
        lo, hi = (c.expected_dE_range_eV + [-1.0, -0.8])[:2]
        entries[c.name] = {
            "dE_ngs": round(0.5 * (lo + hi), 3),
            "dE_gs": c.dE_gs_eV,
            "functional_group": c.functional_group,
            "volatility": c.volatility,
            "removability": c.removability,
            "smiles": c.smiles,
            "provenance": "ai-proposed",
            "extrapolated": True,
            "citations": c.citations,
        }
    return entries
