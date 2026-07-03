"""Agentic inhibitor/precursor selection agent (ADR-005, Deliverable #2).

The Layer-3 ReAct designer, specialized for AS-ALD. It:

1. Retrieves candidate inhibitors/precursors from the Layer-1 knowledge graph.
2. Ranks them against a human-editable ``selection_criteria.md`` (volatility,
   functional-group <-> site compatibility, chemisorb-vs-physisorb differential, sterics,
   removability), grounded in the committed :class:`ASALDSpec`.
3. Encodes the chosen ``(inhibitor, precursor)`` pair and its literature adsorption-energy
   priors into a runnable ``ValidationPlan`` for the ``surface_reactivity`` engine.
4. On a Reflection ``refine`` it advances to the next-ranked candidate pair (the swarm-style
   "keep exploring" behavior), bounded by ``MAX_VALIDATION_ITERS``.

Without an LLM key it uses a deterministic ranking so offline runs are fully reproducible.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..config import get_settings
from ..models import (
    ASALDSpec,
    OfficialHypothesis,
    Reflection,
    SuccessCriterion,
    ValidationPlan,
)

logger = logging.getLogger(__name__)

VALID_DOMAINS = {"surface_reactivity"}

_VOL = {"low": 0.0, "medium": 0.5, "high": 1.0}

# Built-in fallback library, literature-informed. NGS numbers for a-SiN are estimates
# (published DFT uses metal/oxide NGS, e.g. aniline chemisorbs -3.59 eV on Ru, -2.17 eV
# on Co); GS (SiO2) physisorption numbers are taken from the cited DFT where available
# (aniline -0.57 eV, Langmuir 2023 10.1021/acs.langmuir.2c03214).
_DEFAULT_LIBRARY = {
    "inhibitors": {
        "acetic acid": {"dE_ngs": -1.00, "dE_gs": -0.20, "functional_group": "carboxylic acid", "volatility": "high", "removability": "high"},
        "pivalic acid": {"dE_ngs": -0.95, "dE_gs": -0.22, "functional_group": "carboxylic acid", "volatility": "high", "removability": "high"},
        "ethylbutyric acid": {"dE_ngs": -0.98, "dE_gs": -0.24, "functional_group": "carboxylic acid", "volatility": "high", "removability": "high"},
        "methanesulfonic acid": {"dE_ngs": -1.15, "dE_gs": -0.25, "functional_group": "sulfonic acid", "volatility": "medium", "removability": "medium"},
        "aniline": {"dE_ngs": -0.90, "dE_gs": -0.57, "functional_group": "aromatic amine", "volatility": "high", "removability": "high"},
        "octadecylphosphonic acid": {"dE_ngs": -1.30, "dE_gs": -0.30, "functional_group": "phosphonic acid", "volatility": "low", "removability": "low"},
        "dmatms": {
            "dE_ngs": -0.80, "dE_gs": -0.85, "functional_group": "aminosilane",
            "volatility": "high", "removability": "high",
            "site_reactivity": {
                "SiO2": {"OH": {"deltaEr_eV": -0.85, "Ea_eV": 0.48},
                         "O_bridge": {"deltaEr_eV": 0.64, "Ea_eV": 1.50}},
                "SiN": {"NH2": {"deltaEr_eV": -0.80, "Ea_eV": 1.34},
                        "NH_bridge": {"deltaEr_eV": -0.70, "Ea_eV": 1.54}},
            },
        },
        "ets": {
            "dE_ngs": -0.95, "dE_gs": -0.30, "functional_group": "chlorosilane",
            "volatility": "high", "removability": "high",
            "site_reactivity": {
                "SiO2": {"OH": {"deltaEr_eV": -0.30, "Ea_eV": 1.10},
                         "O_bridge": {"deltaEr_eV": 0.74, "Ea_eV": 1.46}},
                "SiN": {"NH2": {"deltaEr_eV": -0.95, "Ea_eV": 0.79},
                        "NH_bridge": {"deltaEr_eV": -0.85, "Ea_eV": 0.80}},
            },
        },
    },
    "precursors": {
        "BDEAS": {"target_film": "SiOx"},
        "DIPAS": {"target_film": "SiOx"},
        "HCDS": {"target_film": "SiOx"},
        "TDMAT": {"target_film": "TiN"},
        "TMA": {"target_film": "Al2O3"},
    },
}


_MERGE_KEYS = ("dE_ngs", "dE_gs", "functional_group", "volatility", "removability",
               "site_reactivity", "smiles")

# Precursor preferred reactive sites (Kim et al. 2026 screening protocol).
PRECURSOR_SITE_PREFS: dict[str, list[str]] = {
    "BDEAS": ["OH"],
    "DIPAS": ["OH"],
    "HCDS": ["OH"],
    "TMA": ["OH", "O_bridge"],
    "DMAI": ["OH", "O_bridge"],
    "TDMAT": ["NH2"],
}


def _load_manual_library(path: str) -> dict:
    """Parse the ```json``` candidate block from selection_criteria.md (human override)."""
    p = Path(path)
    if not p.exists():
        logger.info("selection_criteria.md not found at %s; no manual overrides", path)
        return {}
    text = p.read_text(encoding="utf-8")
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        logger.info("no json candidate block in %s; no manual overrides", path)
        return {}
    try:
        lib = json.loads(m.group(1))
        if isinstance(lib, dict):
            return lib
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to parse candidate block (%s); ignoring manual overrides", exc)
    return {}


def _load_kg_candidates(run_id: str) -> dict:
    """Load the literature-mined candidate library written by Layer 1 (kg_candidates.json)."""
    settings = get_settings()
    p = Path(settings.artifacts_path) / run_id / "kg_candidates.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("inhibitors", {}) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to read kg_candidates.json (%s)", exc)
        return {}


def _merge_libraries(manual: dict, kg: dict, run_id: str) -> tuple[dict, dict]:
    """Merge inhibitor priors with precedence set by PRIORS_SOURCE.

    Base is the built-in default; KG-mined and manual layers overlay it. In ``auto`` mode
    KG-mined values win; in ``manual`` mode selection_criteria.md wins. Returns the merged
    ``{inhibitors, precursors}`` library plus a ``provenance`` map recording, per inhibitor,
    which layer set ``dE_ngs`` and whether that value is extrapolated from another surface.
    """
    settings = get_settings()
    manual_inh = (manual or {}).get("inhibitors", {})
    default_inh = _DEFAULT_LIBRARY["inhibitors"]

    # Precedence: later entries override earlier ones for the same key.
    if settings.priors_source.lower() == "manual":
        layers = [("builtin", default_inh), ("kg-mined", kg), ("manual", manual_inh)]
    else:  # auto (default): literature (KG) wins over the shipped manual defaults
        layers = [("builtin", default_inh), ("manual", manual_inh), ("kg-mined", kg)]

    # Case-insensitive names: 'ETS' (manual file) and 'ets' (built-in) are the same
    # molecule; without normalization they enter the pool twice and burn a screening
    # slot on a duplicate.
    layers = [
        (src, {k.strip().lower(): v for k, v in (layer or {}).items()})
        for src, layer in layers
    ]
    names = set().union(*(set(layer.keys()) for _, layer in layers))
    merged: dict[str, dict] = {}
    provenance: dict[str, dict] = {}
    for name in names:
        entry: dict = {}
        prov = {"dE_ngs_source": "builtin", "ngs_extrapolated": False, "source_ids": []}
        for src_name, layer in layers:
            src = layer.get(name)
            if not src:
                continue
            for k in _MERGE_KEYS:
                v = src.get(k)
                if v is None:
                    continue
                if k == "dE_ngs":
                    new_extrap = bool(src.get("ngs_extrapolated", False))
                    # A dE_ngs extrapolated from a DIFFERENT NGS material (e.g. aniline
                    # on Ru metal, -3.59 eV) must never displace a surface-matched prior;
                    # it may only fill a gap. Otherwise it corrupts both the ranking and
                    # the MLIP calibration reference.
                    if new_extrap and "dE_ngs" in entry and not prov["ngs_extrapolated"]:
                        continue
                    entry[k] = v
                    prov["dE_ngs_source"] = src_name
                    prov["ngs_extrapolated"] = new_extrap
                    prov["source_ids"] = src.get("source_ids", prov["source_ids"])
                else:
                    entry[k] = v
        # A candidate with NO energetics evidence must not inherit favorable numbers
        # silently: mark it so ranking places it below any evidenced candidate and the
        # calibration reference is suppressed. The numeric defaults keep it runnable.
        prov["prior_missing"] = "dE_ngs" not in entry
        if prov["prior_missing"]:
            prov["dE_ngs_source"] = "none"
        entry.setdefault("dE_ngs", -1.0)
        entry.setdefault("dE_gs", -0.2)
        merged[name] = entry
        provenance[name] = prov

    precursors = (manual or {}).get("precursors") or _DEFAULT_LIBRARY["precursors"]
    return {"inhibitors": merged, "precursors": precursors}, provenance


class ExperimentDesigner:
    """AS-ALD selection agent (kept named ExperimentDesigner for the Layer-3 graph)."""

    def __init__(self, offline: bool = False) -> None:
        self.offline = offline

    def build_candidate_pool(
        self,
        official: OfficialHypothesis,
        concept_names: list[str] | None = None,
        pool_size: int | None = None,
    ) -> tuple[dict, dict, dict]:
        """Assemble the candidate library: built-in + manual + KG-mined (+ proposer).

        When ``pool_size`` is given (screening funnel), the generative proposer fills
        the pool up to that size with novel, SMILES-validated candidates (tagged
        ai-proposed / extrapolated). Returns ``(library, provenance, meta)``.
        """
        settings = get_settings()
        concept_names = concept_names or []
        spec = official.asald or ASALDSpec()

        manual = _load_manual_library(settings.selection_criteria_path)
        kg = _load_kg_candidates(official.run_id)
        library, provenance = _merge_libraries(manual, kg, official.run_id)

        # Phase 2: let the generative agent invent NOVEL candidates and merge them in.
        # Tagged ai-proposed + extrapolated so they can never be "supported" on priors
        # alone. In funnel mode the proposer also fills the pool to pool_size.
        n_known = len(library.get("inhibitors", {}))
        want = 0
        if getattr(settings, "use_inhibitor_proposer", False):
            want = getattr(settings, "n_proposed_inhibitors", 3)
        if pool_size is not None:
            want = max(want, pool_size - n_known)
        n_proposed = 0
        if want > 0:
            n_proposed = self._merge_proposed(
                library, provenance, spec, concept_names, n=want
            )

        meta = {
            "n_kg": len(kg),
            "n_manual": len((manual or {}).get("inhibitors", {})),
            "n_builtin": len(_DEFAULT_LIBRARY["inhibitors"]),
            "n_proposed": n_proposed,
            "pool_size": len(library.get("inhibitors", {})),
        }
        return library, provenance, meta

    def design(
        self,
        official: OfficialHypothesis,
        concept_names: list[str] | None = None,
        prior_critique: Reflection | None = None,
        iteration: int = 0,
    ) -> ValidationPlan:
        settings = get_settings()
        concept_names = concept_names or []
        spec = official.asald or ASALDSpec()

        library, provenance, meta = self.build_candidate_pool(official, concept_names)
        n_proposed = meta["n_proposed"]
        n_kg, n_manual = meta["n_kg"], meta["n_manual"]

        ranked = self._rank_inhibitors(library, spec, concept_names, provenance)
        # On refinement, advance to the next-ranked inhibitor (persist / keep exploring).
        idx = min(iteration, len(ranked) - 1)
        inhibitor, props = ranked[idx]

        # Compute tier for this iteration. Default: the global COMPUTE_TIER. When the AI
        # planner is enabled (ADR-009), let it choose BOTH the inhibitor and the tier per
        # iteration, bounded by AI_PLANNER_MAX_TIER. Without the planner these flags are
        # inert and the deterministic rank-index schedule at the global tier is used.
        tier = settings.compute_tier
        planner_rationale = ""
        if getattr(settings, "use_ai_planner", False) and ranked:
            from ..agents.experiment_planner import ExperimentPlanner

            iplan = ExperimentPlanner(offline=self.offline).plan(
                spec,
                ranked,
                iteration=iteration,
                max_iters=settings.max_validation_iters,
                max_tier=settings.ai_planner_max_tier,
                default_tier=settings.compute_tier,
                concept_names=concept_names,
                prior_critique=prior_critique,
            )
            by_lower = {name.lower(): (name, p) for name, p in ranked}
            inhibitor, props = by_lower.get(iplan.inhibitor.lower(), (inhibitor, props))
            tier = iplan.compute_tier
            planner_rationale = iplan.rationale

        prov = provenance.get(inhibitor, {"dE_ngs_source": "builtin",
                                          "ngs_extrapolated": False, "source_ids": []})

        precursor, target_film = self._choose_precursor(library, spec)

        trace = [
            f"Prior sources merged (mode={settings.priors_source}): {n_kg} KG-mined, "
            f"{n_manual} manual, {len(_DEFAULT_LIBRARY['inhibitors'])} built-in defaults.",
            f"Retrieved {len(ranked)} inhibitor candidates"
            + (f" ({n_proposed} AI-proposed novel)" if n_proposed else "")
            + "; ranked by differential adsorption + volatility + removability.",
            f"Selected '{inhibitor}' (rank {idx + 1}); dE_ngs from '{prov['dE_ngs_source']}'"
            + (" [extrapolated from another NGS material]" if prov["ngs_extrapolated"] else "")
            + ".",
            f"Paired with precursor '{precursor}' for target film {target_film}.",
        ]
        # Coherence guard: the committed inhibitor should resolve to a real candidate. If
        # it does not (e.g. the hypothesis parser failed to recognize the named molecule),
        # the "honor committed choice" pin never fires and a DIFFERENT molecule gets tested
        # -- flag it loudly rather than silently swapping (the aniline-for-ETS bug).
        committed = (spec.inhibitor or "").strip().lower()
        known_keys = {k.lower() for k in library.get("inhibitors", {})}
        if committed and committed not in known_keys:
            logger.warning(
                "committed inhibitor '%s' is not a known candidate; testing '%s' instead",
                spec.inhibitor, inhibitor,
            )
            trace.append(
                f"WARNING: committed inhibitor '{spec.inhibitor}' did not resolve to a "
                f"known candidate (hypothesis parse may have failed); tested '{inhibitor}' "
                f"by ranking instead."
            )
        elif committed and inhibitor.lower() != committed and not planner_rationale:
            trace.append(
                f"NOTE: committed '{spec.inhibitor}' but ranking selected '{inhibitor}'."
            )
        if planner_rationale:
            trace.append(
                f"AI planner (max tier {settings.ai_planner_max_tier}): tier {tier} -- "
                f"{planner_rationale}"
            )
        if prov["source_ids"]:
            trace.append(f"dE_ngs supported by citations: {', '.join(prov['source_ids'])}.")
        if prior_critique and prior_critique.decision == "refine":
            trace.append(f"Refinement: {prior_critique.critique}")

        return self.build_plan(
            spec, inhibitor, props, prov, precursor, target_film,
            tier=tier, ensemble_n=settings.surface_ensemble_n,
            seed=42 + iteration, iteration=iteration, trace=trace,
        )

    def build_plan(
        self,
        spec: ASALDSpec,
        inhibitor: str,
        props: dict,
        prov: dict,
        precursor: str,
        target_film: str,
        *,
        tier: int,
        ensemble_n: int,
        seed: int,
        iteration: int,
        trace: list[str],
    ) -> ValidationPlan:
        """Encode one (inhibitor, precursor) candidate into a runnable ValidationPlan."""
        settings = get_settings()
        return ValidationPlan(
            domain="surface_reactivity",
            method=(
                f"AS-ALD differential-reactivity protocol (ADR-009): inhibitor "
                f"adsorption screen on {spec.non_growth_surface} (NGS) vs "
                f"{spec.growth_surface} (GS) over a gated surface ensemble, "
                f"blocking coverage -> nucleation delay -> S(N)."
            ),
            rationale=(
                f"'{inhibitor}' ({props.get('functional_group', 'n/a')}) is predicted to "
                f"chemisorb on the NGS and physisorb on the GS, giving a large differential "
                f"blocking coverage."
            ),
            reasoning_trace=trace,
            assumptions=[
                "Adsorption-energy priors are literature/xTB values; Tier-1 MLIP recomputes "
                "them within a single calculator/head/dtype.",
                "Only chemisorbed, purge-surviving inhibitor blocks the precursor.",
                "Selectivity is reported as mean +/- std over the surface ensemble.",
            ],
            seed=seed,
            iteration=iteration,
            data_spec={
                "inhibitor": inhibitor,
                "precursor": precursor,
                "growth_surface": spec.growth_surface,
                "non_growth_surface": spec.non_growth_surface,
                "target_film": target_film,
                "target_thickness_nm": spec.target_thickness_nm,
                "target_selectivity": spec.target_selectivity,
                "inhibitor_smiles": props.get("smiles"),
                "dE_ngs_eV": props["dE_ngs"],
                "dE_gs_eV": props["dE_gs"],
                "dE_prior_std": 0.08,
                # An extrapolated, AI-guessed, or missing dE_ngs is NOT a valid
                # calibration reference for the MLIP (that produced the earlier
                # 3.3 eV "error").
                "literature_dE_ngs_eV": (
                    None
                    if prov.get("ngs_extrapolated")
                    or prov.get("prior_missing")
                    or prov.get("dE_ngs_source") in ("ai-proposed", "none")
                    else props["dE_ngs"]
                ),
                "temperature_K": settings.ald_temperature_k,
                "dose_ratio": 1.0,
                "ensemble_n": ensemble_n,
                "compute_tier": tier,
                "provenance_refs": spec.provenance_refs,
                "prior_source": prov.get("dE_ngs_source", "builtin"),
                "prior_extrapolated": bool(prov.get("ngs_extrapolated")),
                "prior_missing": bool(prov.get("prior_missing")),
                "prior_source_ids": prov.get("source_ids", []),
                "site_reactivity": props.get("site_reactivity"),
                "literature_Ea_ngs_eV": _literature_ea_ngs(props),
            },
            success_criteria=[
                SuccessCriterion(
                    metric="S_at_target", operator=">=", threshold=spec.target_selectivity,
                    description=f"selectivity meets the {spec.target_selectivity:.0%} target",
                ),
                SuccessCriterion(
                    metric="differential_blocking", operator=">=", threshold=0.5,
                    description="NGS is blocked substantially more than GS",
                ),
                SuccessCriterion(
                    metric="dE_ngs_mean_eV", operator="<", threshold=-0.7,
                    description="chemisorption on the non-growth surface",
                ),
            ],
        )

    def _merge_proposed(self, library, provenance, spec, concept_names,
                        n: int | None = None, prior_results: list | None = None) -> int:
        """Generate novel candidates and merge them into ``library`` in-place.

        Returns the number of novel candidates added. Existing library names are never
        overwritten; provenance records them as ai-proposed / extrapolated. ``prior_results``
        (a failed prior generation's screened rows) makes the proposer design a better next
        generation instead of guessing.
        """
        try:
            from ..agents.inhibitor_proposer import InhibitorProposer, to_library_entries
        except Exception as exc:  # noqa: BLE001
            logger.warning("inhibitor proposer unavailable (%s)", exc)
            return 0

        settings = get_settings()
        inhibitors = library.setdefault("inhibitors", {})
        existing = set(inhibitors.keys())
        if n is None:
            n = getattr(settings, "n_proposed_inhibitors", 3)
        try:
            proposer = InhibitorProposer(offline=self.offline)
            candidates = proposer.propose(
                spec, concept_names=concept_names, existing_names=existing,
                n=n, citations=list(spec.provenance_refs or []),
                prior_results=prior_results,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("inhibitor proposer failed (%s)", exc)
            return 0

        added = 0
        for name, entry in to_library_entries(candidates).items():
            if name in inhibitors:
                continue
            inhibitors[name] = entry
            provenance[name] = {
                "dE_ngs_source": "ai-proposed",
                "ngs_extrapolated": True,
                "source_ids": entry.get("citations", []),
            }
            added += 1
        if added:
            logger.info("proposer added %d novel inhibitor candidate(s)", added)
        return added

    # ──────────────────────── ranking ────────────────────────

    def _rank_inhibitors(
        self, library: dict, spec: ASALDSpec, concept_names: list[str],
        provenance: dict | None = None, honor_committed: bool = True,
    ) -> list[tuple[str, dict]]:
        """Rank candidates by the Tier-0 heuristic.

        ``honor_committed=True`` (legacy single-candidate mode) pins the committed
        inhibitor to rank 1 so the hypothesis is always tested first. The screening
        funnel passes ``False``: the committed molecule competes honestly and the
        recommendation agent reports whether it won.
        """
        inhibitors = library.get("inhibitors", {})
        provenance = provenance or {}
        kg_text = " ".join(concept_names).lower()

        def score(name: str, props: dict) -> float:
            differential = props["dE_gs"] - props["dE_ngs"]     # want large positive
            vol = _VOL.get(str(props.get("volatility", "medium")).lower(), 0.5)
            rem = _VOL.get(str(props.get("removability", "medium")).lower(), 0.5)
            s = differential + 0.2 * vol + 0.2 * rem
            if name.lower() in kg_text:      # grounded in the literature KG -> preferred
                s += 0.3
            # A dE_ngs extrapolated from a different NGS material is unreliable evidence:
            # cap its contribution so a huge metal-surface number cannot dominate ranking.
            if provenance.get(name, {}).get("ngs_extrapolated"):
                s = min(s, 0.8 + 0.2 * vol + 0.2 * rem)
                s -= 0.4
            # No energetics evidence at all (defaults were substituted): rank below any
            # evidenced candidate so fabricated defaults cannot flood a large pool.
            if provenance.get(name, {}).get("prior_missing"):
                s = min(s, 0.4)
            # Site-matched screening (Kim et al. 2026): inhibitor should passivate
            # the sites the precursor uses on the NGS.
            s += _site_match_score(props, spec.precursor)
            if props.get("provenance") == "ai-proposed":
                s -= 0.5
            if honor_committed and name.lower() == spec.inhibitor.lower():
                s += 100.0               # honor the committed choice first
            return s

        ranked = sorted(inhibitors.items(), key=lambda kv: score(*kv), reverse=True)
        if not ranked:  # ensure at least the committed inhibitor is runnable
            ranked = [(spec.inhibitor, {"dE_ngs": -1.0, "dE_gs": -0.2,
                                        "functional_group": "n/a"})]
        return ranked

    def rank_candidates(
        self, library: dict, spec: ASALDSpec, concept_names: list[str],
        provenance: dict, honor_committed: bool = False,
    ) -> list[tuple[str, dict]]:
        """Public honest ranking used by the screening funnel."""
        return self._rank_inhibitors(
            library, spec, concept_names, provenance, honor_committed=honor_committed
        )

    @staticmethod
    def _choose_precursor(library: dict, spec: ASALDSpec) -> tuple[str, str]:
        precursors = library.get("precursors", {})
        if spec.precursor in precursors:
            return spec.precursor, precursors[spec.precursor].get("target_film", spec.target_film)
        # else pick a precursor matching the target film, defaulting to the committed one.
        for name, props in precursors.items():
            if props.get("target_film", "").lower() == spec.target_film.lower():
                return name, props["target_film"]
        return spec.precursor, spec.target_film


def _literature_ea_ngs(props: dict) -> float | None:
    """Extract a representative NGS Ea from site-resolved priors if present."""
    sr = props.get("site_reactivity") or {}
    sin = sr.get("SiN") or {}
    for st in ("NH2", "NH_bridge", "OH"):
        ea = (sin.get(st) or {}).get("Ea_eV")
        if ea is not None:
            return float(ea)
    return None


def _site_match_score(props: dict, precursor: str) -> float:
    """Kim et al. 2026 site match for 'passivate the NGS, grow on the GS'.

    A good inhibitor (a) chemisorbs on the NON-growth surface's amine sites (SiN -NH2 /
    -NH), passivating it, and (b) does NOT react the growth sites the precursor needs
    (SiO2 -OH for BDEAS-type), which would kill selectivity. Both terms are scaled by the
    reaction exothermicity (deltaEr) so, e.g., ETS (weak on SiO2-OH) clearly beats DMATMS
    (strongly reacts SiO2-OH). The previous version searched for the precursor's -OH site
    inside the SiN dict, so the bonus was always 0 for BDEAS and the screen never worked.
    """
    sr = props.get("site_reactivity") or {}
    if not sr:
        return 0.0
    score = 0.0
    # (a) reward passivating the NGS (SiN) at its amine sites -- more exothermic is better
    sin = sr.get("SiN") or {}
    for st in ("NH2", "NH_bridge"):
        d = (sin.get(st) or {}).get("deltaEr_eV")
        if d is not None and d < 0:
            score += min(0.4, -0.4 * d)
    # (b) penalize reacting the growth sites the precursor uses (SiO2 -OH) -- this blocks
    # the surface we want to grow on, so a strong exothermic dEr there is disqualifying.
    prefs = PRECURSOR_SITE_PREFS.get(precursor.upper(), ["OH"])
    sio2 = sr.get("SiO2") or {}
    for st in prefs:
        d = (sio2.get(st) or sio2.get(st.replace("_bridge", "")) or {}).get("deltaEr_eV")
        if d is not None and d < 0:
            score -= min(0.6, -0.6 * d)
    return score
