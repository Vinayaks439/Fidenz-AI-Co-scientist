"""Layer-3 screening-funnel campaign (batch candidate screening).

Replaces "one candidate per reflect iteration" with a proper funnel:

    pool (N=10-50, configurable)             designer library + proposer fill
      -> Tier-0 prior rank (all N)           heuristic, honest (no committed pin)
      -> MLIP batch screen (shortlist M)     SHARED gated slab ensembles, so every
                                             candidate is scored on identical geometry
      -> full-fidelity re-run (top K)        full ensemble at the configured tier
                                             (xTB cross-check included at tier >= 2)
      -> RecommendationAgent                 winner + runners-up + risks

Every candidate's rich results are preserved under ``screening/`` in the run dir;
``screening_results.json`` holds the campaign table and ``recommendation.json`` the
final judgement. The winner's rich results are restored to ``asald_results.json`` so
the Layer-4 stitcher's single-candidate deep-dive describes the recommended molecule.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

from ..config import get_settings
from ..models import ASALDSpec, OfficialHypothesis, ValidationResult
from ..surfaces import build_ensemble
from .designer import ExperimentDesigner
from .surface_reactivity import SurfaceReactivityValidator

logger = logging.getLogger(__name__)

POOL_MIN, POOL_MAX = 2, 50


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "candidate"


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _row_from(result: ValidationResult, rich: dict | None, stage: str) -> dict:
    m = {mm.name: mm.value for mm in result.metrics}
    calib = (rich or {}).get("calibration_vs_literature") or {}
    sel = (rich or {}).get("selectivity") or {}
    ds = result.plan.data_spec
    return {
        "inhibitor": ds.get("inhibitor"),
        "precursor": ds.get("precursor"),
        "stage": stage,
        "status": "ok",
        "S_mean": m.get("S_at_target"),
        "S_std": sel.get("S_at_target_std"),
        "differential_blocking": m.get("differential_blocking"),
        "dE_ngs_mean_eV": m.get("dE_ngs_mean_eV"),
        "dE_gs_mean_eV": m.get("dE_gs_mean_eV"),
        "verdict": result.verdict.value,
        "confidence": result.confidence,
        "prior_source": ds.get("prior_source"),
        "prior_extrapolated": bool(ds.get("prior_extrapolated")),
        "prior_missing": bool(ds.get("prior_missing")),
        "smiles": ds.get("inhibitor_smiles"),
        "calibration_flag": calib.get("validity_flag"),
        "ensemble_n": ds.get("ensemble_n"),
        "compute_tier": ds.get("compute_tier"),
    }


def _stash_rich(run_dir: Path, name: str) -> dict | None:
    """Copy this candidate's asald_results.json into screening/ and return it."""
    src = run_dir / "asald_results.json"
    if not src.exists():
        return None
    dst_dir = run_dir / "screening"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"asald_{_slug(name)}.json"
    shutil.copyfile(src, dst)
    try:
        return json.loads(src.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def run_screening_campaign(
    run_id: str,
    official: OfficialHypothesis,
    concept_names: list[str] | None = None,
    offline: bool = False,
    datasets_dir: Path | None = None,
    logs_dir: Path | None = None,
) -> tuple[dict, ValidationResult]:
    """Execute the funnel; return ``(screening_dict, winner_ValidationResult)``."""
    settings = get_settings()
    concept_names = concept_names or []
    spec = official.asald or ASALDSpec()
    run_dir = settings.artifacts_path / run_id
    datasets_dir = datasets_dir or run_dir / "datasets"
    logs_dir = logs_dir or run_dir / "logs"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    pool_size = _clamp(settings.screen_pool_size, POOL_MIN, POOL_MAX)
    tier = settings.compute_tier
    seed0 = 42  # one seed for the whole campaign: identical slabs for every candidate

    designer = ExperimentDesigner(offline=offline)
    validator = SurfaceReactivityValidator()

    # ---- Stage A: candidate pool -----------------------------------------------------
    library, provenance, meta = designer.build_candidate_pool(
        official, concept_names, pool_size=pool_size
    )
    precursor, target_film = designer._choose_precursor(library, spec)
    logger.info(
        "screening pool: %d candidates (%d built-in, %d KG-mined, %d manual, "
        "%d AI-proposed)", meta["pool_size"], meta["n_builtin"], meta["n_kg"],
        meta["n_manual"], meta["n_proposed"],
    )

    # ---- Stage B: Tier-0 prior rank over the full pool (honest, no committed pin) ----
    ranked = designer.rank_candidates(
        library, spec, concept_names, provenance, honor_committed=False
    )
    rows: dict[str, dict] = {}
    for rank0, (name, props) in enumerate(ranked, start=1):
        prov = provenance.get(name, {})
        rows[name] = {
            "inhibitor": name,
            "precursor": precursor,
            "stage": "tier0",
            "status": "ok",
            "tier0_rank": rank0,
            "S_mean": None,
            "S_std": None,
            "differential_blocking": None,
            "dE_ngs_prior_eV": props.get("dE_ngs"),
            "dE_gs_prior_eV": props.get("dE_gs"),
            "functional_group": props.get("functional_group"),
            "volatility": props.get("volatility"),
            "removability": props.get("removability"),
            "smiles": props.get("smiles"),
            "prior_source": prov.get("dE_ngs_source", "builtin"),
            "prior_extrapolated": bool(prov.get("ngs_extrapolated")),
            "prior_missing": bool(prov.get("prior_missing")),
        }

    # ---- Stage C: batch screen the shortlist on SHARED slabs -------------------------
    m_short = _clamp(settings.screen_shortlist_m, 1, len(ranked))
    shortlist = list(ranked[:m_short])
    # The committed hypothesis molecule is always screened (even if prior-ranked low)
    # so the final recommendation can speak to the hypothesis with computed numbers.
    committed = (spec.inhibitor or "").strip().lower()
    if committed and committed not in {n.lower() for n, _ in shortlist}:
        extra = next(((n, p) for n, p in ranked if n.lower() == committed), None)
        if extra is not None:
            shortlist.append(extra)

    n_batch = max(1, settings.screen_ensemble_n)
    logger.info("batch screen: %d candidate(s) on shared ensembles (n=%d, tier=%d)",
                len(shortlist), n_batch, tier)
    gs_batch = build_ensemble(spec.growth_surface, n=n_batch, seed0=seed0,
                              compute_tier=tier)
    ngs_batch = build_ensemble(spec.non_growth_surface, n=n_batch, seed0=seed0 + 1000,
                               compute_tier=tier)

    results: dict[str, ValidationResult] = {}

    def _screen_one(name: str, props: dict, ensemble_n: int, surfaces, stage: str,
                    iteration: int) -> None:
        prov = provenance.get(name, {})
        trace = [
            f"Screening funnel {stage} stage: pool={meta['pool_size']}, "
            f"shortlist={len(shortlist)}, candidate '{name}' "
            f"(tier-0 rank {rows[name].get('tier0_rank')}).",
            f"Paired with precursor '{precursor}' for target film {target_film}.",
            "Shared slab ensembles (identical seeds for every candidate) make the "
            "comparison apples-to-apples.",
        ]
        plan = designer.build_plan(
            spec, name, props, prov, precursor, target_film,
            tier=tier, ensemble_n=ensemble_n, seed=seed0, iteration=iteration,
            trace=trace,
        )
        try:
            res = validator.run(
                run_id=run_id, hypothesis=official.statement, plan=plan,
                datasets_dir=datasets_dir, logs_dir=logs_dir, surfaces=surfaces,
            )
        except Exception as exc:  # noqa: BLE001 -- one bad molecule must not kill the campaign
            logger.warning("screening of '%s' failed (%s); skipping", name, exc)
            rows[name].update({"stage": stage, "status": f"failed: {exc}"})
            return
        rich = _stash_rich(run_dir, name)
        keep = {k: rows[name].get(k) for k in
                ("tier0_rank", "functional_group", "volatility", "removability")}
        rows[name].update(_row_from(res, rich, stage))
        rows[name].update({k: v for k, v in keep.items() if v is not None})
        results[name] = res

    for i, (name, props) in enumerate(shortlist):
        _screen_one(name, props, n_batch, (gs_batch, ngs_batch), "mlip_batch", i)

    # ---- Stage D: full-fidelity re-run of the top K ----------------------------------
    screened = [rows[n] for n, _ in shortlist if rows[n].get("S_mean") is not None]
    screened.sort(key=lambda r: (r["S_mean"], r.get("differential_blocking") or 0.0),
                  reverse=True)
    k = _clamp(settings.screen_top_k, 1, max(1, len(screened)))
    top_names = [r["inhibitor"] for r in screened[:k]]

    n_full = max(n_batch, settings.surface_ensemble_n)
    if top_names:
        if n_full != n_batch:
            gs_full = build_ensemble(spec.growth_surface, n=n_full, seed0=seed0,
                                     compute_tier=tier)
            ngs_full = build_ensemble(spec.non_growth_surface, n=n_full,
                                      seed0=seed0 + 1000, compute_tier=tier)
        else:
            gs_full, ngs_full = gs_batch, ngs_batch
        logger.info("top-%d full-fidelity re-run (n=%d, tier=%d): %s",
                    len(top_names), n_full, tier, ", ".join(top_names))
        by_name = dict(shortlist)
        for j, name in enumerate(top_names):
            _screen_one(name, by_name[name], n_full, (gs_full, ngs_full),
                        "full", len(shortlist) + j)

    # ---- Winner + recommendation ------------------------------------------------------
    finalists = [rows[n] for n in top_names if rows[n].get("S_mean") is not None]
    if not finalists:  # every top-k re-run failed; fall back to the batch ranking
        finalists = screened
    if not finalists:
        raise RuntimeError(
            "screening campaign produced no computed candidate; check the engine logs"
        )
    finalists.sort(key=lambda r: (r["S_mean"], r.get("differential_blocking") or 0.0),
                   reverse=True)
    winner = finalists[0]["inhibitor"]
    winner_result = results[winner]

    # Restore the winner's rich results as the run's headline asald_results.json
    # (each candidate run overwrote it) so Layer 4's deep-dive shows the winner.
    stash = run_dir / "screening" / f"asald_{_slug(winner)}.json"
    if stash.exists():
        shutil.copyfile(stash, run_dir / "asald_results.json")

    from ..agents.recommender import RecommendationAgent

    all_rows = sorted(
        rows.values(),
        key=lambda r: (r.get("S_mean") is not None, r.get("S_mean") or 0.0,
                       -(r.get("tier0_rank") or 999)),
        reverse=True,
    )
    recommendation = RecommendationAgent(offline=offline).recommend(
        all_rows, committed_inhibitor=spec.inhibitor,
        target_selectivity=spec.target_selectivity,
    )

    screening = {
        "config": {
            "mode": "funnel",
            "pool_size": meta["pool_size"],
            "shortlist_m": len(shortlist),
            "top_k": len(top_names),
            "batch_ensemble_n": n_batch,
            "full_ensemble_n": n_full,
            "compute_tier": tier,
            "seed0": seed0,
            "target_selectivity": spec.target_selectivity,
            "target_thickness_nm": spec.target_thickness_nm,
        },
        "pool": meta,
        "rows": all_rows,
        "shortlist": [n for n, _ in shortlist],
        "top_k": top_names,
        "winner": winner,
        "recommendation": recommendation.model_dump(),
    }
    (run_dir / "screening_results.json").write_text(
        json.dumps(screening, indent=2), encoding="utf-8"
    )
    (run_dir / "recommendation.json").write_text(
        json.dumps(recommendation.model_dump(), indent=2), encoding="utf-8"
    )
    logger.info("screening complete: winner '%s' (S=%s); recommendation saved",
                winner, finalists[0].get("S_mean"))
    return screening, winner_result
