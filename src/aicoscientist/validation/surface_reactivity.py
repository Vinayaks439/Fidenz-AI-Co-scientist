"""``surface_reactivity`` validation engine (ADR-004 / ADR-006 / ADR-009).

Implements the graded in-silico test for area-selective ALD as the ADR-009 five-step
protocol, over an *ensemble* of experiment-faithful amorphous surfaces:

1. Build & gate surfaces (Deliverable #1): N a-SiO2 (GS) + N a-SiN (NGS) slabs, each
   passed through the fidelity gate; failures are discarded.
2. Inhibitor adsorption screen: dE_ads on GS vs NGS -- chemisorb on NGS, physisorb on GS.
   Tier 1 computes dE with a foundation MLIP; Tier 0 uses literature/xTB priors (per the
   selection agent) with per-surface scatter.
3. Effective (chemisorbed, purge-surviving) blocking coverage; the DIFFERENTIAL blocking
   theta(NGS) - theta(GS) is the selectivity driver.
4. [Optional Tier-2] precursor barrier (lower bound; calibrate vs literature DFT).
5. Selectivity & verdict: differential blocking -> nucleation delay -> S(N), reported as
   mean +/- std over the ensemble, with a literature-calibration validity flag.

Emits the paper-ready ``asald_results.json`` and ``surface_fidelity.json`` alongside the
standard repo ``ValidationResult``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..models import ValidationPlan, ValidationResult, ValidationVerdict
from ..surfaces import build_ensemble, ensemble_fidelity_summary
from .base import apply_criteria, metric
from .selectivity_model import (
    SelectivityModel,
    blocking_coverage_from_dE,
    coverage_from_dE,
    site_fractions_from_densities,
    site_reactivity,
    site_resolved_blocking,
)

# Kim et al. 2026 built-in site-resolved priors (deltaEr, Ea in eV).
_BUILTIN_SITE_PRIORS: dict[str, dict[str, dict[str, dict]]] = {
    "dmatms": {
        "SiO2": {
            "OH": {"deltaEr_eV": -0.85, "Ea_eV": 0.48},
            "O_bridge": {"deltaEr_eV": 0.64, "Ea_eV": 1.50},
        },
        "SiN": {
            "NH2": {"deltaEr_eV": -0.80, "Ea_eV": 1.34},
            "NH_bridge": {"deltaEr_eV": -0.70, "Ea_eV": 1.54},
        },
    },
    "ets": {
        "SiO2": {
            "OH": {"deltaEr_eV": -0.30, "Ea_eV": 1.10},
            "O_bridge": {"deltaEr_eV": 0.74, "Ea_eV": 1.46},
        },
        "SiN": {
            "NH2": {"deltaEr_eV": -0.95, "Ea_eV": 0.79},
            "NH_bridge": {"deltaEr_eV": -0.85, "Ea_eV": 0.80},
        },
    },
}

# Default site-density fractions when slab atom counts unavailable (Kim et al. 2026).
_DEFAULT_SITE_FRACTIONS: dict[str, dict[str, float]] = {
    "SiO2": {"OH": 0.616, "O_bridge": 0.384},       # 6.19 / (6.19+3.86)
    "SiN": {"NH2": 0.525, "NH_bridge": 0.475},     # 3.91 / (3.91+3.53)
}

logger = logging.getLogger(__name__)


class SurfaceReactivityValidator:
    domain = "surface_reactivity"

    def run(
        self,
        run_id: str,
        hypothesis: str,
        plan: ValidationPlan,
        datasets_dir: Path,
        logs_dir: Path,
    ) -> ValidationResult:
        settings = get_settings()
        spec = plan.data_spec or {}
        run_dir = datasets_dir.parent

        inhibitor = spec.get("inhibitor", "acetic acid")
        # For AI-proposed novel compounds, build the molecule from the emitted SMILES.
        mol_ident = spec.get("inhibitor_smiles") or inhibitor
        prior_source = spec.get("prior_source", "builtin")
        is_novel = prior_source == "ai-proposed"
        precursor = spec.get("precursor", "BDEAS")
        gs_label = spec.get("growth_surface", "a-SiO2")
        ngs_label = spec.get("non_growth_surface", "a-SiN")
        target_nm = float(spec.get("target_thickness_nm", 10.0))
        target_sel = float(spec.get("target_selectivity", 0.90))
        T = float(spec.get("temperature_K", settings.ald_temperature_k))
        dose_ratio = float(spec.get("dose_ratio", 1.0))
        n = int(spec.get("ensemble_n", settings.surface_ensemble_n))
        tier = int(spec.get("compute_tier", settings.compute_tier))

        dE_ngs_prior = float(spec.get("dE_ngs_eV", -1.0))
        dE_gs_prior = float(spec.get("dE_gs_eV", -0.2))
        prior_std = float(spec.get("dE_prior_std", 0.08))
        literature_dE_ngs = spec.get("literature_dE_ngs_eV", dE_ngs_prior)
        barrier = spec.get("barrier")

        seed0 = plan.seed or 42

        # ---- Step 1: build & gate the surface ensembles -----------------------------
        gs_surfaces = build_ensemble(
            gs_label, n=n, seed0=seed0, compute_tier=tier,
            target_density=spec.get("target_density_gs"),
        )
        ngs_surfaces = build_ensemble(
            ngs_label, n=n, seed0=seed0 + 1000, compute_tier=tier,
            target_density=spec.get("target_density_ngs"),
        )
        gs_pass = [s for s in gs_surfaces if s.passed] or gs_surfaces
        ngs_pass = [s for s in ngs_surfaces if s.passed] or ngs_surfaces

        # Persist slab geometries so Layer 4 can render atomic-model figures
        # (and so a judge can inspect the exact structures behind the numbers).
        self._dump_slabs(gs_surfaces + ngs_surfaces, datasets_dir)

        fidelity = {
            "growth_surface": ensemble_fidelity_summary(gs_surfaces),
            "non_growth_surface": ensemble_fidelity_summary(ngs_surfaces),
        }
        (run_dir / "surface_fidelity.json").write_text(
            json.dumps(fidelity, indent=2), encoding="utf-8"
        )

        # ---- Step 2: inhibitor adsorption screen ------------------------------------
        calc, engine = self._maybe_calculator(tier, settings)
        dE_ngs_list, cfg_ngs = self._adsorption_samples(
            ngs_pass, mol_ident, dE_ngs_prior, prior_std, calc, engine, settings
        )
        dE_gs_list, cfg_gs = self._adsorption_samples(
            gs_pass, mol_ident, dE_gs_prior, prior_std, calc, engine, settings
        )
        dE_ngs = np.array(dE_ngs_list, float)
        dE_gs = np.array(dE_gs_list, float)

        site_reactivity_report = None
        use_site = getattr(settings, "use_site_resolved_reactivity", True)

        # ---- Step 3: coverage + differential blocking -------------------------------
        if use_site:
            block_ngs, block_gs, site_reactivity_report = self._site_resolved_blocking(
                ngs_pass, gs_pass, inhibitor, spec, dE_ngs, dE_gs, T, dose_ratio, settings
            )
            theta_ngs = block_ngs.copy()
            theta_gs = block_gs.copy()
        else:
            theta_ngs = np.array([coverage_from_dE(d, T, dose_ratio) for d in dE_ngs])
            theta_gs = np.array([coverage_from_dE(d, T, dose_ratio) for d in dE_gs])
            block_ngs = np.array([blocking_coverage_from_dE(d, T, dose_ratio) for d in dE_ngs])
            block_gs = np.array([blocking_coverage_from_dE(d, T, dose_ratio) for d in dE_gs])

        # ---- Step 4b: RSA steric cap (Phase 2) --------------------------------------
        # A bulky inhibitor cannot reach full monolayer; cap blocking at the RSA jamming
        # fraction for its footprint on the real surface area. Tier-1+ only, so Tier-0
        # numeric runs are unchanged.
        rsa_info = None
        if tier >= 1 and getattr(settings, "use_rsa_coverage", True):
            try:
                from .mlip import build_molecule
                from .rsa import (
                    apply_rsa_cap,
                    molecule_footprint_diameter_nm,
                    rsa_cap_fraction,
                )

                mol = build_molecule(mol_ident)
                ngs_area = ngs_pass[0].area_nm2 if ngs_pass else 1.0
                gs_area = gs_pass[0].area_nm2 if gs_pass else 1.0
                rsa_ngs = rsa_cap_fraction(mol, ngs_area, seed=seed0)
                rsa_gs = rsa_cap_fraction(mol, gs_area, seed=seed0 + 1)
                block_ngs = np.array([apply_rsa_cap(b, rsa_ngs) for b in block_ngs])
                block_gs = np.array([apply_rsa_cap(b, rsa_gs) for b in block_gs])
                rsa_info = {
                    "inhibitor_footprint_diam_nm": round(
                        molecule_footprint_diameter_nm(mol), 3
                    ),
                    "rsa_jamming_fraction_ngs": round(rsa_ngs, 4),
                    "rsa_jamming_fraction_gs": round(rsa_gs, 4),
                }
            except Exception as exc:  # noqa: BLE001 -- keep ideal blocking on failure
                logger.warning("RSA coverage skipped (%s)", exc)

        # ---- Step 5: per-surface selectivity -> ensemble mean +/- std ---------------
        model = SelectivityModel()
        s_samples = []
        for bn, bg in zip(block_ngs, block_gs):
            delay = model.nucleation_delay_cycles(bn, bg)
            s_samples.append(
                model.selectivity_at_thickness(delay, target_nm)["selectivity_at_target"]
            )
        s_samples = np.array(s_samples)
        s_mean, s_std = float(s_samples.mean()), float(s_samples.std())

        verdict_str = (
            "supported" if s_mean >= target_sel
            else "partially_supported" if s_mean >= 0.75 * target_sel
            else "rejected"
        )
        # An AI-proposed novel compound cannot be "supported" on Tier-0 literature priors
        # alone: it has no direct experimental/DFT grounding, so it must be validated on the
        # real slabs (Tier-1 MLIP) first. Cap at partially_supported until then.
        novel_review_note = None
        if is_novel and tier < 1 and verdict_str == "supported":
            verdict_str = "partially_supported"
            novel_review_note = (
                "AI-proposed novel compound: verdict capped at partially_supported because "
                "it has only Tier-0 priors; run Tier-1 (MLIP on real slabs) to confirm."
            )
        verdict = {
            "supported": ValidationVerdict.SUPPORTED,
            "partially_supported": ValidationVerdict.PARTIALLY_SUPPORTED,
            "rejected": ValidationVerdict.REJECTED,
        }[verdict_str]

        # Calibration vs literature (rigor flag).
        calib = None
        if literature_dE_ngs is not None:
            abs_err = abs(float(dE_ngs.mean()) - float(literature_dE_ngs))
            prior_source = spec.get("prior_source", "builtin")
            extrapolated = bool(spec.get("prior_extrapolated", False))
            flag = "ok" if abs_err < 0.3 else "review"
            if extrapolated:  # NGS prior taken from a different surface material
                flag = "review"
            calib = {
                "predicted_dE_ngs_eV": round(float(dE_ngs.mean()), 4),
                "literature_dE_ngs_eV": round(float(literature_dE_ngs), 4),
                "abs_error_eV": round(abs_err, 4),
                "prior_source": prior_source,
                "prior_extrapolated": extrapolated,
                "prior_source_ids": spec.get("prior_source_ids", []),
                "validity_flag": flag,
            }
            # Compare predicted Ea vs literature (Kim et al. 2026) when available.
            lit_ea = spec.get("literature_Ea_ngs_eV")
            if lit_ea is not None and site_reactivity_report:
                pred_ea = site_reactivity_report.get("ngs_mean_Ea_eV")
                if pred_ea is not None:
                    ea_err = abs(float(pred_ea) - float(lit_ea))
                    calib["literature_Ea_ngs_eV"] = round(float(lit_ea), 4)
                    calib["predicted_Ea_ngs_eV"] = round(float(pred_ea), 4)
                    calib["Ea_abs_error_eV"] = round(ea_err, 4)
                    if ea_err > 0.3:
                        calib["validity_flag"] = "review"

            # Tier-2 GFN2-xTB spot-check: an independent cross-check of the MLIP dE.
            if tier >= 2 and calc is not None:
                try:
                    from .xtb import spotcheck_dE, xtb_available

                    surf = next((s for s in ngs_pass if s.atoms is not None), None)
                    if xtb_available() and surf is not None:
                        from .mlip import build_molecule

                        n_sp = getattr(settings, "xtb_spotcheck_sites", 1)
                        xr = spotcheck_dE(
                            surf.atoms, build_molecule(mol_ident), surf.material,
                            n_sites=n_sp, n_rot=1, heights=(2.4,),
                        )
                        xtb_err = abs(float(xr["dE_ads_eV"]) - float(dE_ngs.mean()))
                        calib["xtb_dE_ngs_eV"] = round(float(xr["dE_ads_eV"]), 4)
                        calib["xtb_method"] = xr["method"]
                        calib["xtb_vs_mlip_abs_error_eV"] = round(xtb_err, 4)
                        if xtb_err > 0.4:  # MLIP and xTB disagree strongly -> flag
                            calib["validity_flag"] = "review"
                    else:
                        calib["xtb_spotcheck"] = "unavailable (tblite not installed)"
                except Exception as exc:  # noqa: BLE001
                    calib["xtb_spotcheck"] = f"failed: {exc}"

        # Selectivity curve for the Layer-4 figure (mean differential blocking).
        delay_mean = model.nucleation_delay_cycles(float(block_ngs.mean()), float(block_gs.mean()))
        n_cyc, thk_gs, thk_ngs, s_curve = model.selectivity_curve(delay_mean)

        rich = {
            "hypothesis": {
                "statement": hypothesis,
                "growth_surface": gs_label,
                "non_growth_surface": ngs_label,
                "inhibitor": inhibitor,
                "precursor": precursor,
                "target_film": spec.get("target_film", "SiOx"),
                "target_thickness_nm": target_nm,
                "target_selectivity": target_sel,
                "provenance_refs": spec.get("provenance_refs", []),
            },
            "surface_ensemble": {
                "n_surfaces_gs": int(dE_gs.size),
                "n_surfaces_ngs": int(dE_ngs.size),
                "fidelity_reports": (
                    fidelity["growth_surface"]["reports"]
                    + fidelity["non_growth_surface"]["reports"]
                ),
                "all_surfaces_passed_gate": bool(
                    fidelity["growth_surface"]["all_passed"]
                    and fidelity["non_growth_surface"]["all_passed"]
                ),
            },
            "inhibitor_adsorption": {
                "engine": engine,
                "dE_ngs_mean_eV": round(float(dE_ngs.mean()), 4),
                "dE_ngs_std_eV": round(float(dE_ngs.std()), 4),
                "dE_gs_mean_eV": round(float(dE_gs.mean()), 4),
                "dE_gs_std_eV": round(float(dE_gs.std()), 4),
                "theta_eq_ngs_mean": round(float(theta_ngs.mean()), 4),
                "theta_eq_gs_mean": round(float(theta_gs.mean()), 4),
                "blocking_ngs_mean": round(float(block_ngs.mean()), 4),
                "blocking_gs_mean": round(float(block_gs.mean()), 4),
                "differential_blocking": round(float(block_ngs.mean() - block_gs.mean()), 4),
                "differential_selectivity_signal": round(
                    float(dE_gs.mean() - dE_ngs.mean()), 4
                ),
                "configs_ngs": cfg_ngs,
                "configs_gs": cfg_gs,
                "rsa_coverage": rsa_info,
                "site_resolved": site_reactivity_report,
            },
            "precursor_barrier": barrier,
            "selectivity": {
                "metric": "S = (Thk_GS - Thk_NGS)/(Thk_GS + Thk_NGS)",
                "target": target_sel,
                "target_thickness_nm": target_nm,
                "S_at_target_mean": round(s_mean, 4),
                "S_at_target_std": round(s_std, 4),
                "curve": {
                    "cycle": [int(c) for c in n_cyc[::10]],
                    "thk_gs_nm": [round(float(v) / 10, 3) for v in thk_gs[::10]],
                    "thk_ngs_nm": [round(float(v) / 10, 3) for v in thk_ngs[::10]],
                    "S": [round(float(v), 4) for v in s_curve[::10]],
                },
            },
            "calibration_vs_literature": calib,
            "novel_compound": {
                "is_ai_proposed": is_novel,
                "smiles": spec.get("inhibitor_smiles"),
                "validated_on_real_slabs": bool(is_novel and tier >= 1),
                "note": novel_review_note,
            } if is_novel else None,
            "verdict": verdict_str,
            "provenance": {
                "engine": engine,
                "compute_tier": tier,
                "mlip_model": settings.mlip_model if tier >= 1 else None,
                "mlip_device": settings.resolved_mlip_device if tier >= 1 else None,
                "temperature_K": T,
                "dose_ratio": dose_ratio,
                "ensemble_n": n,
                "seed": seed0,
            },
        }
        (run_dir / "asald_results.json").write_text(
            json.dumps(rich, indent=2), encoding="utf-8"
        )
        (logs_dir / f"surface_reactivity_iter{plan.iteration}.json").write_text(
            json.dumps({"engine": engine, "tier": tier, "verdict": verdict_str,
                        "S_at_target_mean": round(s_mean, 4)}, indent=2),
            encoding="utf-8",
        )

        # ---- repo-schema metrics + verdict ------------------------------------------
        metrics = [
            metric("S_at_target", round(s_mean, 4),
                   note=f"selectivity at {target_nm} nm (mean over ensemble, +/-{s_std:.3f})"),
            metric("differential_blocking",
                   round(float(block_ngs.mean() - block_gs.mean()), 4),
                   note="theta_block(NGS) - theta_block(GS): the selectivity driver"),
            metric("dE_ngs_mean_eV", round(float(dE_ngs.mean()), 4),
                   note="inhibitor adsorption on NGS (chemisorption expected < -0.7)"),
            metric("dE_gs_mean_eV", round(float(dE_gs.mean()), 4),
                   note="inhibitor adsorption on GS (physisorption expected > -0.3)"),
        ]
        if calib is not None:
            metrics.append(
                metric("calibration_abs_error_eV", calib["abs_error_eV"],
                       note=f"MLIP-vs-literature dE delta; flag={calib['validity_flag']}")
            )
        metrics = apply_criteria(metrics, plan.success_criteria)

        confidence = self._confidence(s_mean, target_sel, s_std, calib)
        narrative = (
            f"Tested '{inhibitor}' inhibitor / '{precursor}' precursor over {dE_ngs.size} "
            f"{ngs_label} (NGS) and {dE_gs.size} {gs_label} (GS) gated surfaces using {engine}. "
            f"Differential blocking {block_ngs.mean() - block_gs.mean():.3f}; "
            f"S = {s_mean:.3f} +/- {s_std:.3f} at {target_nm} nm vs target {target_sel:.2f} "
            f"-> {verdict_str}."
        )

        return ValidationResult(
            run_id=run_id,
            hypothesis_statement=hypothesis,
            plan=plan,
            metrics=metrics,
            verdict=verdict,
            confidence=confidence,
            narrative=narrative,
            artifact_paths={
                "asald_results": str(run_dir / "asald_results.json"),
                "surface_fidelity": str(run_dir / "surface_fidelity.json"),
                "log": str(logs_dir / f"surface_reactivity_iter{plan.iteration}.json"),
            },
        )

    # ──────────────────────── helpers ────────────────────────

    @staticmethod
    def _dump_slabs(surfaces, datasets_dir: Path) -> None:
        """Write each built slab to datasets/ as extxyz (skipped for Tier-0 prior runs)."""
        try:
            from ase.io import write as ase_write
        except Exception:  # noqa: BLE001 -- no ase at Tier 0; nothing to dump
            return
        datasets_dir.mkdir(parents=True, exist_ok=True)
        for s in surfaces:
            if s.atoms is None:
                continue
            name = f"slab_{s.material}_{s.seed}.extxyz"
            try:
                ase_write(datasets_dir / name, s.atoms)
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not save slab %s (%s)", name, exc)

    def _maybe_calculator(self, tier: int, settings):
        """Return (calc, engine_label). Falls back to Tier-0 on any import/setup error."""
        if tier < 1:
            return None, "tier0-literature-priors"
        try:
            from .mlip import make_calculator

            calc = make_calculator(settings.mlip_model, settings.resolved_mlip_device)
            return calc, f"{settings.mlip_model}@{settings.resolved_mlip_device}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLIP unavailable (%s); falling back to Tier-0 priors", exc)
            return None, "tier0-literature-priors (MLIP unavailable)"

    def _adsorption_samples(self, surfaces, inhibitor, prior_mean, prior_std, calc, engine,
                            settings=None):
        """Return (dE_per_surface, configs).

        Runs the multi-site/orientation MLIP search per surface when a calculator and an
        ASE slab are available; otherwise falls back to the literature prior + per-surface
        scatter (Tier-0). ``configs`` collects the sampled placements for the figures.
        """
        out: list[float] = []
        configs: list[dict] = []
        mol = None
        if calc is not None:
            try:
                from .mlip import build_molecule

                mol = build_molecule(inhibitor)
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not build molecule '%s' (%s); using priors",
                               inhibitor, exc)
                mol = None

        n_sites = getattr(settings, "n_adsorption_sites", 4) if settings else 4
        n_rot = getattr(settings, "adsorption_rotations", 4) if settings else 4
        heights = tuple(settings.adsorption_heights) if settings else (1.8, 2.4)

        for s in surfaces:
            dE = None
            if calc is not None and mol is not None and s.atoms is not None:
                try:
                    from .mlip import adsorption_energy_search

                    res = adsorption_energy_search(
                        s.atoms, mol, calc, s.material,
                        n_sites=n_sites, n_rot=n_rot, heights=heights,
                    )
                    dE = res["dE_ads_eV"]
                    configs.append({"seed": s.seed, "dE_ads_eV": dE,
                                    "n_configs": res["n_configs"], "regime": res["regime"]})
                except Exception as exc:  # noqa: BLE001
                    logger.warning("MLIP adsorption search failed on surface %s (%s); "
                                   "using prior", s.seed, exc)
                    dE = None
            if dE is None:
                rng = np.random.default_rng(s.seed)
                dE = float(prior_mean + rng.normal(0.0, prior_std))
                configs.append({"seed": s.seed, "dE_ads_eV": round(dE, 4),
                                "source": "prior"})
            out.append(dE)
        return out, configs

    def _site_resolved_blocking(
        self, ngs_surfaces, gs_surfaces, inhibitor, spec,
        dE_ngs, dE_gs, T, dose_ratio, settings,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Compute blocking via Kim et al. 2026 site-resolved deltaEr + Ea model."""
        inh_key = inhibitor.strip().lower()
        site_priors_spec = spec.get("site_reactivity") or {}
        dose_t = getattr(settings, "dose_time_s", 60.0)

        def priors_for(material: str, dE_terminal: float) -> dict[str, dict]:
            merged = dict(_BUILTIN_SITE_PRIORS.get(inh_key, {}).get(material, {}))
            for st, p in (site_priors_spec.get(material) or {}).items():
                merged[st] = {**merged.get(st, {}), **p}
            if not merged:
                st = "OH" if material == "SiO2" else "NH2"
                merged[st] = {"deltaEr_eV": dE_terminal, "Ea_eV": None}
            return merged

        def uses_full_site_model(ngs_priors: dict, gs_priors: dict) -> bool:
            if inh_key in _BUILTIN_SITE_PRIORS or site_priors_spec:
                return True
            return len(ngs_priors) > 1 or len(gs_priors) > 1

        def fractions_for(
            surfaces, material: str, priors: dict[str, dict]
        ) -> dict[str, float]:
            if surfaces and surfaces[0].fidelity_report.get("site_densities"):
                fr = site_fractions_from_densities(
                    surfaces[0].fidelity_report["site_densities"]
                )
                if fr:
                    return fr
            if len(priors) == 1:
                st = next(iter(priors))
                return {st: 1.0}
            return _DEFAULT_SITE_FRACTIONS.get(
                material,
                {"OH": 1.0} if material == "SiO2" else {"NH2": 1.0},
            )

        ngs_priors_mean = priors_for("SiN", float(dE_ngs.mean()))
        gs_priors_mean = priors_for("SiO2", float(dE_gs.mean()))
        full_site = uses_full_site_model(ngs_priors_mean, gs_priors_mean)

        def blocking_one(surfaces, material, dE_arr) -> np.ndarray:
            if not full_site:
                # Terminal-site special case: preserve legacy blocking curve.
                return np.array([
                    blocking_coverage_from_dE(d, T, dose_ratio) for d in dE_arr
                ])
            priors = priors_for(material, float(dE_arr.mean()))
            fracs = fractions_for(surfaces, material, priors)
            reactivities = {
                st: site_reactivity(
                    p.get("deltaEr_eV", -0.5),
                    p.get("Ea_eV"),
                    T=T,
                    dose_time_s=dose_t,
                )
                for st, p in priors.items()
            }
            b = site_resolved_blocking(fracs, reactivities)
            rng = np.random.default_rng(42)
            return np.array([max(0.0, min(1.0, b + rng.normal(0, 0.03))) for _ in surfaces])

        block_ngs = blocking_one(ngs_surfaces, "SiN", dE_ngs)
        block_gs = blocking_one(gs_surfaces, "SiO2", dE_gs)

        ngs_priors = ngs_priors_mean
        report = {
            "model": "kim2026_site_resolved" if full_site else "terminal_legacy",
            "ngs_site_priors": ngs_priors,
            "gs_site_priors": gs_priors_mean,
            "ngs_mean_Ea_eV": next(
                (p.get("Ea_eV") for p in ngs_priors.values() if p.get("Ea_eV")), None
            ),
            "blocking_ngs_mean": round(float(block_ngs.mean()), 4),
            "blocking_gs_mean": round(float(block_gs.mean()), 4),
        }
        return block_ngs, block_gs, report

    @staticmethod
    def _confidence(s_mean: float, target: float, s_std: float, calib: dict | None) -> float:
        decisiveness = min(1.0, abs(s_mean - target) / max(target, 1e-6))
        conf = 0.55 + 0.35 * decisiveness
        conf -= min(0.15, s_std)            # wide ensemble spread lowers confidence
        if calib and calib.get("validity_flag") == "review":
            conf -= 0.15                     # uncalibrated MLIP lowers confidence
        return round(max(0.3, min(0.97, conf)), 3)
