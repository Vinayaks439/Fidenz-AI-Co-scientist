"""Figure generation for the Layer-4 manuscript (ADR-007).

Renders the full IEEE figure suite directly from the run artifacts:

* ``selectivity_figure``   -- S vs oxide thickness with target line + ensemble band.
* ``growth_curves_figure`` -- Thk_GS / Thk_NGS vs ALD cycle (the nucleation delay).
* ``energetics_figure``    -- adsorption energies on NGS vs GS against the
  chemisorption/physisorption thresholds and the literature anchor.
* ``site_density_figure``  -- per-site-type densities vs the Kim 2026 acceptance bands.
* ``slab_figure``          -- atomic models of the generated GS/NGS slabs (top + side
  views), loaded from ``datasets/*.extxyz`` or rebuilt deterministically from the
  recorded seed.
* ``molecule_figure``      -- 3D ball-and-stick render of the inhibitor molecule.

Every function returns the written ``Path`` or ``None`` (missing optional dependency,
missing data) so the manuscript always compiles, just with fewer figures.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Element colors (jmol-ish) for slab/molecule renders.
_ECOLOR = {"Si": "#F0C8A0", "O": "#FF0D0D", "N": "#3050F8", "H": "#E8E8E8",
           "C": "#909090", "P": "#FF8000", "S": "#FFFF30", "Cl": "#1FF01F"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 200})
    return plt


# ────────────────────────── data-driven plots ──────────────────────────


def selectivity_figure(rich: dict, out_path: Path) -> Path | None:
    """S vs GS oxide thickness, with the target line and the ensemble band."""
    try:
        plt = _plt()
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable (%s); skipping figure", exc)
        return None

    sel = rich.get("selectivity", {})
    curve = sel.get("curve", {})
    thk_gs, s = curve.get("thk_gs_nm", []), curve.get("S", [])
    if not thk_gs or not s:
        return None
    target = sel.get("target", 0.9)
    target_nm = sel.get("target_thickness_nm", 10.0)
    s_std = sel.get("S_at_target_std", 0.0) or 0.0

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(thk_gs, s, "-", color="#1f77b4", lw=1.5, label=r"$S$(thickness)")
    if s_std:
        ax.fill_between(thk_gs, [min(1, v + s_std) for v in s],
                        [max(-1, v - s_std) for v in s],
                        color="#1f77b4", alpha=0.15, label=r"ensemble $\pm\sigma$")
    ax.axhline(target, ls="--", color="#d62728", lw=1,
               label=f"target {target:.0%} @ {target_nm:g} nm")
    ax.axvline(target_nm, ls=":", color="gray", lw=1)
    ax.set_xlabel("GS oxide thickness (nm)")
    ax.set_ylabel(r"$S=(T_{GS}-T_{NGS})/(T_{GS}+T_{NGS})$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def growth_curves_figure(rich: dict, out_path: Path) -> Path | None:
    """Film thickness vs ALD cycle on GS and NGS -- the nucleation delay, visualized."""
    try:
        plt = _plt()
    except Exception:  # noqa: BLE001
        return None
    sel = rich.get("selectivity", {})
    curve = sel.get("curve", {})
    cyc = curve.get("cycle", [])
    gs, ngs = curve.get("thk_gs_nm", []), curve.get("thk_ngs_nm", [])
    if not cyc or not gs:
        return None
    target_nm = sel.get("target_thickness_nm", 10.0)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(cyc, gs, "-", color="#2ca02c", lw=1.5, label="GS (a-SiO$_2$)")
    if ngs:
        ax.plot(cyc, ngs, "-", color="#d62728", lw=1.5, label="NGS (a-SiN$_x$)")
    ax.axhline(target_nm, ls="--", color="gray", lw=1,
               label=f"{target_nm:g} nm evaluation point")
    ax.set_xlabel("ALD cycle")
    ax.set_ylabel("Film thickness (nm)")
    ax.legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def energetics_figure(rich: dict, out_path: Path) -> Path | None:
    """Adsorption energies on NGS vs GS with regime thresholds + literature anchor."""
    try:
        plt = _plt()
    except Exception:  # noqa: BLE001
        return None
    ads = rich.get("inhibitor_adsorption", {})
    vals = [ads.get("dE_ngs_mean_eV"), ads.get("dE_gs_mean_eV")]
    errs = [ads.get("dE_ngs_std_eV", 0) or 0, ads.get("dE_gs_std_eV", 0) or 0]
    if vals[0] is None or vals[1] is None:
        return None

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    bars = ax.bar(["NGS (a-SiN$_x$)", "GS (a-SiO$_2$)"], vals, yerr=errs,
                  color=["#d62728", "#2ca02c"], width=0.5, capsize=4, alpha=0.85)
    ax.axhline(-0.7, ls="--", color="k", lw=0.8)
    ax.text(1.35, -0.7, "chemisorption\n($<-0.7$ eV)", fontsize=6, va="center")
    ax.axhline(-0.3, ls=":", color="k", lw=0.8)
    ax.text(1.35, -0.3, "physisorption\n($>-0.3$ eV)", fontsize=6, va="center")
    calib = rich.get("calibration_vs_literature") or {}
    lit = calib.get("literature_dE_ngs_eV")
    if lit is not None:
        ax.plot([0], [lit], marker="*", ms=10, color="#ff7f0e", ls="none",
                label=f"literature NGS anchor ({lit} eV)")
        ax.legend(loc="lower right", fontsize=6.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f" {v:.2f}", fontsize=7,
                ha="center", va="bottom" if v >= 0 else "top")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel(r"$\Delta E_{\mathrm{ads}}$ (eV)")
    ax.set_xlim(-0.6, 2.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def site_density_figure(fidelity: dict, out_path: Path) -> Path | None:
    """Per-site-type surface densities vs the Kim 2026 acceptance bands."""
    try:
        plt = _plt()
    except Exception:  # noqa: BLE001
        return None

    panels = []
    for key, label in (("growth_surface", "GS: a-SiO$_2$"),
                       ("non_growth_surface", "NGS: a-SiN$_x$")):
        reps = (fidelity.get(key) or {}).get("reports") or []
        if not reps:
            continue
        rep = reps[0]
        checks = rep.get("site_type_checks", {})
        dens = rep.get("site_densities", {})
        items = [(st, dens.get(st, 0.0), checks.get(st, {}).get("band"))
                 for st in ("OH", "O_bridge", "NH2", "NH_bridge")
                 if dens.get(st)]
        if items:
            panels.append((label, items, rep.get("crystalline_reference", {})))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 2.7),
                             squeeze=False)
    pretty = {"OH": "-OH", "O_bridge": "-O-", "NH2": "-NH$_2$", "NH_bridge": "-NH-"}
    for ax, (label, items, cryst) in zip(axes[0], panels):
        xs = range(len(items))
        for x, (st, d, band) in zip(xs, items):
            if band:
                ax.fill_between([x - 0.35, x + 0.35], band[0], band[1],
                                color="#2ca02c", alpha=0.18, zorder=0)
            ax.bar([x], [d], width=0.5, color="#1f77b4", zorder=2)
            ax.text(x, d, f" {d:.2f}", ha="center", va="bottom", fontsize=7)
        for st, ref in (cryst or {}).items():
            ax.axhline(ref, ls="--", color="#9467bd", lw=1)
            ax.text(len(items) - 0.5, ref, f"crystalline ref {ref}", fontsize=6,
                    va="bottom", ha="right", color="#9467bd")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([pretty.get(st, st) for st, _, _ in items])
        ax.set_ylabel("site density (nm$^{-2}$)")
        ax.set_title(label)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ────────────────────────── atomic-model renders ──────────────────────────


def _load_or_rebuild_slabs(run_dir: Path, rich: dict):
    """Return ``[(label, Atoms), ...]`` for GS and NGS, or [] when impossible.

    Prefers the exact geometries the validation run saved to ``datasets/``;
    otherwise rebuilds deterministically from the recorded seed (same builder,
    same seed => same slab).
    """
    try:
        from ase.io import read as ase_read
    except Exception:  # noqa: BLE001
        return []

    hyp = rich.get("hypothesis", {})
    prov = rich.get("provenance", {})
    seed0 = int(prov.get("seed", 42))
    pairs = [(hyp.get("growth_surface", "a-SiO2"), "SiO2", seed0),
             (hyp.get("non_growth_surface", "a-SiN"), "SiN", seed0 + 1000)]

    out = []
    for label, mat, seed in pairs:
        atoms = None
        saved = sorted((run_dir / "datasets").glob(f"slab_{mat}_*.extxyz"))
        if saved:
            try:
                atoms = ase_read(saved[0])
            except Exception:  # noqa: BLE001
                atoms = None
        if atoms is None:  # rebuild deterministically from the recorded seed
            try:
                from ..surfaces import build_ensemble

                s = build_ensemble(label, n=1, seed0=seed, compute_tier=1)[0]
                atoms = s.atoms
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not rebuild %s slab for figure (%s)", label, exc)
        if atoms is not None:
            out.append((label, atoms))
    return out


def slab_figure(run_dir: Path, rich: dict, out_path: Path) -> Path | None:
    """Top + side atomic-model views of the GS and NGS slabs used in the run."""
    try:
        plt = _plt()
        from ase.visualize.plot import plot_atoms
    except Exception as exc:  # noqa: BLE001
        logger.warning("ase/matplotlib unavailable (%s); skipping slab figure", exc)
        return None

    slabs = _load_or_rebuild_slabs(run_dir, rich)
    if not slabs:
        return None

    colors_of = None
    try:
        colors_of = lambda a: [_ECOLOR.get(sym, "#B0B0B0") for sym in a.get_chemical_symbols()]  # noqa: E731
    except Exception:  # noqa: BLE001
        pass

    fig, axes = plt.subplots(len(slabs), 2, figsize=(5.2, 2.6 * len(slabs)),
                             squeeze=False)
    for row, (label, atoms) in enumerate(slabs):
        for col, (rot, view) in enumerate((("0x,0y,0z", "top view"),
                                           ("-90x,0y,0z", "side view"))):
            ax = axes[row][col]
            kwargs = {"radii": 0.45, "rotation": rot}
            if colors_of is not None:
                kwargs["colors"] = colors_of(atoms)
            try:
                plot_atoms(atoms, ax, **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("plot_atoms failed for %s (%s)", label, exc)
                plt.close(fig)
                return None
            ax.set_title(f"{label} — {view}", fontsize=9)
            ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def molecule_figure(rich: dict, out_path: Path) -> Path | None:
    """3D render of the inhibitor molecule (rdkit 2D fallback -> ase render)."""
    hyp = rich.get("hypothesis", {})
    name = rich.get("novel_compound", {}) or {}
    ident = name.get("smiles") or hyp.get("inhibitor")
    if not ident:
        return None

    # Preferred: 3D conformer via the validation builder, rendered with ase.
    try:
        plt = _plt()
        from ase.visualize.plot import plot_atoms

        from ..validation.mlip import build_molecule

        atoms = build_molecule(ident)
        fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.3))
        for ax, rot in zip(axes, ("0x,0y,0z", "-90x,0y,0z")):
            plot_atoms(atoms, ax, radii=0.4, rotation=rot,
                       colors=[_ECOLOR.get(s, "#B0B0B0")
                               for s in atoms.get_chemical_symbols()])
            ax.set_axis_off()
        fig.suptitle(f"Inhibitor: {hyp.get('inhibitor', ident)}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
    except Exception as exc:  # noqa: BLE001
        logger.warning("3D molecule render failed (%s); trying rdkit 2D", exc)

    try:  # rdkit 2D depiction fallback
        from rdkit import Chem
        from rdkit.Chem import Draw

        from ..validation.mlip import NAME_TO_SMILES

        smiles = NAME_TO_SMILES.get(str(ident).strip().lower(), ident)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Draw.MolToFile(mol, str(out_path), size=(600, 400))
        return out_path
    except Exception as exc:  # noqa: BLE001
        logger.warning("molecule figure unavailable (%s)", exc)
        return None
