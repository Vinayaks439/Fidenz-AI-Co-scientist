"""Amorphous surface builder (Deliverable #1, ADR-003).

Implements a *melt-quench-cleave-saturate-condense* pipeline with an explicit target
site density and a structural-fidelity gate, then generates an *ensemble* of N slabs per
condition (selectivity is surface-model sensitive, so we report distributions, not point
estimates).

Two tiers, chosen by ``compute_tier`` so the same code runs on a laptop or a GPU box:

* Tier 0 (default, pure-python): produces a controlled site inventory at the requested
  target density with a melt-quench disorder knob (``cooling_rate``). No ASE required.
* Tier >= 1 (ASE available): additionally materializes an ASE ``Atoms`` slab whose
  surface silanol / amine sites are placed at the target density, so the fidelity gate
  counts *real* sites and the reactivity engine can place adsorbates on it.

The condensation sweep nudges an out-of-band draw back toward the experimental band,
mirroring "iteratively condense neighboring OH/NH pairs until the surface hits a target
density inside the experimental band".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .descriptors import describe
from .fidelity_gate import SITE_BANDS, SurfaceFidelityGate

try:  # ASE is optional (Tier >= 1); Tier 0 runs without it.
    from ase import Atoms

    _HAVE_ASE = True
except Exception:  # noqa: BLE001
    _HAVE_ASE = False

# Base slab footprint used to convert a target density into an integer site count.
_BASE_AREA_NM2 = 9.0  # ~3 nm x 3 nm
_SLAB_HEIGHT_A = 20.0


@dataclass
class Surface:
    """One generated, gated amorphous slab in the ensemble."""

    material: str            # band key: "SiO2" | "SiN"
    label: str               # display label, e.g. "a-SiO2"
    seed: int
    target_density: float
    site_density_per_nm2: float
    n_sites: int
    area_nm2: float
    passed: bool
    fidelity_report: dict = field(default_factory=dict)
    descriptors: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)  # slab source / phase / miller / capping
    atoms: Optional[object] = None  # ASE Atoms when Tier >= 1


def _target_density_for(material_key: str) -> float:
    """Default terminal-site target density (Kim et al. 2026 amorphous values)."""
    from .hydroxylation import TARGET_DENSITIES

    td = TARGET_DENSITIES.get(material_key, {})
    if material_key == "SiO2":
        return td.get("OH", 6.19)
    return td.get("NH2", 3.91)


def _build_atoms(material_key: str, n_sites: int, area_nm2: float, rng) -> "Atoms":
    """Materialize a slab with exactly ``n_sites`` countable surface groups.

    Each surface site is an isolated vertical unit standing on its own Si anchor so the
    gate's neighbor heuristic counts it as a single silanol (SiO2) or amine (SiN). Sites
    are spread on a jittered grid with spacing large enough to prevent cross-bonding.
    """
    area_A2 = area_nm2 * 100.0
    a = float(np.sqrt(area_A2))
    ncol = max(1, int(np.ceil(np.sqrt(n_sites))))
    spacing = a / ncol
    symbols: list[str] = []
    positions: list[tuple[float, float, float]] = []

    z_anchor = _SLAB_HEIGHT_A * 0.5
    placed = 0
    for r in range(ncol):
        for c in range(ncol):
            if placed >= n_sites:
                break
            jx = (rng.random() - 0.5) * spacing * 0.25
            jy = (rng.random() - 0.5) * spacing * 0.25
            x = (c + 0.5) * spacing + jx
            y = (r + 0.5) * spacing + jy
            # Si anchor (part of the bulk termination) directly beneath the site.
            symbols.append("Si")
            positions.append((x, y, z_anchor))
            if material_key == "SiO2":
                # silanol: Si-O-H  (O binds exactly one Si -> counted as silanol)
                symbols.append("O")
                positions.append((x, y, z_anchor + 1.63))
                symbols.append("H")
                positions.append((x + 0.3, y, z_anchor + 2.60))
            else:
                # surface amine: Si-N(-H)(-H)  (N-H group -> counted as reactive amine)
                symbols.append("N")
                positions.append((x, y, z_anchor + 1.72))
                symbols.append("H")
                positions.append((x + 0.3, y, z_anchor + 2.70))
                symbols.append("H")
                positions.append((x - 0.3, y, z_anchor + 2.70))
            placed += 1

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[a, a, _SLAB_HEIGHT_A + 10.0],
        pbc=[True, True, False],
    )
    return atoms


def _procedural_slab(key: str, target: float, seed: int, miller, supercell):
    """Build a crystalline-derived, hydroxylated ASE slab for Tier-1 (Phase 1).

    Returns ``(atoms, provenance)`` or raises so the caller can fall back to the toy slab.
    """
    from .crystal_slabs import build_slab
    from .hydroxylation import saturate_surface

    atoms, slab_prov = build_slab(
        key, miller_index=tuple(miller), supercell=tuple(supercell)
    )
    # Per-seed geometric diversity so the ensemble samples real disorder, not one slab.
    atoms.rattle(stdev=0.05, seed=seed)
    atoms, sat = saturate_surface(atoms, key, target_density=target, seed=seed)
    provenance = {**slab_prov, **{f"cap_{k}": v for k, v in sat.items()}}
    return atoms, provenance


def _soft_core_relax(atoms, min_dist: float = 1.5, max_passes: int = 12) -> float:
    """Push apart atoms closer than ``min_dist`` until clear (in place).

    A cheap repulsion-only relaxation run after each melt-quench displacement so the
    amorphized network has no unphysical overlaps (which would fail the descriptor gate).
    Returns the final minimum interatomic distance.
    """
    md = 0.0
    for _ in range(max_passes):
        pos = atoms.get_positions()
        d = atoms.get_all_distances(mic=True)
        np.fill_diagonal(d, 1e9)
        md = float(d.min())
        if md >= min_dist:
            break
        moved = np.zeros_like(pos)
        for i in range(len(atoms)):
            close = np.where(d[i] < min_dist)[0]
            for j in close:
                v = pos[i] - pos[j]
                n = float(np.linalg.norm(v))
                if n < 1e-6:
                    v = np.random.default_rng(i * 997 + j).normal(size=3)
                    n = float(np.linalg.norm(v))
                moved[i] += 0.6 * (min_dist - d[i, j]) * v / n
        atoms.set_positions(pos + moved)
    return md


def _amorphize(atoms, cooling_rate: float, seed: int, anneal_steps: int = 6):
    """Melt-quench amorphization of a crystalline slab's surface region.

    Disorders the top half of the slab with a decreasing-amplitude displacement schedule
    (a cooling ramp) while leaving the lower bulk near-crystalline, so the MLIP still sees
    a physical bulk but the reactive surface carries realistic positional disorder. A
    repulsion relaxation after every step keeps the network overlap-free. Returns
    ``(atoms, rmsd_A)`` where ``rmsd_A`` is the RMS surface displacement from the
    crystalline reference (the amorphization strength).
    """
    atoms = atoms.copy()
    rng = np.random.default_rng(seed)
    pos0 = atoms.get_positions().copy()
    z = pos0[:, 2]
    surf = z >= 0.5 * (z.min() + z.max())  # amorphize the top half; keep the bulk intact
    amp0 = 0.06 * (0.5 + cooling_rate)  # faster cooling -> more frozen-in disorder (Angstrom)
    for step in range(anneal_steps):
        amp = amp0 * (1.0 - step / anneal_steps)  # cooling schedule
        disp = rng.normal(0.0, amp, size=pos0.shape)
        disp[~surf] *= 0.1  # bulk barely moves
        atoms.set_positions(atoms.get_positions() + disp)
        _soft_core_relax(atoms, min_dist=1.5)
    rmsd = float(np.sqrt(np.mean(np.sum((atoms.get_positions()[surf] - pos0[surf]) ** 2, axis=1))))
    return atoms, rmsd


def _amorphous_slab(key: str, target: float, seed: int, miller, supercell, cooling_rate: float):
    """Build a melt-quench amorphized, hydroxylated ASE slab for Tier >= 1 (Phase 2).

    Real crystalline bulk -> surface amorphization -> Table-1 passivation + bridge anneal.
    Returns ``(atoms, provenance)`` or raises so the caller can fall back to the toy slab.
    """
    from .crystal_slabs import build_slab
    from .hydroxylation import saturate_surface

    atoms, slab_prov = build_slab(key, miller_index=tuple(miller), supercell=tuple(supercell))
    atoms, rmsd = _amorphize(atoms, cooling_rate=cooling_rate, seed=seed)
    atoms, sat = saturate_surface(atoms, key, target_density=target, seed=seed)
    provenance = {
        **slab_prov,
        "source": "procedural-amorphous",
        "phase": f"amorphized-{slab_prov.get('phase', key)}",
        "cooling_rate": cooling_rate,
        "amorphization_rmsd_A": round(rmsd, 3),
        **{f"cap_{k}": v for k, v in sat.items()},
    }
    return atoms, provenance


def _md_melt_quench(atoms, settings, seed: int, overrides=None):
    """Real MLIP molecular-dynamics melt-quench of a crystalline slab's mobile region.

    Uses the Tier-1 MACE calculator as the interatomic potential and ASE Langevin
    dynamics: seed velocities at the melt temperature, hold to melt/disorder, quench down
    a temperature ramp, then relax to a 0 K minimum. The bottom fraction is frozen as a
    bulk anchor so the slab keeps a crystalline substrate and does not drift/evaporate.
    ``overrides`` (from the LLM tuner) may set ``melt_temperature_k`` / ``quench_steps``.
    Raises on NaN / instability so the caller can fall back to the geometric amorphizer.
    """
    import numpy as np
    from ase import units
    from ase.constraints import FixAtoms
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.optimize import LBFGS

    from ..validation.mlip import make_calculator

    if settings is None:
        from ..config import get_settings

        settings = get_settings()
    overrides = overrides or {}

    calc = make_calculator(settings.mlip_model, settings.resolved_mlip_device,
                           dispersion=getattr(settings, "mlip_dispersion", True))

    atoms = atoms.copy()
    z = atoms.get_positions()[:, 2]
    cut = z.min() + settings.mq_fix_bottom_frac * (z.max() - z.min())
    atoms.set_constraint(FixAtoms(mask=[zi < cut for zi in z]))
    atoms.calc = calc

    dt = settings.mq_timestep_fs * units.fs
    t_melt = float(overrides.get("melt_temperature_k", settings.mq_melt_temperature_k))
    t_final = float(settings.mq_final_temperature_k)
    quench_steps = int(overrides.get("quench_steps", settings.mq_quench_steps))

    from ..validation.progress import pbar

    MaxwellBoltzmannDistribution(atoms, temperature_K=t_melt)
    dyn = Langevin(atoms, dt, temperature_K=t_melt, friction=settings.mq_friction)

    # 1) melt / equilibrate at the melt temperature
    print(f"[melt-quench] melting {int(settings.mq_melt_steps)} steps @ {t_melt:.0f}K "
          f"(dt={settings.mq_timestep_fs}fs, {len(atoms)} atoms)", flush=True)
    dyn.run(int(settings.mq_melt_steps))

    # 2) quench: step the thermostat setpoint from melt -> final over ~10 stages
    n_stages = 10
    per = max(1, quench_steps // n_stages)
    print(f"[melt-quench] quenching {quench_steps} steps {t_melt:.0f}->{t_final:.0f}K",
          flush=True)
    for i in pbar(range(n_stages), desc="melt-quench quench", total=n_stages):
        t_i = t_melt + (t_final - t_melt) * (i + 1) / n_stages
        dyn.set_temperature(temperature_K=t_i)
        dyn.run(per)

    # 3) relax to a 0 K local minimum
    print("[melt-quench] relaxing to 0 K minimum", flush=True)
    LBFGS(atoms, logfile=None).run(fmax=0.1, steps=200)

    pos = atoms.get_positions()
    if not np.all(np.isfinite(pos)):
        raise RuntimeError("melt-quench produced non-finite coordinates (unstable MD)")
    atoms.set_constraint()  # clear the freeze before passivation
    return atoms


def _md_amorphous_slab(key: str, target: float, seed: int, miller, supercell, settings,
                       overrides=None):
    """Build a genuinely (MD) melt-quench amorphized, hydroxylated ASE slab (Tier >= 1).

    Real crystalline bulk -> MLIP melt-quench MD -> Table-1 passivation + bridge anneal.
    ``overrides`` (from the LLM tuner) may set ``melt_temperature_k`` / ``quench_steps``.
    Returns ``(atoms, provenance)`` or raises so the caller can fall back.
    """
    import numpy as np

    from .crystal_slabs import build_slab
    from .hydroxylation import saturate_surface

    if settings is None:
        from ..config import get_settings

        settings = get_settings()
    overrides = overrides or {}

    atoms, slab_prov = build_slab(key, miller_index=tuple(miller), supercell=tuple(supercell))
    pos0 = atoms.get_positions().copy()
    atoms = _md_melt_quench(atoms, settings, seed, overrides=overrides)

    z = atoms.get_positions()[:, 2]
    surf = z >= 0.5 * (z.min() + z.max())
    rmsd = (
        float(np.sqrt(np.mean(np.sum((atoms.get_positions()[surf] - pos0[surf]) ** 2, axis=1))))
        if surf.any() else 0.0
    )
    atoms, sat = saturate_surface(atoms, key, target_density=target, seed=seed)
    provenance = {
        **slab_prov,
        "source": "mlip-md-melt-quench",
        "phase": f"md-amorphized-{slab_prov.get('phase', key)}",
        "mq_melt_T_K": float(overrides.get("melt_temperature_k", settings.mq_melt_temperature_k)),
        "mq_quench_steps": int(overrides.get("quench_steps", settings.mq_quench_steps)),
        "mq_autotuned": bool(overrides),
        "amorphization_rmsd_A": round(rmsd, 3),
        **{f"cap_{k}": v for k, v in sat.items()},
    }
    return atoms, provenance


def build_surface(
    material: str,
    target_density: Optional[float] = None,
    seed: int = 0,
    cooling_rate: float = 0.3,
    compute_tier: int = 0,
    max_condense_steps: int = 6,
    slab_source: Optional[str] = None,
    slab_miller: Optional[tuple] = None,
    supercell: Optional[tuple] = None,
) -> Surface:
    """Build one gated slab targeting ``target_density`` sites/nm^2.

    Tier 0 -> numeric site inventory. Tier >= 1 -> an ASE slab: a melt-quench amorphized
    hydroxylated surface (``slab_source='amorphous'``, Phase 2, default), a crystalline-
    derived hydroxylated surface (``'procedural'``, the crystalline reference), or the
    legacy toy slab (``'toy'``). Slab-build failures fall back to the toy slab.
    """
    gate = SurfaceFidelityGate(material)
    key = gate.material
    lo, hi = gate.lo, gate.hi
    target = target_density if target_density is not None else _target_density_for(key)

    # Resolve slab settings from config when not explicitly provided.
    if slab_source is None or slab_miller is None or supercell is None:
        try:
            from ..config import get_settings

            s = get_settings()
            slab_source = slab_source or s.slab_source
            slab_miller = slab_miller or s.slab_miller_for(key)
            supercell = supercell or s.slab_supercell
        except Exception:  # noqa: BLE001
            slab_source = slab_source or "procedural"
            slab_miller = slab_miller or (1, 0, 0)
            supercell = supercell or (2, 2)

    rng = np.random.default_rng(seed)
    # Melt-quench disorder: faster cooling (higher cooling_rate) -> more scatter.
    sigma = 0.18 * target * (0.5 + cooling_rate)
    density = float(target + rng.normal(0.0, sigma))

    # Condensation sweep: nudge an out-of-band draw back toward the band center.
    band_center = 0.5 * (lo + hi)
    for _ in range(max_condense_steps):
        if lo <= density <= hi:
            break
        density += 0.5 * (band_center - density)
    density = max(0.01, density)

    n_sites = max(1, int(round(density * _BASE_AREA_NM2)))
    area_nm2 = _BASE_AREA_NM2

    atoms = None
    descriptors: dict = {}
    provenance: dict = {"source": "tier0-numeric"}
    if compute_tier >= 1 and _HAVE_ASE:
        if slab_source in ("md-amorphous", "melt-quench"):
            # Real MLIP melt-quench MD. On instability/blow-up, degrade gracefully to the
            # cheap geometric amorphizer, then the toy slab -- never crash the pipeline.
            mq_overrides = None
            try:
                from ..config import get_settings

                _s = get_settings()
                if getattr(_s, "mq_autotune", False):
                    # LLM tuner picks melt_T/quench once per material (cached), reused here.
                    from .param_tuner import tuned_overrides

                    sig = (
                        round(float(_s.mq_melt_temperature_k)),
                        int(_s.mq_quench_steps),
                        int(_s.mq_autotune_trials),
                        str(_s.mq_autotune_probe_supercell),
                        int(_s.mq_autotune_probe_quench),
                        str(_s.mlip_model),
                        str(_s.resolved_mlip_device),
                    )
                    mq_overrides = tuned_overrides(
                        key, tuple(slab_miller), tuple(supercell), sig
                    )
            except Exception:  # noqa: BLE001 -- tuning is best-effort; use config defaults
                mq_overrides = None
            try:
                atoms, provenance = _md_amorphous_slab(
                    key, density, seed, slab_miller, supercell, None, overrides=mq_overrides
                )
            except Exception as exc:  # noqa: BLE001
                try:
                    atoms, provenance = _amorphous_slab(
                        key, density, seed, slab_miller, supercell, cooling_rate
                    )
                    provenance["md_fallback_reason"] = str(exc)
                except Exception as exc2:  # noqa: BLE001
                    atoms = _build_atoms(key, n_sites, area_nm2, rng)
                    provenance = {"source": "toy-fallback", "reason": f"{exc}; {exc2}"}
        elif slab_source == "amorphous":
            try:
                atoms, provenance = _amorphous_slab(
                    key, density, seed, slab_miller, supercell, cooling_rate
                )
            except Exception as exc:  # noqa: BLE001 -- fall back to the toy slab
                atoms = _build_atoms(key, n_sites, area_nm2, rng)
                provenance = {"source": "toy-fallback", "reason": str(exc)}
        elif slab_source == "procedural":
            try:
                atoms, provenance = _procedural_slab(
                    key, density, seed, slab_miller, supercell
                )
            except Exception as exc:  # noqa: BLE001 -- fall back to the toy slab
                atoms = _build_atoms(key, n_sites, area_nm2, rng)
                provenance = {"source": "toy-fallback", "reason": str(exc)}
        else:
            atoms = _build_atoms(key, n_sites, area_nm2, rng)
            provenance = {"source": "toy"}
        descriptors = describe(atoms)
        report = gate.check(atoms=atoms, seed=seed, descriptors=descriptors)
        n_sites = report["n_sites"]
        area_nm2 = report["area_nm2"]
    else:
        report = gate.check(n_sites=n_sites, area_nm2=area_nm2, seed=seed)
        descriptors = {"n_sites": n_sites, "note": "Tier-0 site inventory (no atoms)"}

    return Surface(
        material=key,
        label=material,
        seed=seed,
        target_density=round(target, 3),
        site_density_per_nm2=report["site_density_per_nm2"],
        n_sites=report["n_sites"],
        area_nm2=report["area_nm2"],
        passed=report["passed"],
        fidelity_report=report,
        descriptors=descriptors,
        provenance=provenance,
        atoms=atoms,
    )


def build_ensemble(
    material: str,
    n: int = 5,
    target_density: Optional[float] = None,
    seed0: int = 42,
    cooling_rate: float = 0.3,
    compute_tier: int = 0,
) -> list[Surface]:
    """Generate an ensemble of N independent gated slabs for one surface condition."""
    from ..validation.progress import pbar

    out = []
    for i in pbar(range(n), desc=f"Build {material} slabs", total=n):
        out.append(
            build_surface(
                material,
                target_density=target_density,
                seed=seed0 + i,
                cooling_rate=cooling_rate,
                compute_tier=compute_tier,
            )
        )
    return out


def ensemble_fidelity_summary(surfaces: list[Surface]) -> dict:
    """Aggregate an ensemble's fidelity for ``surface_fidelity.json``."""
    dens = [s.site_density_per_nm2 for s in surfaces]
    return {
        "material": surfaces[0].material if surfaces else None,
        "n_surfaces": len(surfaces),
        "target_density_per_nm2": surfaces[0].target_density if surfaces else None,
        "acceptance_band": (
            list(SITE_BANDS[surfaces[0].material]) if surfaces else None
        ),
        "site_density_mean": round(float(np.mean(dens)), 3) if dens else 0.0,
        "site_density_std": round(float(np.std(dens)), 3) if dens else 0.0,
        "n_passed": int(sum(s.passed for s in surfaces)),
        "all_passed": bool(all(s.passed for s in surfaces)) if surfaces else False,
        "slab_provenance": surfaces[0].provenance if surfaces else {},
        "descriptors_example": surfaces[0].descriptors if surfaces else {},
        "reports": [s.fidelity_report for s in surfaces],
    }
