"""Surface fidelity gate (Deliverable #1, ADR-003).

Gate every generated slab on its EXPERIMENTAL surface-site density before it is allowed
into a reactivity calculation. Per-site-type bands follow Kim et al. 2026 (Appl. Surf.
Sci.) measured densities for a-SiO2 and a-SiNx, including bridge sites (-O-, -NH-).

Works on an ASE ``Atoms`` slab (real silanol / amine / bridge site counting) or on a
supplied site count + area, so it runs with or without ASE installed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Overall terminal-site acceptance bands (sites / nm^2) -- backward compatible.
# Kim et al. 2026: a-SiO2 silanol -OH 6.19±0.51; a-SiNx amine -NH2 3.91±1.06.
SITE_BANDS: dict[str, tuple[float, float]] = {
    "SiO2": (4.5, 7.5),   # silanol -OH (paper: 6.19 ± 0.51)
    "SiN": (2.5, 5.5),    # amine -NH2 (paper: 3.91 ± 1.06)
}

# Per-site-type bands (Kim et al. 2026, amorphous PECVD surfaces).
SITE_TYPE_BANDS: dict[str, dict[str, tuple[float, float]]] = {
    "SiO2": {
        "OH": (4.5, 7.5),          # silanol (paper: 6.19 ± 0.51)
        "O_bridge": (2.0, 6.0),    # siloxane bridge (paper: 3.86 ± 1.28)
    },
    "SiN": {
        "NH2": (2.5, 5.5),         # amine (paper: 3.91 ± 1.06)
        "NH_bridge": (2.0, 5.5),   # imide bridge (paper: 3.53 ± 0.88)
    },
}

# Crystalline reference densities (paper Section 3.1).
CRYSTALLINE_DENSITIES: dict[str, dict[str, float]] = {
    "SiO2": {"OH": 9.57},
    "SiN": {"NH2": 5.97},
}


def _material_key(material: str) -> str:
    """Map 'a-SiO2', 'SiOx', 'a-SiN', 'SiNx' onto a band key."""
    if "sio" in material.lower() or "silica" in material.lower() or "oxide" in material.lower():
        return "SiO2"
    if "sin" in material.lower() or "nitride" in material.lower():
        return "SiN"
    if material in SITE_BANDS:
        return material
    raise ValueError(f"no acceptance band for material '{material}'")


class SurfaceFidelityGate:
    def __init__(self, material: str):
        self.key = _material_key(material)
        self.material = self.key
        self.lo, self.hi = SITE_BANDS[self.key]

    def count_sites_by_type(self, atoms, surface_depth: float = 2.5) -> dict[str, int]:
        """Return per-site-type counts using an ``ase.neighborlist`` bond graph.

        The Kim et al. 2026 bands are *surface* densities, so counting is restricted to
        the exposed top surface with two windows:

        * **Terminal sites** (OH / NH2) are counted over the *top half* of the slab. They
          only exist where passivation put them (the top termination -- the bulk carries
          no H), and a rough amorphous termination spreads them over several Angstrom, so
          a shallow window would systematically under-count real silanols/amines.
        * **Bridge sites** (siloxane -O- / imide -NH-) are chemically identical to bulk
          network atoms, so they are only counted within ``surface_depth`` Angstrom of
          the topmost heavy (non-H) atom -- the one exposed layer a precursor can reach.
          Without this, bulk bridging O/N makes every real slab fail the bridge gate.

        Site types (Kim et al. 2026):
          OH         -- silanol: O-H whose O binds exactly one Si
          O_bridge   -- siloxane: O bonded to 2 Si, no H
          NH2        -- amine: N-H whose N binds exactly one Si
          NH_bridge  -- imide: N bonded to 2 Si with >= 1 H
        """
        from ase.data import atomic_numbers
        from ase.neighborlist import NeighborList, natural_cutoffs

        H = atomic_numbers["H"]
        O = atomic_numbers["O"]
        N = atomic_numbers["N"]
        Si = atomic_numbers["Si"]
        nums = atoms.numbers

        z = atoms.get_positions()[:, 2]
        heavy_z = z[nums != H]
        z_top_heavy = float(heavy_z.max()) if len(heavy_z) else float(z.max())
        z_bridge_cut = z_top_heavy - surface_depth                 # exposed bridge layer
        z_terminal_cut = 0.5 * (float(z.min()) + float(z.max()))   # capped top half

        cutoffs = natural_cutoffs(atoms, mult=1.2)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)

        def neighbor_types(i: int):
            idx, _ = nl.get_neighbors(i)
            return [nums[j] for j in idx]

        counts = {"OH": 0, "O_bridge": 0, "NH2": 0, "NH_bridge": 0}
        for k, zk in enumerate(nums):
            if z[k] < z_terminal_cut:
                continue
            types = neighbor_types(k)
            if zk == O:
                n_si = sum(1 for t in types if t == Si)
                if H in types and n_si == 1:
                    counts["OH"] += 1
                elif H not in types and n_si == 2 and z[k] >= z_bridge_cut:
                    counts["O_bridge"] += 1
            elif zk == N:
                n_si = sum(1 for t in types if t == Si)
                if H in types:
                    if n_si == 2 and z[k] >= z_bridge_cut:
                        counts["NH_bridge"] += 1
                    elif n_si == 1:
                        counts["NH2"] += 1
        return counts

    def count_sites(self, atoms) -> tuple[int, float, dict[str, float]]:
        """Return (n_terminal_sites, area_nm2, site_densities_per_nm2).

        ``n_terminal_sites`` counts terminal reactive sites (OH or NH2) for backward
        compatibility with the overall ``SITE_BANDS`` gate.
        """
        cell = atoms.get_cell()
        area_A2 = np.linalg.norm(np.cross(cell[0], cell[1]))
        area_nm2 = float(area_A2) / 100.0

        by_type = self.count_sites_by_type(atoms)
        if self.material == "SiO2":
            n_terminal = by_type["OH"]
        else:
            n_terminal = by_type["NH2"]

        densities = {
            k: round(v / area_nm2, 3) if area_nm2 else 0.0
            for k, v in by_type.items()
        }
        return n_terminal, area_nm2, densities

    def check(
        self,
        atoms=None,
        n_sites: Optional[int] = None,
        area_nm2: Optional[float] = None,
        seed: Optional[int] = None,
        descriptors: Optional[dict] = None,
        site_densities: Optional[dict[str, float]] = None,
    ) -> dict:
        per_type_dens: dict[str, float] = site_densities or {}
        if atoms is not None:
            n_sites, area_nm2, per_type_dens = self.count_sites(atoms)
        if n_sites is None or area_nm2 is None:
            raise ValueError("provide an ASE Atoms slab, or n_sites and area_nm2")
        density = n_sites / area_nm2 if area_nm2 else 0.0
        density_ok = self.lo <= density <= self.hi

        # Per-site-type gate (Kim et al. 2026 bands).
        type_bands = SITE_TYPE_BANDS.get(self.material, {})
        type_checks: dict[str, dict] = {}
        types_ok = True
        for st, (lo, hi) in type_bands.items():
            d = per_type_dens.get(st, 0.0)
            ok = lo <= d <= hi if d > 0 else True  # skip empty types
            if d > 0 and not ok:
                types_ok = False
            type_checks[st] = {"density": d, "band": [lo, hi], "passed": ok}

        descriptors_ok, descriptor_reasons = True, []
        if descriptors is not None:
            from .descriptors import descriptors_physical

            descriptors_ok, descriptor_reasons = descriptors_physical(
                descriptors, self.material
            )

        return {
            "material": self.material,
            "site_density_per_nm2": round(density, 3),
            "acceptance_band": [self.lo, self.hi],
            "site_densities": per_type_dens,
            "site_type_checks": type_checks,
            "crystalline_reference": CRYSTALLINE_DENSITIES.get(self.material, {}),
            "n_sites": int(n_sites),
            "area_nm2": round(area_nm2, 3),
            "density_passed": bool(density_ok),
            "site_types_passed": bool(types_ok),
            "descriptors_passed": bool(descriptors_ok),
            "descriptor_reasons": descriptor_reasons,
            "passed": bool(density_ok and types_ok and descriptors_ok),
            "seed": seed,
        }
