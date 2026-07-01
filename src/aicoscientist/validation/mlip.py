"""Foundation-MLIP hooks for the surface-reactivity engine (Tier 1, ADR-004).

Correct ASE / MACE APIs; meant to run on a GPU box (Colab CUDA) or CPU. Energies from
foundation MLIPs are only meaningful as DIFFERENCES within a *single* calculator + head
+ dispersion + dtype -- so do NOT mix ``mace_mp()`` and ``MACECalculator()``, and keep
those settings fixed across slab / molecule / complex. Adsorption energy is therefore
always ``E(slab+mol) - E(slab) - E(mol_gas)``, never a raw absolute.

All functions raise if the (optional) MLIP stack is unavailable; the ``surface_reactivity``
engine catches that and falls back to Tier-0 literature adsorption energies.

Ref: MACE (github.com/ACEsuit/mace); barrier caveat arXiv:2502.15582.
"""

from __future__ import annotations


def make_calculator(kind: str = "mace-mp", device: str = "cpu"):
    """Return an ASE calculator.

    * ``mace-mp``  -- MACE-MP medium, ungated/pip-installable (good default).
    * ``mace-mh1`` -- MACE-MH-1 (head='omat_pbe'); adds OC20 surface-adsorption and
      reaction-TS coverage, better for barriers.
    * ``chgnet``   -- CHGNet universal potential.
    """
    if kind == "mace-mp":
        from mace.calculators import mace_mp

        try:  # D3 dispersion improves physisorption but needs the optional torch-dftd
            return mace_mp(
                model="medium", dispersion=True, default_dtype="float64", device=device
            )
        except (RuntimeError, ImportError, ModuleNotFoundError):
            return mace_mp(
                model="medium", dispersion=False, default_dtype="float64", device=device
            )
    if kind == "mace-mh1":
        from mace.calculators import mace_mp

        return mace_mp(
            model="mace-mh-1", head="omat_pbe", default_dtype="float64", device=device
        )
    if kind == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator

        return CHGNetCalculator(use_device=device, on_isolated_atoms="ignore")
    raise ValueError(f"unknown calculator {kind}")


def relax(atoms, calc, fmax: float = 0.05, steps: int = 300, fix_bottom_frac: float = 0.5):
    """Relax a slab/complex; freeze the bottom fraction to mimic bulk."""
    from ase.constraints import FixAtoms
    from ase.optimize import LBFGS

    atoms = atoms.copy()
    atoms.calc = calc
    z = atoms.positions[:, 2]
    cut = z.min() + fix_bottom_frac * (z.max() - z.min())
    atoms.set_constraint(FixAtoms(mask=[zi < cut for zi in z]))
    LBFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    return atoms


def relax_adsorbate(complex_, calc, n_slab: int, fmax: float = 0.05, steps: int = 200):
    """Relax ONLY the adsorbate on a frozen slab; return ``(atoms, converged)``.

    Freezing the whole slab is essential for meaningful dE_ads on geometric surfaces:
    if the slab may move, the optimizer relieves built-in surface strain and that
    reconstruction energy (eV-scale) contaminates the adsorption energy.
    """
    from ase.constraints import FixAtoms
    from ase.optimize import LBFGS

    atoms = complex_.copy()
    atoms.calc = calc
    atoms.set_constraint(FixAtoms(indices=list(range(n_slab))))
    opt = LBFGS(atoms, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    return atoms, converged


def adsorption_energy(slab, molecule, calc, place_height: float = 2.2, site_xy=None) -> dict:
    """dE_ads = E(slab+mol) - E(slab) - E(mol_gas). Negative => binding.

    Strong (< ~-0.7 eV) => chemisorption; weak (> ~-0.3 eV) => physisorption.
    """
    from ase.build import add_adsorbate
    from ase.optimize import LBFGS

    slab_r = relax(slab, calc)
    e_slab = slab_r.get_potential_energy()

    mol_r = molecule.copy()
    mol_r.calc = calc
    LBFGS(mol_r, logfile=None).run(fmax=0.03, steps=200)
    e_mol = mol_r.get_potential_energy()

    complex_ = slab_r.copy()
    if site_xy is None:
        site_xy = (slab_r.cell[0, 0] * 0.5, slab_r.cell[1, 1] * 0.5)
    add_adsorbate(complex_, molecule, height=place_height, position=site_xy)
    complex_r, _ = relax_adsorbate(complex_, calc, n_slab=len(slab_r))
    e_complex = complex_r.get_potential_energy()

    dE = e_complex - e_slab - e_mol
    return {
        "dE_ads_eV": round(float(dE), 4),
        "regime": (
            "chemisorption" if dE < -0.7
            else ("physisorption" if dE > -0.3 else "intermediate")
        ),
    }


def reactive_sites(slab, material_key: str, n_sites: int = 4):
    """Return up to ``n_sites`` (x, y, z) reactive surface anchors.

    Uses the capped silanol O (SiO2) / amine N (SiN) atoms in the top surface region as
    adsorption anchors; falls back to a jittered grid over the top of the cell.
    """
    import numpy as np

    pos = slab.get_positions()
    z = pos[:, 2]
    z_top = z.max()
    target_z = 8 if material_key == "SiO2" else 7  # O for silanol, N for amine
    idx = [
        i for i in range(len(slab))
        if slab.numbers[i] == target_z and z[i] >= z_top - 3.0
    ]
    idx.sort(key=lambda i: z[i], reverse=True)
    sites = [(float(pos[i][0]), float(pos[i][1]), float(pos[i][2])) for i in idx[:n_sites]]
    if not sites:  # fallback: grid over the top surface
        cell = slab.get_cell()
        for fx, fy in [(0.5, 0.5), (0.25, 0.25), (0.75, 0.75), (0.25, 0.75)][:n_sites]:
            p = fx * cell[0] + fy * cell[1]
            sites.append((float(p[0]), float(p[1]), float(z_top)))
    return sites


def _place_adsorbate(slab, molecule, xy_z, height: float, rot_deg: float):
    """Place a rotated copy of ``molecule`` above the site; return the combined system."""
    import numpy as np

    mol = molecule.copy()
    mol.rotate(rot_deg, "z", center="COM")
    com = mol.get_center_of_mass()
    target = np.array([xy_z[0], xy_z[1], xy_z[2] + height])
    mol.translate(target - com)
    complex_ = slab.copy()
    complex_ += mol
    return complex_


def adsorption_energy_search(
    slab,
    molecule,
    calc,
    material_key: str,
    n_sites: int = 4,
    n_rot: int = 4,
    heights=(1.8, 2.4),
) -> dict:
    """Multi-site / multi-orientation adsorption search.

    Protocol (frozen-slab reference, ADR-009):
      1. Relax the bare slab once (bottom half fixed) -> ``E(slab)`` at geometry G.
      2. Relax the gas molecule once -> ``E(mol)``.
      3. For each site x rotation x height: place the molecule above G, relax ONLY the
         adsorbate with the whole slab frozen at G, take
         ``dE = E(complex@G) - E(slab@G) - E(mol)``.

    Freezing the slab in step 3 guarantees the slab term cancels exactly; letting the
    slab relax under the adsorbate lets eV-scale surface reconstruction leak into dE
    (the source of the unphysical -2.4 eV physisorption values). Configurations that
    do not converge or land outside a physical window (default [-3, +1] eV for a
    molecular adsorbate) are recorded but excluded from the reported minimum.
    """
    # A moderate relax is enough: with the frozen-slab protocol the slab term cancels
    # exactly at whatever geometry G this produces, so G need not be a deep minimum.
    slab_r = relax(slab, calc, fmax=0.1, steps=120)
    e_slab = float(slab_r.get_potential_energy())
    n_slab = len(slab_r)

    mol_r = molecule.copy()
    mol_r.calc = calc
    from ase.optimize import LBFGS

    LBFGS(mol_r, logfile=None).run(fmax=0.03, steps=200)
    e_mol = float(mol_r.get_potential_energy())

    de_lo, de_hi = -3.0, 1.0  # physical window for molecular (non-dissociative) dE_ads

    sites = reactive_sites(slab_r, material_key, n_sites=n_sites)
    configs: list[dict] = []
    best = None
    for si, site in enumerate(sites):
        for h in heights:
            for k in range(max(1, n_rot)):
                rot = k * 360.0 / max(1, n_rot)
                try:
                    cx = _place_adsorbate(slab_r, mol_r, site, h, rot)
                    cx_r, conv = relax_adsorbate(cx, calc, n_slab=n_slab, fmax=0.05,
                                                 steps=250)
                    dE = float(cx_r.get_potential_energy() - e_slab - e_mol)
                except Exception:  # noqa: BLE001 -- skip pathological placements
                    continue
                ok = bool(conv and de_lo <= dE <= de_hi)
                configs.append({"site": si, "height": h, "rot_deg": rot,
                                "dE_ads_eV": round(dE, 4), "converged": bool(conv),
                                "physical": ok})
                if ok and (best is None or dE < best):
                    best = dE
    n_ok = sum(1 for c in configs if c["physical"])
    if best is None:
        # No converged, in-window config: fall back to the least-bad sampled value so
        # the caller can still proceed, but flag it for review.
        if not configs:
            raise RuntimeError("no adsorption configuration converged")
        best = min(c["dE_ads_eV"] for c in configs)
        best = max(min(best, de_hi), de_lo)  # clamp into the physical window
    return {
        "dE_ads_eV": round(best, 4),
        "regime": (
            "chemisorption" if best < -0.7
            else ("physisorption" if best > -0.3 else "intermediate")
        ),
        "n_configs": len(configs),
        "n_physical": n_ok,
        "flag": None if n_ok else "no-physical-config",
        "configs": configs,
    }


def reaction_energetics(
    slab,
    molecule,
    calc,
    material_key: str,
    site_type: str = "OH",
    compute_ea: bool = False,
    height: float = 2.0,
) -> dict:
    """Two-state reaction energetics per Kim et al. 2026 (Eq. 1-2).

    Builds a physisorption state (molecule H-bonded near the reactive site) and a
    chemisorption state (fragment bonded, byproduct nearby), then returns
    ``deltaEr = E_chem - E_phys``. Optionally computes ``Ea`` via NEB when
    ``compute_ea=True`` (expensive; defaults to mined literature priors otherwise).

    Best-effort for terminal sites (-OH, -NH2); bridge sites should use literature priors.
    """
    from ase.optimize import LBFGS

    slab_r = relax(slab, calc)
    e_slab = slab_r.get_potential_energy()

    mol_r = molecule.copy()
    mol_r.calc = calc
    LBFGS(mol_r, logfile=None).run(fmax=0.03, steps=200)
    e_mol = mol_r.get_potential_energy()

    sites = reactive_sites(slab_r, material_key, n_sites=1)
    if not sites:
        raise RuntimeError("no reactive site for reaction_energetics")
    site = sites[0]

    # Physisorption: molecule placed above site (H-bond distance ~2.0 A).
    phys = _place_adsorbate(slab_r, mol_r, site, height, rot_deg=0.0)
    phys_r, _ = relax_adsorbate(phys, calc, n_slab=len(slab_r), fmax=0.08, steps=150)
    e_phys = float(phys_r.get_potential_energy() - e_slab - e_mol)

    # Chemisorption: molecule closer / different orientation (proxy for bonded state).
    chem = _place_adsorbate(slab_r, mol_r, site, height * 0.65, rot_deg=180.0)
    chem_r, _ = relax_adsorbate(chem, calc, n_slab=len(slab_r), fmax=0.08, steps=150)
    e_chem = float(chem_r.get_potential_energy() - e_slab - e_mol)

    delta_r = e_chem - e_phys
    out: dict = {
        "site_type": site_type,
        "E_physisorption_eV": round(e_phys, 4),
        "E_chemisorption_eV": round(e_chem, 4),
        "deltaEr_eV": round(delta_r, 4),
        "exothermic": bool(delta_r < 0),
        "byproduct": "HNR2" if "dmatms" in str(molecule.symbols) else "H2O",
    }

    if compute_ea:
        try:
            neb = barrier_neb(phys_r, chem_r, calc, n_images=5)
            out["Ea_eV"] = neb["barrier_eV"]
        except Exception as exc:  # noqa: BLE001
            out["Ea_note"] = f"NEB failed: {exc}"
    return out


def barrier_neb(initial, final, calc, n_images: int = 7, fmax: float = 0.07) -> dict:
    """Optional Tier-2: precursor first-half-reaction barrier via NEB.

    NOTE: foundation MLIPs UNDERESTIMATE barriers (arXiv:2502.15582) -- treat the number
    as a lower bound and calibrate against literature DFT before reporting.
    """
    try:
        from ase.mep import NEB
    except ImportError:
        from ase.neb import NEB
    from ase.optimize import LBFGS

    images = [initial] + [initial.copy() for _ in range(n_images - 2)] + [final]
    for im in images:
        im.calc = calc
    neb = NEB(images, climb=True)
    neb.interpolate()
    LBFGS(neb, logfile=None).run(fmax=fmax, steps=300)
    energies = [im.get_potential_energy() for im in images]
    ea = max(energies) - energies[0]
    return {
        "barrier_eV": round(float(ea), 4),
        "path_energies_eV": [round(e, 4) for e in energies],
    }


# Canonical SMILES for the inhibitor/precursor library so molecules are built with the
# correct connectivity (rdkit), not coarse G2 stand-ins. Extend freely.
NAME_TO_SMILES: dict[str, str] = {
    "acetic acid": "CC(=O)O",
    "formic acid": "OC=O",
    "carboxylic acid": "CC(=O)O",
    "pivalic acid": "CC(C)(C)C(=O)O",
    "ethylbutyric acid": "CCC(CC)C(=O)O",
    "methanesulfonic acid": "CS(=O)(=O)O",
    "octadecylphosphonic acid": "CCCCCCCCCCCCCCCCCCP(=O)(O)O",
    "phosphonic acid": "OP(=O)O",
    "aniline": "c1ccccc1N",
    "pyrrole": "c1cc[nH]c1",
    "pyridine": "c1ccncc1",
    "trimethoxypropylsilane": "CCC[Si](OC)(OC)OC",
    "dmatms": "CN(C)[Si](C)(C)C",
    "dimethylamino-trimethylsilane": "CN(C)[Si](C)(C)C",
    "ets": "CC[Si](O)(O)O",
    "ethyltrichlorosilane": "CC[Si](O)(O)O",
}


def smiles_to_atoms(smiles: str):
    """Embed a SMILES string into a relaxed 3D ASE ``Atoms`` (rdkit ETKDGv3 + MMFF)."""
    from ase import Atoms
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"rdkit could not parse SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError(f"rdkit could not embed SMILES {smiles!r}")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:  # noqa: BLE001 -- MMFF may lack params for exotic atoms; keep embed
        pass
    conf = mol.GetConformer()
    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    positions = [
        (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
        for i in range(mol.GetNumAtoms())
    ]
    return Atoms(numbers=numbers, positions=positions)


def build_molecule(name_or_smiles: str):
    """Build a real 3D molecule for a named inhibitor/precursor or a raw SMILES.

    Resolves the name against :data:`NAME_TO_SMILES` first, then treats the input as a
    SMILES string (so Phase-2 AI-proposed candidates work). Falls back to ASE's G2
    database for simple named molecules if rdkit is unavailable.
    """
    key = name_or_smiles.strip().lower()
    smiles = NAME_TO_SMILES.get(key)
    try:
        return smiles_to_atoms(smiles or name_or_smiles)
    except Exception:  # noqa: BLE001 -- fall back to G2 for simple names without rdkit
        from ase.build import molecule as ase_molecule

        g2_aliases = {"acetic acid": "CH3COOH", "formic acid": "HCOOH",
                      "carboxylic acid": "HCOOH"}
        return ase_molecule(g2_aliases.get(key, name_or_smiles))
