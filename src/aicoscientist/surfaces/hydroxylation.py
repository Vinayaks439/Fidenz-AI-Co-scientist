"""Surface saturation for procedural slabs (Phase 1 + Kim et al. 2026 Table-1).

Implements the paper's passivation scheme (Table 1) for dangling bonds after cleavage,
then equilibrates the termination with chemically real moves -- silanol/amine
condensation (2 terminals -> bridge + volatile byproduct), bridge hydrolysis/ammonolysis
(bridge -> terminals), and imide H-desorption -- until both the terminal and bridge site
densities sit inside the Kim et al. 2026 experimental bands. Every move creates or breaks
real bonds under hard distance constraints, so the fidelity gate counts real sites.
"""

from __future__ import annotations

import numpy as np

# Approximate bond lengths (Angstrom).
_SI_O = 1.63
_O_H = 0.96
_SI_N = 1.74
_N_H = 1.02
_SI_H = 1.48

# Placement exclusion radii for a new terminal heavy atom (O/N):
#   to any OTHER Si: > Si-O/Si-N covalent cutoff (~2.15 A at natural_cutoffs mult=1.2),
#     so the group bonds exactly one Si;
#   to other O/N:    > steric contact (non-bonded O...O in silica is ~2.6 A, but capping
#     groups may sit closer; the covalent O-O cutoff is only ~1.6 A).
_NO_BOND_SI = 2.25
_NO_BOND_ON = 1.9

# Kim et al. 2026 target densities (sites/nm^2) for amorphous PECVD surfaces.
TARGET_DENSITIES: dict[str, dict[str, float]] = {
    "SiO2": {"OH": 6.19, "O_bridge": 3.86},
    "SiN": {"NH2": 3.91, "NH_bridge": 3.53},
}


# ──────────────────────── geometry helpers ────────────────────────

def _coordination(atoms, cutoff: float = 2.0) -> list[int]:
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, 99.0)
    return [int((d[i] < cutoff).sum()) for i in range(len(atoms))]


def _n_neighbors_of(atoms, i: int, znum: int, cutoff: float = 2.0) -> int:
    d = atoms.get_distances(i, list(range(len(atoms))), mic=True)
    return int(sum(1 for j in range(len(atoms))
                   if j != i and d[j] < cutoff and atoms.numbers[j] == znum))


def _missing_direction(atoms, i: int, cutoff: float = 2.0) -> np.ndarray:
    pos = atoms.get_positions()
    d = atoms.get_distances(i, list(range(len(atoms))), mic=True)
    vecs = []
    for j in range(len(atoms)):
        if j == i or d[j] >= cutoff:
            continue
        v = pos[j] - pos[i]
        n = np.linalg.norm(v)
        if n > 1e-6:
            vecs.append(v / n)
    if not vecs:
        return np.array([0.0, 0.0, 1.0])
    direction = -np.sum(vecs, axis=0)
    n = np.linalg.norm(direction)
    if n < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return direction / n


def _perp(direction: np.ndarray) -> np.ndarray:
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(direction, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    p = ref - np.dot(ref, direction) * direction
    return p / (np.linalg.norm(p) + 1e-9)


def _top_surface_mask(atoms, coord_cutoff: float = 2.0) -> tuple[np.ndarray, float]:
    z = atoms.get_positions()[:, 2]
    return z >= 0.5 * (z.min() + z.max()), float(z.max())


def _bond_graph(atoms):
    """Fresh covalent-cutoff neighbour list (matches the fidelity gate's counting)."""
    from ase.neighborlist import NeighborList, natural_cutoffs

    nl = NeighborList(natural_cutoffs(atoms, mult=1.2),
                      self_interaction=False, bothways=True)
    nl.update(atoms)
    return nl


def _mic_dists(atoms, p: np.ndarray) -> np.ndarray:
    """Distances from point ``p`` to every atom, minimum-image in the a/b plane."""
    cell = atoms.get_cell()
    delta = atoms.get_positions() - p
    for k in range(2):  # slab: periodic in a/b only
        v = np.array(cell[k])
        L = np.linalg.norm(v)
        if L > 1e-6:
            u = v / L
            proj = delta @ u
            delta -= np.outer(np.round(proj / L) * L, u)
    return np.linalg.norm(delta, axis=1)


# ──────────────────────── verified group placement ────────────────────────

def _place_group(atoms, anchor: int, material_key: str, rng, max_tries: int = 40) -> bool:
    """Bond a fresh terminal group (OH / NH2) to ``anchor`` Si under hard constraints.

    The new heavy atom must sit at bond length from its anchor, beyond the covalent
    cutoff of every *other* Si (so it bonds exactly one Si), sterically clear of other
    O/N/H, and every new H must be clash-free. Directions are sampled upward-biased and
    away from the anchor's existing bonds. Returns False when the anchor valence is
    genuinely full or no clear direction exists.
    """
    from ase import Atom

    is_oxide = material_key == "SiO2"
    bond = _SI_O if is_oxide else _SI_N
    nums = atoms.numbers
    pos = atoms.get_positions()
    base = pos[anchor]

    # Anchor valence = bonded O/N/H (Si-Si contacts inside the covalent cutoff are NOT
    # bonds in these networks and must not count).
    nl = _bond_graph(atoms)
    existing = []
    n_bonds = 0
    for j in nl.get_neighbors(anchor)[0]:
        if nums[j] not in (1, 7, 8):
            continue
        n_bonds += 1
        v = pos[j] - base
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            existing.append(v / n)
    if n_bonds >= 4:
        return False

    idx = np.arange(len(atoms))
    si_mask = (nums == 14) & (idx != anchor)
    on_mask = np.isin(nums, (7, 8))
    h_mask = nums == 1

    for _ in range(max_tries):
        v = rng.normal(size=3)
        v[2] = abs(v[2]) + 0.4
        v /= np.linalg.norm(v)
        if existing and min(-float(np.dot(v, e)) for e in existing) < -0.30:
            continue  # < ~72 deg from an existing bond
        heavy_p = base + bond * v
        d = _mic_dists(atoms, heavy_p)
        d_si = float(np.min(d[si_mask])) if np.any(si_mask) else 9.0
        d_on = float(np.min(d[on_mask])) if np.any(on_mask) else 9.0
        d_h = float(np.min(d[h_mask])) if np.any(h_mask) else 9.0
        if d_si < _NO_BOND_SI or d_on < _NO_BOND_ON or d_h < 1.15:
            continue
        tilt = _perp(v)
        if is_oxide:
            h1 = heavy_p + _O_H * (0.55 * v + 0.83 * tilt)
            if float(np.min(_mic_dists(atoms, h1))) < 1.1:
                continue
            atoms.append(Atom("O", heavy_p))
            atoms.append(Atom("H", h1))
        else:
            h1 = heavy_p + _N_H * (0.55 * v + 0.83 * tilt)
            h2 = heavy_p + _N_H * (0.55 * v - 0.83 * tilt)
            if float(np.min(_mic_dists(atoms, h1))) < 1.1:
                continue
            if float(np.min(_mic_dists(atoms, h2))) < 1.1:
                continue
            atoms.append(Atom("N", heavy_p))
            atoms.append(Atom("H", h1))
            atoms.append(Atom("H", h2))
        return True
    return False


# ──────────────────────── Table-1 passivation ────────────────────────

def table1_passivate(atoms, material_key: str, coord_cutoff: float = 2.0) -> object:
    """Apply Kim et al. 2026 Table-1 passivation to top-surface dangling bonds.

    The paper's X^n+ notation counts *dangling bonds* (missing valence), so the mapping
    is driven by ``dangling = valence - coordination``:

      SiO2: Si^1+ -> SiOH; Si^2+ -> Si(OH)H; Si^3+ -> Si(OH)2H; O^1+ -> OH.
      SiN:  Si^1+ -> SiNH2; Si^2+ -> Si=NH; Si^3+ -> Si(NH)H;
            N^1+ -> NH (imide Si-NH-Si); N^2+ -> NH2; O^1+ -> OH.

    Every cap is placed under the same hard distance constraints as the equilibration
    moves, so passivation cannot over-coordinate an anchor or create atomic overlaps.
    """
    from ase import Atom

    atoms = atoms.copy()
    rng = np.random.default_rng(0)
    coord = _coordination(atoms, coord_cutoff)
    top, _ = _top_surface_mask(atoms, coord_cutoff)
    nums = atoms.numbers.copy()
    Si, O, N = 14, 8, 7
    valence = {Si: 4, O: 2, N: 3}

    def add_h_checked(anchor_idx, base, direction, bond_len):
        h_pos = base + bond_len * direction
        d = _mic_dists(atoms, h_pos)
        d[anchor_idx] = np.inf  # the anchor is the atom the H bonds to
        if float(np.min(d)) >= 1.0:
            atoms.append(Atom("H", h_pos))
            return True
        return False

    n_orig = len(nums)
    for i in range(n_orig):
        if not top[i]:
            continue
        znum = int(nums[i])
        if znum not in valence:
            continue
        dangling = valence[znum] - coord[i]
        if dangling <= 0:
            continue
        base = atoms.get_positions()[i]
        d = _missing_direction(atoms, i, coord_cutoff)
        if d[2] < 0:
            d = np.array([d[0], d[1], abs(d[2])])
        d /= (np.linalg.norm(d) + 1e-9)

        if znum == Si:
            # Terminal group(s) for the first dangling bond(s), H for the remainder.
            n_groups = 2 if dangling >= 3 and material_key == "SiO2" else 1
            for _ in range(n_groups):
                _place_group(atoms, i, material_key, rng)
            if dangling - n_groups >= 1:
                add_h_checked(i, base, d, _SI_H)
        elif znum == N and material_key == "SiN":
            for k in range(min(dangling, 2)):  # N^1+ -> NH (imide); N^2+ -> NH2
                tilt = _perp(d)
                direction = d + (0.6 if k == 0 else -0.6) * tilt
                direction /= np.linalg.norm(direction)
                add_h_checked(i, base, direction, _N_H)
        elif znum == O and _n_neighbors_of(atoms, i, Si, coord_cutoff) == 1:
            add_h_checked(i, base, d, _O_H)
    return atoms


# ──────────────────────── site inventory ────────────────────────

def _surface_sites(atoms, material_key: str, surface_depth: float = 2.5):
    """Return (terminal_sites, bridge_sites) index lists matching the gate's counting.

    terminal: SiO2 -> silanol O (1 Si + H); SiN -> amine N (1 Si + H), top half.
    bridge:   SiO2 -> siloxane O (2 Si, no H); SiN -> imide N (2 Si + H), within
              ``surface_depth`` of the topmost heavy atom.
    """
    H, Si = 1, 14
    heavy = 8 if material_key == "SiO2" else 7
    nums = atoms.numbers
    z = atoms.get_positions()[:, 2]
    heavy_z = z[nums != H]
    z_top = float(heavy_z.max()) if len(heavy_z) else float(z.max())
    z_bridge_cut = z_top - surface_depth
    z_terminal_cut = 0.5 * (float(z.min()) + float(z.max()))

    nl = _bond_graph(atoms)
    terminals, bridges = [], []
    for i in range(len(atoms)):
        if nums[i] != heavy or z[i] < z_terminal_cut:
            continue
        nbrs = nl.get_neighbors(i)[0]
        types = [nums[j] for j in nbrs]
        n_si = types.count(Si)
        has_h = H in types
        if material_key == "SiO2":
            if has_h and n_si == 1:
                terminals.append(i)
            elif not has_h and n_si == 2 and z[i] >= z_bridge_cut:
                bridges.append(i)
        else:
            if has_h and n_si == 1:
                terminals.append(i)
            elif has_h and n_si == 2 and z[i] >= z_bridge_cut:
                bridges.append(i)
    return terminals, bridges


def _h_of(atoms, i: int) -> list[int]:
    nl = _bond_graph(atoms)
    return [j for j in nl.get_neighbors(i)[0] if atoms.numbers[j] == 1]


def _si_of(atoms, i: int) -> list[int]:
    nl = _bond_graph(atoms)
    return [j for j in nl.get_neighbors(i)[0] if atoms.numbers[j] == 14]


# ──────────────────────── equilibration moves ────────────────────────

def condense_pair(atoms, material_key: str, rng) -> bool:
    """2 SiOH -> Si-O-Si + H2O (or 2 SiNH2 -> Si-NH-Si + NH3): form one real bridge.

    Picks two terminal groups whose anchor Si atoms are close enough that the surviving
    heavy atom, placed between them, genuinely bonds both; deletes the other group and
    the excess H (the volatile byproduct leaves).
    """
    terminals, _ = _surface_sites(atoms, material_key)
    if len(terminals) < 2:
        return False
    bond = _SI_O if material_key == "SiO2" else _SI_N
    pos = atoms.get_positions()
    pairs = []
    for a_idx, t1 in enumerate(terminals):
        si1 = _si_of(atoms, t1)
        if not si1:
            continue
        for t2 in terminals[a_idx + 1:]:
            si2 = _si_of(atoms, t2)
            if not si2 or si2[0] == si1[0]:
                continue
            d_si = float(np.linalg.norm(pos[si1[0]] - pos[si2[0]]))
            if d_si < 2.0 * bond + 0.4:  # bridge geometry reachable
                pairs.append((d_si, t1, t2, si1[0], si2[0]))
    if not pairs:
        return False
    pairs.sort()
    _, t1, t2, s1, s2 = pairs[int(rng.integers(0, min(3, len(pairs))))]

    d_si = float(np.linalg.norm(pos[s1] - pos[s2]))
    lift = float(np.sqrt(max(bond * bond - (d_si / 2.0) ** 2, 0.09)))
    mid = 0.5 * (pos[s1] + pos[s2]) + np.array([0.0, 0.0, lift])

    keep_h = 1 if material_key == "SiN" else 0  # imide keeps one N-H
    h1 = _h_of(atoms, t1)
    h2 = _h_of(atoms, t2)
    atoms.positions[t1] = mid
    if keep_h and h1:
        atoms.positions[h1[0]] = mid + np.array([0.0, 0.0, _N_H])
    to_del = sorted({t2, *h2, *h1[keep_h:]}, reverse=True)
    for j in to_del:
        del atoms[j]
    return True


def hydrolyze_bridge(atoms, material_key: str, rng, max_tries: int = 60) -> bool:
    """Open one bridge into a terminal group (hydrolysis / ammonolysis).

    Implemented as a *pivot*: the bridge heavy atom detaches from one anchor Si, swings
    to a verified clash-free position bonded only to the kept Si, and gains the H that
    turns it into a silanol / amine (the imide N already carries one H). The detached Si
    is capped with an uncounted Si-H. Because no new heavy atom has to fit into the
    congested network, this succeeds where fresh-group placement cannot.
    """
    from ase import Atom

    _, bridges = _surface_sites(atoms, material_key)
    if not bridges:
        return False
    bond = _SI_O if material_key == "SiO2" else _SI_N
    nums = atoms.numbers
    idx = np.arange(len(atoms))

    order = list(rng.permutation(len(bridges)))
    for pick in order:
        b = int(bridges[pick])
        sis = _si_of(atoms, b)
        if len(sis) < 2:
            continue
        pos = atoms.get_positions()
        # Detach from the lower Si; the terminal group should point into the vacuum.
        s_keep, s_det = ((sis[0], sis[1]) if pos[sis[0]][2] >= pos[sis[1]][2]
                         else (sis[1], sis[0]))
        base = pos[s_keep]

        nl = _bond_graph(atoms)
        existing = []
        for j in nl.get_neighbors(s_keep)[0]:
            if j == b or nums[j] == 1:
                continue
            v = pos[j] - base
            n = float(np.linalg.norm(v))
            if n > 1e-6:
                existing.append(v / n)

        si_mask = (nums == 14) & (idx != s_keep)
        on_mask = np.isin(nums, (7, 8)) & (idx != b)
        h_of_b = _h_of(atoms, b)
        h_mask = (nums == 1) & (~np.isin(idx, h_of_b))

        for _ in range(max_tries):
            v = rng.normal(size=3)
            v[2] = abs(v[2]) + 0.5
            v /= np.linalg.norm(v)
            if existing and min(-float(np.dot(v, e)) for e in existing) < -0.30:
                continue
            new_p = base + bond * v
            d = _mic_dists(atoms, new_p)
            d_si = float(np.min(d[si_mask])) if np.any(si_mask) else 9.0
            d_on = float(np.min(d[on_mask])) if np.any(on_mask) else 9.0
            d_h = float(np.min(d[h_mask])) if np.any(h_mask) else 9.0
            if d_si < _NO_BOND_SI or d_on < _NO_BOND_ON or d_h < 1.15:
                continue

            tilt = _perp(v)
            atoms.positions[b] = new_p
            # Re-seat the existing H (imide keeps its H; siloxane O has none yet).
            for k, hj in enumerate(h_of_b):
                atoms.positions[hj] = new_p + (_N_H if material_key == "SiN" else _O_H) * (
                    0.55 * v + (0.83 if k == 0 else -0.83) * tilt)
            # Add the H that completes the terminal group.
            bond_h = _O_H if material_key == "SiO2" else _N_H
            sign = 0.83 if not h_of_b else -0.83
            h_new = new_p + bond_h * (0.55 * v + sign * tilt)
            if float(np.min(_mic_dists(atoms, h_new)[idx != b])) >= 1.0:
                atoms.append(Atom("H", h_new))
            # Cap the abandoned Si valence with an uncounted Si-H.
            v_det = (pos[b] - pos[s_det])
            n_det = float(np.linalg.norm(v_det))
            if n_det > 1e-6:
                h_cap = pos[s_det] + _SI_H * (v_det / n_det)
                if float(np.min(_mic_dists(atoms, h_cap))) >= 1.0:
                    atoms.append(Atom("H", h_cap))
            return True
    return False


def remove_bridge(atoms, material_key: str, rng) -> bool:
    """Dissolve one counted surface bridge: delete its heavy atom (+H) and cap both
    anchor Si with uncounted Si-H (the geometric analogue of the anneal re-incorporating
    a strained surface bridge into the network)."""
    from ase import Atom

    _, bridges = _surface_sites(atoms, material_key)
    if not bridges:
        return False
    b = int(rng.choice(bridges))
    sis = _si_of(atoms, b)
    hs = _h_of(atoms, b)
    pos = atoms.get_positions()
    caps = []
    for s in sis:
        v = pos[b] - pos[s]
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            caps.append(pos[s] + _SI_H * (v / n))
    for j in sorted({b, *hs}, reverse=True):
        del atoms[j]
    for c in caps:
        if float(np.min(_mic_dists(atoms, c))) >= 1.0:
            atoms.append(Atom("H", c))
    return True


def carve_terminal(atoms, material_key: str, rng, max_tries: int = 60) -> bool:
    """Create a terminal group out of the network when fresh placement is impossible.

    SiO2: pick an *uncounted* sub-surface siloxane O (2 Si) near the top, pivot it onto
    its upper Si and protonate -> silanol (hydrolysis of a network Si-O-Si).
    SiN:  pick a surface network N (3 Si), detach it from two Si (each capped Si-H),
    pivot onto the kept Si and add 2 H -> amine (ammonolysis of network Si-N bonds).
    """
    from ase import Atom

    H, Si = 1, 14
    heavy = 8 if material_key == "SiO2" else 7
    want_si = 2 if material_key == "SiO2" else 3
    bond = _SI_O if material_key == "SiO2" else _SI_N
    bond_h = _O_H if material_key == "SiO2" else _N_H
    nums = atoms.numbers
    z = atoms.get_positions()[:, 2]
    heavy_z = z[nums != H]
    z_top = float(heavy_z.max()) if len(heavy_z) else float(z.max())

    nl = _bond_graph(atoms)
    candidates = []
    for i in range(len(atoms)):
        if nums[i] != heavy or z[i] < z_top - 5.0:
            continue
        nbrs = nl.get_neighbors(i)[0]
        types = [nums[j] for j in nbrs]
        if types.count(Si) == want_si and H not in types:
            candidates.append(i)
    if not candidates:
        return False
    idx = np.arange(len(atoms))

    for pick in rng.permutation(len(candidates)):
        i = int(candidates[pick])
        pos = atoms.get_positions()
        sis = sorted(_si_of(atoms, i), key=lambda s: -pos[s][2])
        if len(sis) < want_si:
            continue
        s_keep, detached = sis[0], sis[1:]
        base = pos[s_keep]

        existing = []
        for j in nl.get_neighbors(s_keep)[0]:
            if j == i or nums[j] == 1:
                continue
            v = pos[j] - base
            n = float(np.linalg.norm(v))
            if n > 1e-6:
                existing.append(v / n)

        si_mask = (nums == 14) & (idx != s_keep)
        on_mask = np.isin(nums, (7, 8)) & (idx != i)
        h_mask = nums == 1

        for _ in range(max_tries):
            v = rng.normal(size=3)
            v[2] = abs(v[2]) + 0.5
            v /= np.linalg.norm(v)
            if existing and min(-float(np.dot(v, e)) for e in existing) < -0.30:
                continue
            new_p = base + bond * v
            d = _mic_dists(atoms, new_p)
            d_si = float(np.min(d[si_mask])) if np.any(si_mask) else 9.0
            d_on = float(np.min(d[on_mask])) if np.any(on_mask) else 9.0
            d_h = float(np.min(d[h_mask])) if np.any(h_mask) else 9.0
            if d_si < _NO_BOND_SI or d_on < _NO_BOND_ON or d_h < 1.15:
                continue
            tilt = _perp(v)
            atoms.positions[i] = new_p
            n_h = 1 if material_key == "SiO2" else 2
            for k in range(n_h):
                h_p = new_p + bond_h * (0.55 * v + (0.83 if k == 0 else -0.83) * tilt)
                mask = np.arange(len(atoms)) != i
                if float(np.min(_mic_dists(atoms, h_p)[mask])) >= 1.0:
                    atoms.append(Atom("H", h_p))
            for s_d in detached:  # cap freed valences with uncounted Si-H
                v_d = pos[i] - pos[s_d]
                n_d = float(np.linalg.norm(v_d))
                if n_d > 1e-6:
                    h_cap = pos[s_d] + _SI_H * (v_d / n_d)
                    if float(np.min(_mic_dists(atoms, h_cap))) >= 1.0:
                        atoms.append(Atom("H", h_cap))
            return True
    return False


def desorb_imide_h(atoms, material_key: str, rng) -> bool:
    """Imide H desorption: Si-NH-Si -> bare network Si-N-Si (SiN excess-bridge move)."""
    if material_key != "SiN":
        return False
    _, bridges = _surface_sites(atoms, material_key)
    if not bridges:
        return False
    b = int(rng.choice(bridges))
    hs = _h_of(atoms, b)
    if not hs:
        return False
    for j in sorted(hs, reverse=True):
        del atoms[j]
    return True


def add_terminal(atoms, material_key: str, rng) -> bool:
    """Add one silanol / amine group on a top-surface Si with free valence."""
    nums = atoms.numbers
    z = atoms.get_positions()[:, 2]
    zmax = float(z.max())
    nl = _bond_graph(atoms)

    def bond_load(i):  # bonded O/N/H only; Si-Si contacts are not bonds
        return sum(1 for j in nl.get_neighbors(i)[0] if nums[j] in (1, 7, 8))

    si = [i for i in range(len(atoms))
          if nums[i] == 14 and z[i] >= zmax - 5.0 and bond_load(i) < 4]
    if not si:
        return False
    si.sort(key=lambda i: (bond_load(i), -z[i]))
    for i in si:
        if _place_group(atoms, i, material_key, rng):
            return True
    return False


def remove_terminal(atoms, material_key: str, rng) -> bool:
    """Remove one terminal group (heavy atom + its H's); the anchor Si keeps a H cap."""
    from ase import Atom

    terminals, _ = _surface_sites(atoms, material_key)
    if not terminals:
        return False
    pos = atoms.get_positions()
    t = max(terminals, key=lambda i: pos[i][2])  # peel from the top down
    hs = _h_of(atoms, t)
    sis = _si_of(atoms, t)
    direction = None
    anchor_p = None
    if sis:
        anchor_p = pos[sis[0]].copy()
        v = pos[t] - pos[sis[0]]
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            direction = v / n
    for j in sorted({t, *hs}, reverse=True):
        del atoms[j]
    if anchor_p is not None and direction is not None:
        h_pos = anchor_p + _SI_H * direction  # cap the freed valence (uncounted Si-H)
        if float(np.min(_mic_dists(atoms, h_pos))) >= 1.0:
            atoms.append(Atom("H", h_pos))
    return True


def _declash(atoms, min_dist: float = 0.8, max_passes: int = 30) -> None:
    """Separate genuine atomic overlaps (< ``min_dist``) without disturbing real bonds."""
    for _ in range(max_passes):
        pos = atoms.get_positions()
        d = atoms.get_all_distances(mic=True)
        np.fill_diagonal(d, 1e9)
        if float(d.min()) >= min_dist:
            break
        moved = np.zeros_like(pos)
        for i in range(len(atoms)):
            for j in np.where(d[i] < min_dist)[0]:
                v = pos[i] - pos[j]
                n = float(np.linalg.norm(v))
                if n < 1e-6:
                    v = np.random.default_rng(i * 131 + j).normal(size=3)
                    n = float(np.linalg.norm(v))
                moved[i] += 0.6 * (min_dist - d[i, j]) * v / n
        atoms.set_positions(pos + moved)


# ──────────────────────── the saturation entry point ────────────────────────

def saturate_surface(
    atoms,
    material_key: str,
    target_density: float | None = None,
    seed: int = 0,
    coord_cutoff: float = 2.0,
) -> tuple[object, dict]:
    """Table-1 passivation + site-density equilibration toward Kim et al. 2026 bands.

    After passivation, chemically real moves (condensation, hydrolysis/ammonolysis,
    terminal addition/removal, imide H-desorption) drive the terminal AND bridge site
    densities into the experimental acceptance bands. Returns ``(atoms, info)`` with
    achieved per-site-type densities and the move log in provenance.
    """
    from .fidelity_gate import SITE_TYPE_BANDS, SurfaceFidelityGate

    rng = np.random.default_rng(seed)
    targets = TARGET_DENSITIES.get(material_key, {})
    terminal_key = "OH" if material_key == "SiO2" else "NH2"
    bridge_key = "O_bridge" if material_key == "SiO2" else "NH_bridge"
    if target_density is not None:
        targets = {**targets, terminal_key: target_density}

    atoms = table1_passivate(atoms, material_key, coord_cutoff)

    cell = atoms.get_cell()
    area_nm2 = float(np.linalg.norm(np.cross(cell[0], cell[1]))) / 100.0
    bands = SITE_TYPE_BANDS.get(material_key, {})
    t_lo, t_hi = bands.get(terminal_key, (0.0, 1e9))
    b_lo, b_hi = bands.get(bridge_key, (0.0, 1e9))

    moves: list[str] = []
    stall = 0
    for _ in range(120):
        terminals, bridges = _surface_sites(atoms, material_key)
        dt = len(terminals) / area_nm2 if area_nm2 else 0.0
        db = len(bridges) / area_nm2 if area_nm2 else 0.0
        t_ok = t_lo <= dt <= t_hi
        b_ok = b_lo <= db <= b_hi
        if t_ok and b_ok:
            break
        did = False
        if db > b_hi:
            # Too many bridges: hydrolyse into terminals if those are needed/allowed,
            # otherwise dissolve the bridge without creating a terminal.
            one_more_t_ok = (len(terminals) + 2) / area_nm2 <= t_hi
            if not one_more_t_ok:
                if material_key == "SiN":
                    did = desorb_imide_h(atoms, material_key, rng)
                    moves.append("desorbNH")
                if not did:
                    did = remove_bridge(atoms, material_key, rng)
                    moves.append("bridge-")
            else:
                did = hydrolyze_bridge(atoms, material_key, rng)
                moves.append("hydrolyze")
                if not did:
                    did = remove_bridge(atoms, material_key, rng)
                    moves.append("bridge-")
        elif db < b_lo:
            # Too few bridges: condense terminal pairs (if terminals can spare two),
            # else add terminals first so a later condensation is possible.
            if (len(terminals) - 2) / area_nm2 >= t_lo or dt > t_hi:
                did = condense_pair(atoms, material_key, rng)
                moves.append("condense")
            if not did:
                did = add_terminal(atoms, material_key, rng)
                moves.append("terminal+")
        elif dt < t_lo:
            did = add_terminal(atoms, material_key, rng)
            moves.append("terminal+")
            if not did and (len(bridges) - 1) / area_nm2 >= b_lo:
                did = hydrolyze_bridge(atoms, material_key, rng)
                moves.append("hydrolyze")
            if not did:
                did = carve_terminal(atoms, material_key, rng)
                moves.append("carve+")
        elif dt > t_hi:
            did = remove_terminal(atoms, material_key, rng)
            moves.append("terminal-")
        stall = 0 if did else stall + 1
        if stall >= 3:
            break

    _declash(atoms)  # clear any residual overlaps before the descriptor gate

    gate = SurfaceFidelityGate(material_key)
    n_terminal, area_nm2, site_densities = gate.count_sites(atoms)
    _, bridges = _surface_sites(atoms, material_key)

    info = {
        "material": material_key,
        "area_nm2": round(area_nm2, 3),
        "n_sites": int(n_terminal),
        "site_density_per_nm2": round(n_terminal / area_nm2, 3) if area_nm2 else 0.0,
        "site_densities": site_densities,
        "n_bridges_final": len(bridges),
        "equilibration_moves": moves,
        "target_densities": targets,
        "passivation": "table1_kim2026+band_equilibration",
    }
    return atoms, info
