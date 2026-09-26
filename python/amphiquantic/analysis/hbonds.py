"""Geometric hydrogen-bond detection (D-H...A) and occupancy statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HBond:
    donor: int
    hydrogen: int
    acceptor: int


def _min_image(diff: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    if box is None:
        return diff
    return diff - box * np.round(diff / box)


def hydrogen_bonds(
    positions: np.ndarray,
    donors: np.ndarray,
    hydrogens: np.ndarray,
    acceptors: np.ndarray,
    donor_hydrogen_map: dict[int, int] | None = None,
    distance_cutoff: float = 0.35,
    angle_cutoff: float = 30.0,
    box: np.ndarray | None = None,
) -> list[list[HBond]]:
    """Detect hydrogen bonds per frame using D-H...A distance + angle criteria.

    A hydrogen bond D-H...A is counted when:

    * ``|H - A|`` (or ``|D - A|`` if ``distance_cutoff`` semantics call for
      donor-acceptor) is below ``distance_cutoff`` (nm), and
    * the D-H...A angle deviates from linear (180 deg) by less than
      ``angle_cutoff`` degrees.

    Parameters
    ----------
    positions:
        ``(n_frames, n_atoms, 3)``.
    donors, hydrogens, acceptors:
        Atom index arrays. ``donors`` and ``hydrogens`` must correspond
        1:1 in order (donor[i] is bonded to hydrogen[i]), unless
        ``donor_hydrogen_map`` is given as ``{donor_index: hydrogen_index}``.
    distance_cutoff:
        Max H...A distance in nm (default 0.35 nm = 3.5 A, standard cutoff).
    angle_cutoff:
        Max deviation from linear D-H...A angle, in degrees (default 30).
    box:
        Optional ``(3,)`` orthorhombic box for minimum-image distances.

    Returns
    -------
    A list (length n_frames) of lists of :class:`HBond`.
    """
    positions = np.asarray(positions, dtype=float)
    donors = np.asarray(donors)
    hydrogens = np.asarray(hydrogens)
    acceptors = np.asarray(acceptors)

    if donor_hydrogen_map is not None:
        pairs = list(donor_hydrogen_map.items())
    else:
        if len(donors) != len(hydrogens):
            raise ValueError("donors and hydrogens must be same length, or pass donor_hydrogen_map")
        pairs = list(zip(donors.tolist(), hydrogens.tolist()))

    n_frames = positions.shape[0]
    results: list[list[HBond]] = []

    for f in range(n_frames):
        frame_pos = positions[f]
        frame_bonds: list[HBond] = []
        for d_idx, h_idx in pairs:
            h_pos = frame_pos[h_idx]
            d_pos = frame_pos[d_idx]
            for a_idx in acceptors:
                if a_idx == d_idx:
                    continue
                a_pos = frame_pos[a_idx]
                ha_vec = _min_image(a_pos - h_pos, box)
                dist = np.linalg.norm(ha_vec)
                if dist > distance_cutoff:
                    continue
                dh_vec = _min_image(h_pos - d_pos, box)
                # angle at H between H->D and H->A; linear D-H...A means
                # this angle is ~180 deg.
                cos_theta = np.dot(-dh_vec, ha_vec) / (np.linalg.norm(dh_vec) * np.linalg.norm(ha_vec))
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.degrees(np.arccos(cos_theta))
                if abs(180.0 - theta) <= angle_cutoff:
                    frame_bonds.append(HBond(donor=int(d_idx), hydrogen=int(h_idx), acceptor=int(a_idx)))
        results.append(frame_bonds)

    return results


def hbond_occupancy(hbonds_per_frame: list[list[HBond]]) -> dict[tuple[int, int, int], float]:
    """Fractional occupancy (0..1) of each distinct (donor, hydrogen, acceptor) triple."""
    n_frames = len(hbonds_per_frame)
    if n_frames == 0:
        return {}
    counts: dict[tuple[int, int, int], int] = {}
    for frame_bonds in hbonds_per_frame:
        seen = set()
        for hb in frame_bonds:
            key = (hb.donor, hb.hydrogen, hb.acceptor)
            if key in seen:
                continue
            seen.add(key)
            counts[key] = counts.get(key, 0) + 1
    return {k: v / n_frames for k, v in counts.items()}
