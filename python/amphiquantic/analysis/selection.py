"""Atom selection helper over .aqtop-style atom records.

Works either against a loaded topology dict (as returned by
``amphiquantic.topology.io.load_topology``) or against an explicit
``amphiquantic.analysis.trajectory.Trajectory``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass
class Selection:
    """Indices of selected atoms plus a human-readable description."""

    indices: np.ndarray
    description: str = ""

    def __len__(self) -> int:
        return len(self.indices)

    def __array__(self, dtype=None):
        return np.asarray(self.indices, dtype=dtype)

    def __iter__(self):
        return iter(self.indices.tolist())


def _atom_records(topology_or_traj) -> list[dict[str, Any]]:
    if hasattr(topology_or_traj, "topology"):
        topology = topology_or_traj.topology
        if topology is None:
            raise ValueError("Trajectory has no attached topology for selection")
        return topology["atoms"]
    if isinstance(topology_or_traj, dict):
        return topology_or_traj["atoms"]
    raise TypeError("select() expects a topology dict or a Trajectory with .topology")


def _parse_index_ranges(spec: str) -> set[int]:
    """Parse '0,2,5-8' style index ranges into a set of ints."""
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(chunk))
    return out


def select(
    topology_or_traj,
    name: str | Iterable[str] | None = None,
    resname: str | Iterable[str] | None = None,
    resid: int | Iterable[int] | None = None,
    element: str | Iterable[str] | None = None,
    index: str | Iterable[int] | None = None,
) -> Selection:
    """Select atom indices by name / resname / resid / element / index range.

    Each criterion narrows the candidate set (logical AND). Multiple values
    for one criterion behave as an OR within that criterion. ``resname`` is
    matched against a per-atom ``residue_name`` field when present, falling
    back to no-op (never matches) if the topology doesn't carry residue
    names (many bootstrap/.aqtop fixtures only store ``residue_id``).
    """

    atoms = _atom_records(topology_or_traj)
    n = len(atoms)
    mask = np.ones(n, dtype=bool)
    parts: list[str] = []

    def _as_set(v):
        if v is None:
            return None
        if isinstance(v, (str, int)):
            return {v}
        return set(v)

    name_set = _as_set(name)
    if name_set is not None:
        sel = np.array([a.get("name") in name_set for a in atoms])
        mask &= sel
        parts.append(f"name {sorted(map(str, name_set))}")

    resname_set = _as_set(resname)
    if resname_set is not None:
        sel = np.array([a.get("residue_name") in resname_set for a in atoms])
        mask &= sel
        parts.append(f"resname {sorted(map(str, resname_set))}")

    resid_set = _as_set(resid)
    if resid_set is not None:
        sel = np.array([a.get("residue_id") in resid_set for a in atoms])
        mask &= sel
        parts.append(f"resid {sorted(resid_set)}")

    element_set = _as_set(element)
    if element_set is not None:
        sel = np.array([a.get("element") in element_set for a in atoms])
        mask &= sel
        parts.append(f"element {sorted(map(str, element_set))}")

    if index is not None:
        if isinstance(index, str):
            idx_set = _parse_index_ranges(index)
        else:
            idx_set = set(index)
        sel = np.array([i in idx_set for i in range(n)])
        mask &= sel
        parts.append(f"index {sorted(idx_set)}")

    indices = np.nonzero(mask)[0]
    return Selection(indices=indices, description=" and ".join(parts) or "all")
