"""Physical property helpers for validation."""

from __future__ import annotations

from typing import Any

AMU_TO_G = 1.66053906660e-24
NM3_TO_CM3 = 1e-21


def _box_dim(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def density_g_cm3(topology: dict[str, Any]) -> float:
    """Mass density from atom masses and simulation box volume."""
    box = topology.get("box", {})
    lx = _box_dim(box.get("lx", 0))
    ly = _box_dim(box.get("ly", 0))
    lz = _box_dim(box.get("lz", 0))
    if lx <= 0 or ly <= 0 or lz <= 0:
        raise ValueError("Box dimensions must be positive for density")
    volume_cm3 = lx * ly * lz * NM3_TO_CM3
    mass_g = sum(float(a.get("mass", 0)) for a in topology["atoms"]) * AMU_TO_G
    return mass_g / volume_cm3
