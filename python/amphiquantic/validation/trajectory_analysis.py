"""Parse .aqtrj trajectories and compute MD observables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrajFrame:
    step: int
    natoms: int
    epot: float
    ekin: float
    box_lx: float
    box_ly: float
    box_lz: float
    pbc: bool
    positions: list[tuple[float, float, float]]


def parse_aqtrj(path: str | Path) -> list[TrajFrame]:
    """Parse amphiquantic `.aqtrj` v1 format."""
    text = Path(path).read_text()
    frames: list[TrajFrame] = []
    current: TrajFrame | None = None
    pos: list[tuple[float, float, float]] = []

    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("FRAME "):
            if current is not None:
                current.positions = pos
                frames.append(current)
            parts = line.split()
            current = TrajFrame(
                step=int(parts[1]),
                natoms=int(parts[2]),
                epot=float(parts[3]),
                ekin=float(parts[4]),
                box_lx=float(parts[5]),
                box_ly=float(parts[6]),
                box_lz=float(parts[7]),
                pbc=parts[8] != "0",
                positions=[],
            )
            pos = []
            continue
        if current is None:
            continue
        vals = line.split()
        if len(vals) >= 3:
            pos.append((float(vals[0]), float(vals[1]), float(vals[2])))

    if current is not None:
        current.positions = pos
        frames.append(current)
    return frames


def mean_temperature_from_traj(frames: list[TrajFrame], natoms: int) -> float:
    """Average instantaneous temperature (K) from kinetic energy frames."""
    if not frames or natoms <= 0:
        return 0.0
    k_b = 0.008314462618
    dof = max(3 * natoms, 1)
    temps = [2.0 * f.ekin / (dof * k_b) for f in frames if f.ekin >= 0]
    return sum(temps) / len(temps) if temps else 0.0


def energy_drift(frames: list[TrajFrame], last_n: int | None = None) -> float:
    """Linear slope of potential energy vs frame index (kJ/mol per frame)."""
    if len(frames) < 2:
        return 0.0
    subset = frames[-last_n:] if last_n else frames
    n = len(subset)
    xs = list(range(n))
    ys = [f.epot for f in subset]
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den
