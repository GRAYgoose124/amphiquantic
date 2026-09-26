"""Trajectory analysis toolkit for amphiquantic (.aqtrj / .aqtop).

Pure Python + numpy implementations comparable to the basics of
``gmx analysis`` / ``cpptraj``: RMSD/RMSF, radius of gyration, distance /
angle / dihedral time series, RDF, MSD/diffusion, hydrogen bonds, and
energy-log statistics.
"""

from amphiquantic.analysis.selection import Selection, select
from amphiquantic.analysis.trajectory import Frame, Trajectory, load_aqtrj
from amphiquantic.analysis.structural import (
    kabsch,
    superpose,
    rmsd,
    rmsd_series,
    rmsf,
    radius_of_gyration,
    radius_of_gyration_series,
    distance_series,
    angle_series,
    dihedral_series,
)
from amphiquantic.analysis.rdf import radial_distribution_function, coordination_number
from amphiquantic.analysis.diffusion import unwrap_pbc, mean_squared_displacement, diffusion_coefficient
from amphiquantic.analysis.hbonds import hydrogen_bonds, hbond_occupancy
from amphiquantic.analysis.energy import (
    block_average,
    block_average_error,
    statistical_inefficiency,
    autocorrelation,
    equilibration_time,
)

__all__ = [
    "Selection",
    "select",
    "Frame",
    "Trajectory",
    "load_aqtrj",
    "kabsch",
    "superpose",
    "rmsd",
    "rmsd_series",
    "rmsf",
    "radius_of_gyration",
    "radius_of_gyration_series",
    "distance_series",
    "angle_series",
    "dihedral_series",
    "radial_distribution_function",
    "coordination_number",
    "unwrap_pbc",
    "mean_squared_displacement",
    "diffusion_coefficient",
    "hydrogen_bonds",
    "hbond_occupancy",
    "block_average",
    "block_average_error",
    "statistical_inefficiency",
    "autocorrelation",
    "equilibration_time",
]
