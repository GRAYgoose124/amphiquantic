from numba import njit
from numba.typed import Dict
from numba.core import types

from .mappings import ATOM_COLORS, ATOM_RADII, BOND_DISTANCES

BOND_DISTANCES_NUMBA = Dict.empty(
    key_type=types.UniTuple(types.unicode_type, 2),
    value_type=types.UniTuple(types.float64, 2),
)

for key, value in BOND_DISTANCES.items():
    BOND_DISTANCES_NUMBA[key] = value
