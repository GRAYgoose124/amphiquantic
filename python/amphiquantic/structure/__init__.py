"""Structure I/O: RCSB fetch and mmCIF conversion."""

from amphiquantic.structure.mmcif import mmcif_to_pdb, parse_atom_site
from amphiquantic.structure.prepare_receptor import prepare_apo_file, prepare_apo_pdb
from amphiquantic.structure.rcsb import (
    fetch_mmcif,
    fetch_pdb,
    fetch_structure_text,
    normalize_pdb_id,
    save_structure,
)

__all__ = [
    "fetch_mmcif",
    "fetch_pdb",
    "fetch_structure_text",
    "mmcif_to_pdb",
    "normalize_pdb_id",
    "parse_atom_site",
    "prepare_apo_file",
    "prepare_apo_pdb",
    "protonate_pdb_file",
    "save_structure",
]
