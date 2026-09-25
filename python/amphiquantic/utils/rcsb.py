"""Legacy RCSB helpers — prefer amphiquantic.structure."""

from amphiquantic.structure import fetch_pdb, normalize_pdb_id, save_structure


def save_pdb(pdb_id: str, parent_path: str) -> None:
    save_structure(normalize_pdb_id(pdb_id), parent_path, formats=("pdb",))
