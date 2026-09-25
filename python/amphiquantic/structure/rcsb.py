"""Fetch structures from RCSB PDB (PDB and PDBx/mmCIF)."""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from pathlib import Path

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


def normalize_pdb_id(pdb_id: str) -> str:
    code = pdb_id.strip().upper()
    if not re.fullmatch(r"[0-9][A-Z0-9]{3}", code):
        raise ValueError(f"Invalid PDB ID: {pdb_id!r}")
    return code


def _download(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "amphiquantic/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_pdb(pdb_id: str, timeout: float = 30.0) -> str:
    """Download legacy PDB format from RCSB."""
    code = normalize_pdb_id(pdb_id)
    url = RCSB_PDB_URL.format(pdb_id=code)
    try:
        text = _download(url, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise FileNotFoundError(f"RCSB PDB not found: {code}") from exc
    if "404" in text[:200] and "Not Found" in text:
        raise FileNotFoundError(f"RCSB PDB not found: {code}")
    return text


def fetch_mmcif(pdb_id: str, timeout: float = 30.0) -> str:
    """Download PDBx/mmCIF from RCSB."""
    code = normalize_pdb_id(pdb_id)
    url = RCSB_CIF_URL.format(pdb_id=code)
    try:
        text = _download(url, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise FileNotFoundError(f"RCSB mmCIF not found: {code}") from exc
    return text


def save_structure(
    pdb_id: str,
    output_dir: str | Path,
    formats: tuple[str, ...] = ("pdb", "cif"),
    timeout: float = 30.0,
) -> dict[str, Path]:
    """Fetch and save requested formats; returns paths keyed by extension."""
    code = normalize_pdb_id(pdb_id)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if "pdb" in formats:
        pdb_path = out_dir / f"{code.lower()}.pdb"
        pdb_path.write_text(fetch_pdb(code, timeout=timeout))
        paths["pdb"] = pdb_path
    if "cif" in formats or "mmcif" in formats:
        cif_path = out_dir / f"{code.lower()}.cif"
        cif_path.write_text(fetch_mmcif(code, timeout=timeout))
        paths["cif"] = cif_path
    return paths


def fetch_structure_text(pdb_id: str, fmt: str = "pdb", timeout: float = 30.0) -> str:
    """Return structure text in ``pdb`` or ``cif``/``mmcif`` format."""
    fmt = fmt.lower()
    if fmt == "pdb":
        return fetch_pdb(pdb_id, timeout=timeout)
    if fmt in {"cif", "mmcif", "pdbx"}:
        return fetch_mmcif(pdb_id, timeout=timeout)
    raise ValueError(f"Unsupported format: {fmt}")
