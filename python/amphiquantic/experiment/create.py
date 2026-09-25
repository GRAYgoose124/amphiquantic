"""Create a new experiment from a protocol template."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

import yaml

from amphiquantic.experiment.manifest import write_manifest

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_template_config(protocol_name: str) -> dict[str, Any]:
    path = TEMPLATES_DIR / protocol_name / "config.yaml"
    if not path.is_file():
        raise ValueError(f"Unknown protocol template: {protocol_name}")
    return yaml.safe_load(path.read_text())


def _render_protocol(protocol_name: str, experiment_root: Path) -> str:
    template = (TEMPLATES_DIR / protocol_name / "protocol.sh").read_text()
    return (
        template.replace("{{EXPERIMENT_ROOT}}", str(experiment_root))
        .replace("{{AMPI_ROOT}}", str(REPO_ROOT))
    )


def create_experiment(
    protocol_name: str,
    *,
    experiments_dir: Path | None = None,
    experiment_id: str | None = None,
    receptor: str | Path | None = None,
    ligand: str | Path | None = None,
    ligand_from_pdb: str | None = None,
    ligand_resname: str | None = None,
    ligand_aqtop: str | Path | None = None,
    cofactor: str | Path | None = None,
    chain_b: str | Path | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> Path:
    """Create experiments/{uuid}/ with inputs, protocol.sh, manifest.yaml."""
    experiments_dir = experiments_dir or (REPO_ROOT / "experiments")
    experiment_id = experiment_id or str(uuid.uuid4())
    experiment_root = experiments_dir / experiment_id

    if experiment_root.exists():
        raise FileExistsError(f"Experiment already exists: {experiment_root}")

    config = _load_template_config(protocol_name)
    if config_overrides:
        config.update(config_overrides)

    (experiment_root / "inputs").mkdir(parents=True)
    (experiment_root / "scripts").mkdir(parents=True)
    (experiment_root / "outputs").mkdir(parents=True)

    inputs: dict[str, str] = {}
    if receptor is not None:
        src = Path(receptor)
        if src.is_file():
            dst = experiment_root / "inputs" / "receptor.pdb"
            shutil.copy(src, dst)
            inputs["receptor"] = "inputs/receptor.pdb"
        else:
            ref = str(receptor)
            inputs["receptor"] = ref
            if len(ref) == 4 and ref.isalnum():
                config["pdb_id"] = ref.upper()
    if ligand_from_pdb is not None:
        from amphiquantic.ligand.db import (
            extract_hetatm_residue,
            ligand_smiles_from_hetatm,
            save_ligand_smiles,
        )
        from amphiquantic.structure import fetch_pdb, normalize_pdb_id

        pdb_id, resname = ligand_from_pdb.split(":", 1)
        pdb_id = normalize_pdb_id(pdb_id)
        resname = resname.strip().upper()
        config["ligand_resname"] = resname
        if config.get("pdb_id") is None:
            config["pdb_id"] = pdb_id
        pdb_text = fetch_pdb(pdb_id)
        het = extract_hetatm_residue(pdb_text, resname)
        (experiment_root / "inputs" / "reference_ligand.pdb").write_text(het)
        inputs["reference_ligand"] = "inputs/reference_ligand.pdb"
        smiles = ligand_smiles_from_hetatm(pdb_text, resname)
        save_ligand_smiles(smiles, experiment_root / "inputs" / "ligand.smi")
        inputs["ligand"] = "inputs/ligand.smi"
    if ligand_resname is not None:
        config["ligand_resname"] = ligand_resname.strip().upper()
    if ligand_aqtop is not None:
        src = Path(ligand_aqtop)
        dst = experiment_root / "inputs" / "ligand.aqtop"
        shutil.copy(src, dst)
        inputs["ligand_aqtop"] = "inputs/ligand.aqtop"
    if ligand is not None:
        src = Path(ligand)
        if src.is_file():
            dst = experiment_root / "inputs" / "ligand.smi"
            shutil.copy(src, dst)
            inputs["ligand"] = "inputs/ligand.smi"
        else:
            inputs["ligand"] = str(ligand)
            (experiment_root / "inputs" / "ligand.smi").write_text(str(ligand).strip() + "\n")
    if cofactor is not None:
        src = Path(cofactor)
        dst = experiment_root / "inputs" / "cofactor.smi"
        if src.is_file():
            shutil.copy(src, dst)
        else:
            dst.write_text(str(cofactor).strip() + "\n")
        inputs["cofactor"] = "inputs/cofactor.smi"
    if chain_b is not None:
        src = Path(chain_b)
        dst = experiment_root / "inputs" / "chain_b.pdb"
        shutil.copy(src, dst)
        inputs["chain_b"] = "inputs/chain_b.pdb"

    config["inputs"] = {**config.get("inputs", {}), **inputs}

    steps_src = TEMPLATES_DIR / protocol_name / "scripts"
    if steps_src.is_dir():
        for item in steps_src.iterdir():
            if item.is_file():
                dst = experiment_root / "scripts" / item.name
                shutil.copy(item, dst)
                dst.chmod(0o755)

    protocol_path = experiment_root / "protocol.sh"
    protocol_path.write_text(_render_protocol(protocol_name, experiment_root))
    protocol_path.chmod(0o755)

    (experiment_root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    write_manifest(
        experiment_root,
        experiment_id=experiment_id,
        protocol_name=protocol_name,
        config=config,
        repo_root=REPO_ROOT,
    )
    return experiment_root
