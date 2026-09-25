#!/usr/bin/env bash
set -euo pipefail
: "${AMPI_ROOT:?AMPI_ROOT must be set by protocol.sh}"
# shellcheck disable=SC1091
source "$AMPI_ROOT/scripts/amphi_env.sh"
amphi_setup "$AMPI_ROOT"

amphi_py -c "
import yaml
from pathlib import Path
from amphiquantic.ligand.db import (
    extract_hetatm_residue,
    ligand_smiles_from_hetatm,
    save_ligand_smiles,
    smiles_has_complete_stereo,
    smiles_is_openff_ready,
)
from amphiquantic.structure import fetch_pdb, normalize_pdb_id

root = Path('$AMPHI_EXPERIMENT_ROOT')
inputs = root / 'inputs'
inputs.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load((root / 'config.yaml').read_text())
ref = cfg.get('inputs', {}).get('receptor', '') or cfg.get('pdb_id', '')
holo = inputs / 'receptor.pdb'
if not holo.is_file():
    if len(str(ref)) == 4 and str(ref).isalnum():
        pdb_id = normalize_pdb_id(str(ref))
        text = fetch_pdb(pdb_id)
        holo.write_text(text)
        print(f'fetched receptor {pdb_id}')
    else:
        raise SystemExit(f'receptor input missing: {ref}')

resname = cfg.get('ligand_resname')
if resname:
    ref_lig = inputs / 'reference_ligand.pdb'
    holo_text = holo.read_text()
    if not ref_lig.is_file():
        try:
            het = extract_hetatm_residue(holo_text, str(resname))
        except LookupError:
            print(f'warning: no HETATM {resname} in holo PDB; skip reference ligand')
        else:
            ref_lig.write_text(het)
            print(f'wrote reference ligand {resname}')
    lig_smi = inputs / 'ligand.smi'
    refresh_smi = not lig_smi.is_file()
    if lig_smi.is_file() and not smiles_is_openff_ready(lig_smi.read_text().strip()):
        refresh_smi = True
    if refresh_smi:
        smiles = ligand_smiles_from_hetatm(holo_text, str(resname))
        save_ligand_smiles(smiles, lig_smi)
        print(f'wrote ligand.smi for {resname}')
"
