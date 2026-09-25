#!/usr/bin/env bash
# Complex workflow entry for run_workflow
AMPHI_STEPS_TO_RUN="all"
AMPHI_RUN_STEPS="build"
AMPHI_RECEPTOR="${AMPHI_RECEPTOR:-tests/fixtures/complex/ala_dipeptide.pdb}"
AMPHI_LIGAND="${AMPHI_LIGAND:-CCO}"
AMPHI_COMPLEX_OUT="${AMPHI_COMPLEX_OUT:-output/complex}"
