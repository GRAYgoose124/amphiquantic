#!/usr/bin/env bash
. ./rust/set_data_path.sh
. ./amphiquantic/python_activate.sh

./scripts/run_workflow $*
