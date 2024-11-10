#!/usr/bin/env bash
. ./rust/set_data_path.sh
. ./python/python_activate.sh

./scripts/run_workflow $*
