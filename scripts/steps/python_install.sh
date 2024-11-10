#!/usr/bin/env bash

if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

. ./amphiquantic/python_activate.sh
pip install -e .

