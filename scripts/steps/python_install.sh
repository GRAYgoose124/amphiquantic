#!/usr/bin/env bash

if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

. ./python/python_activate.sh

pip install maturin
pip install -e .

