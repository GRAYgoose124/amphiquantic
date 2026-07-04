#!/usr/bin/env bash

if [ ! -d ".venv" ]; then
    echo "No virtual environment found. Run ./build first (or: uv sync)."
    exit 1
fi

# uv creates and manages .venv by default
. .venv/bin/activate
