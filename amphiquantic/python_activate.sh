#!/usr/bin/env bash

if [ ! -d ".venv" ]; then
    echo "No virtual environment found. Run ./scripts/steps/python_install.sh first."
    echo "The easiest way to do this is to run ./build"
    exit 1
fi

. .venv/bin/activate
