#!/usr/bin/env bash

# Install the Rust extension into the project root .venv (not rust/.venv).
uv run maturin develop --manifest-path rust/Cargo.toml
if [ $? -ne 0 ]; then
    exit 1
fi
