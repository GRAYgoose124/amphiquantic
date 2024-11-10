#!/usr/bin/env bash

cd rust/
maturin develop
if [ $? -ne 0 ]; then
    exit 1
fi
