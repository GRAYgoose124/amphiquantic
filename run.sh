#!/usr/bin/env bash

# set data path
if [ -z "$PDBVIZ_DATA_PATH" ]; then
    . ./rust/set_data_path.sh
fi

# 0 build 
./scripts/0_build_pdbviz.sh

# 1 run
if [ -f "scripts/1_ct-git-ignored.sh" ]; then
    ./scripts/1_ct-git-ignored.sh
else
    ./scripts/1_demo_script.sh
fi