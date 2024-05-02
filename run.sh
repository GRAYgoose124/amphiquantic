#!/usr/bin/env bash

# set data path
if [ -z "$PDBVIZ_DATA_PATH" ]; then
    . ./pdbviz/set_data_path.sh
    echo "PDBVIZ_DATA_PATH set to $PDBVIZ_DATA_PATH"
fi

# 0 build 
./0_build_pdbviz.sh

# 1 run
if [ -f "1_ct-git-ignored.sh" ]; then
    ./1_ct-git-ignored.sh
else
    ./1_demo_script.sh
fi