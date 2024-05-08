#!/usr/bin/env bash

# Get the steps to run from the first argument or use "0123" as default
STEPS_TO_RUN=${1:-"0123"}

# Set data path
if [ -z "$PDBVIZ_DATA_PATH" ]; then
    . ./rust/set_data_path.sh
fi

# Function to run a specific step based on its number
run_step() {
    step=$1
    default_script=$2
    git_ignored_script="scripts/${step}_git-ignored.sh"
    script=${git_ignored_script:-$default_script}
    
    if [ -f "$git_ignored_script" ]; then
        ./$git_ignored_script
    else
        ./$default_script
    fi
}

# Step 0: Build
if [[ "$STEPS_TO_RUN" == *"0"* ]]; then
    ./scripts/0_build_pdbviz.sh
fi

# Step 1
if [[ "$STEPS_TO_RUN" == *"1"* ]]; then
    run_step "1" "./scripts/1_demo_script.sh"
fi

# Step 2
if [[ "$STEPS_TO_RUN" == *"2"* ]]; then
    run_step "2" "./scripts/2_param_script.sh"
fi

# Step 3
if [[ "$STEPS_TO_RUN" == *"3"* ]]; then
    run_step "3" "./scripts/3_minimize.sh"
fi
