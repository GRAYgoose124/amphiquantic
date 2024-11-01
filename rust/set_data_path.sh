#!/usr/bin/env bash

# Use BASH_SOURCE[0] instead of $0 to correctly get the script's path when sourced
AMPHI_HOME_PATH=$(dirname $(dirname $(realpath ${BASH_SOURCE[0]})))

# Build paths based on AMPHI_HOME_PATH
export AMPHI_DATA_PATH=$AMPHI_HOME_PATH/rust/data
export AMPHI_SCRIPTS_PATH=$AMPHI_HOME_PATH/scripts
export AMPHI_STEPS_PATH=$AMPHI_SCRIPTS_PATH/steps
export AMPHI_BIN_PATH=$AMPHI_HOME_PATH/bin

echo "AMPHI_HOME_PATH: $AMPHI_HOME_PATH"
echo "AMPHI_DATA_PATH: $AMPHI_DATA_PATH"
echo "AMPHI_SCRIPTS_PATH: $AMPHI_SCRIPTS_PATH"
echo "AMPHI_BIN_PATH: $AMPHI_BIN_PATH"
