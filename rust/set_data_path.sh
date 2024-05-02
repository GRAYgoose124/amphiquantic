#!/usr/bin/env bash
# Use BASH_SOURCE[0] instead of $0 to correctly get the script's path when sourced
export PDBVIZ_DATA_PATH=$(dirname $(realpath ${BASH_SOURCE[0]}))/data

echo "PDBVIZ_DATA_PATH set to $PDBVIZ_DATA_PATH"
