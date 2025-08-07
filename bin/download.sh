#!/bin/bash
DATASET=$1
CFG_DIR=$2
OUT=$3
DATA_CACHE=$4

# Setup conda environment
ENV_NAME="download-env"
REQUIREMENTS_FILE="download_requirements.txt"
source conda_env.sh
setup_conda_env "$ENV_NAME" "$REQUIREMENTS_FILE"

python download.py \
    --dataset   ${DATASET} \
    --cfg-dir   ${CFG_DIR} \
    --out       ${OUT} \
    --data_cache ${DATA_CACHE}