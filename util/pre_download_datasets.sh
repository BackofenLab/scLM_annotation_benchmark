#!/bin/bash
# Helper script to pre-download and subset datasets sequentially
# This script activates the conda environment and runs the Python pre-download script

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Path to conda environment file
CONDA_ENV_FILE="${REPO_ROOT}/envs/download.yml"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda is not available. Please install conda or activate it first."
    exit 1
fi

# Check if environment file exists
if [ ! -f "${CONDA_ENV_FILE}" ]; then
    echo "[ERROR] Conda environment file not found: ${CONDA_ENV_FILE}"
    exit 1
fi

# Get environment name from the YAML file (use basename without extension)
ENV_NAME=$(basename "${CONDA_ENV_FILE}" .yml)

echo "[INFO] Using conda environment: ${ENV_NAME}"
echo "[INFO] Environment file: ${CONDA_ENV_FILE}"

# Check if environment exists, if not create it
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Creating conda environment from ${CONDA_ENV_FILE}..."
    conda env create -f "${CONDA_ENV_FILE}" -n "${ENV_NAME}"
fi

# Activate the environment and run the Python script
echo "[INFO] Activating conda environment: ${ENV_NAME}"
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# Run the Python script with all passed arguments
echo "[INFO] Running pre-download script..."
python "${SCRIPT_DIR}/pre_download_datasets.py" "$@"

# Store exit code
EXIT_CODE=$?

# Deactivate conda environment
conda deactivate

exit ${EXIT_CODE}

