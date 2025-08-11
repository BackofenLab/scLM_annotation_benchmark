#!/bin/bash
# Conda Environment Management Helper
# Usage: source conda_env.sh
# Then call: setup_conda_env "env_name" "requirements_file" [python_version]

set -euo pipefail

# --- helpers ---
_hash_req() {
  local f="$1"
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$f" | cut -d' ' -f1
  elif command -v md5 >/dev/null 2>&1; then
    md5 -q "$f"
  else
    shasum -a 256 "$f" | cut -d' ' -f1
  fi
}
_env_exists() {
  # exact name match in first column
  conda env list | awk '{print $1}' | grep -xq "$1"
}

check_environment() {
  local env_name="$1"
  local requirements_file="$2"

  if ! _env_exists "$env_name"; then
    echo "[INFO] Environment '$env_name' does not exist"
    return 1
  fi

  local hash_file="/tmp/${env_name}_requirements_hash"
  if [[ ! -f "$requirements_file" ]]; then
    echo "[INFO] Requirements file '$requirements_file' missing; env needs update"
    return 1
  fi

  local current_hash=$(_hash_req "$requirements_file")
  if [[ -f "$hash_file" ]]; then
    local stored_hash
    stored_hash=$(cat "$hash_file")
    if [[ "$current_hash" == "$stored_hash" ]]; then
      echo "[INFO] Environment '$env_name' exists and requirements haven't changed"
      return 0
    fi
    echo "[INFO] Requirements changed; env needs update"
    return 1
  else
    echo "[INFO] No hash file found; env needs update"
    return 1
  fi
}

setup_conda_env() {
  local env_name="$1"
  local requirements_file="$2"
  local python_version="${3:-3.11}"

  # load conda
  local CONDA_BASE
  CONDA_BASE=$(conda info --base)
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"

  if check_environment "$env_name" "$requirements_file"; then
    echo "[INFO] Using existing conda environment '$env_name'"
  else
    if _env_exists "$env_name"; then
      echo "[INFO] Removing existing environment '$env_name'"
      conda env remove -n "$env_name" -y
    fi

    echo "[INFO] Creating environment '$env_name' (python=$python_version, pip<25 from conda-forge)"
    # Create with conda-forge and pin pip<25 to avoid vendoring issues on macOS
    conda create -y -n "$env_name" -c conda-forge --override-channels \
      "python=${python_version}" "pip<25"

    # Belt-and-suspenders: force-reinstall pip<25 from conda-forge
    conda install -y -n "$env_name" -c conda-forge --override-channels --force-reinstall "pip<25"

    echo "[INFO] Installing Python requirements with python -m pip"
    conda run -n "$env_name" python -m pip --version
    conda run -n "$env_name" python -m pip install -r "$requirements_file"

    # Store hash of requirements file
    _hash_req "$requirements_file" > "/tmp/${env_name}_requirements_hash"
    echo "[INFO] Environment '$env_name' created/updated successfully"
  fi
}

run_in_env() {
  local env_name="$1"
  shift
  conda run -n "$env_name" "$@"
}

export -f check_environment
export -f setup_conda_env
export -f run_in_env
