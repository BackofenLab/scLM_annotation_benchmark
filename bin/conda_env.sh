#!/bin/bash
# Conda Environment Management Helper
# Usage: source util/conda_env.sh
# Then call: setup_conda_env "env_name" "requirements_file"

# Function to check if environment needs to be recreated
check_environment() {
    local env_name="$1"
    local requirements_file="$2"
    
    # Check if environment exists
    if ! conda info --envs | grep -q "$env_name"; then
        echo "[INFO] Environment '$env_name' does not exist"
        return 1
    fi
    
    # Check if requirements file has changed since last environment creation
    # Create a hash file to track requirements changes
    local hash_file="/tmp/${env_name}_requirements_hash"
    local current_hash=$(md5sum "$requirements_file" | cut -d' ' -f1)
    
    if [ -f "$hash_file" ]; then
        local stored_hash=$(cat "$hash_file")
        if [ "$current_hash" = "$stored_hash" ]; then
            echo "[INFO] Environment '$env_name' exists and requirements haven't changed"
            return 0
        else
            echo "[INFO] Requirements file has changed, environment needs update"
            return 1
        fi
    else
        echo "[INFO] No hash file found, environment needs update"
        return 1
    fi
}

# Main function to setup conda environment
setup_conda_env() {
    local env_name="$1"
    local requirements_file="$2"
    local python_version="${3:-3.11}"  # Default to Python 3.11
    
    # Load conda shell functions
    CONDA_BASE=$(conda info --base)
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    
    # Check if environment exists and is up to date
    if check_environment "$env_name" "$requirements_file"; then
        echo "[INFO] Using existing conda environment '$env_name'"
        conda activate "$env_name"
    else
        echo "[INFO] Creating/updating conda environment '$env_name' from requirements.txt..."
        # Remove existing environment if it exists
        if conda info --envs | grep -q "$env_name"; then
            echo "[INFO] Removing existing environment '$env_name'"
            conda env remove -n "$env_name" -y
        fi
        
        # Create new environment
        conda create -y -n "$env_name" python="$python_version"
        conda activate "$env_name"
        pip install -r "$requirements_file"
        
        # Store hash of requirements file
        md5sum "$requirements_file" | cut -d' ' -f1 > "/tmp/${env_name}_requirements_hash"
        echo "[INFO] Environment '$env_name' created successfully"
    fi
}

# Function to run a command in a conda environment
run_in_env() {
    local env_name="$1"
    shift  # Remove first argument, leaving the rest as the command
    
    conda run -n "$env_name" "$@"
}

# Export functions so they can be used by sourcing scripts
export -f check_environment
export -f setup_conda_env
export -f run_in_env 