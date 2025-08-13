#!/usr/bin/env python3
"""
Test script to verify virtual environment setup and package installation.
"""
import os
import subprocess
import sys
from pathlib import Path

def setup_virtual_env(venv_path: Path = None) -> None:
    """
    Set up a virtual environment and install required packages.
    
    Args:
        venv_path: Path to create virtual environment (default: ./test_venv)
    """
    if venv_path is None:
        venv_path = Path("test_venv")
    
    # Required packages defined as environment variable
    REQUIRED_PACKAGES = os.getenv("DOWNLOAD_REQUIREMENTS", "tqdm,requests,anndata,pyyaml").split(",")
    
    print(f"[INFO] Setting up virtual environment at {venv_path}")
    print(f"[INFO] Required packages: {REQUIRED_PACKAGES}")
    
    # Create virtual environment if it doesn't exist
    if not venv_path.exists():
        print(f"[INFO] Creating virtual environment at {venv_path}")
        result = subprocess.run([sys.executable, "-m", "venv", str(venv_path)], 
                               capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] Failed to create venv: {result.stderr}")
            return
    else:
        print(f"[INFO] Virtual environment already exists at {venv_path}")
    
    # Determine pip executable path
    if os.name == 'nt':  # Windows
        pip_exe = venv_path / "Scripts" / "pip"
        python_exe = venv_path / "Scripts" / "python"
    else:  # Unix-like
        pip_exe = venv_path / "bin" / "pip"
        python_exe = venv_path / "bin" / "python"
    
    print(f"[INFO] Python executable: {python_exe}")
    print(f"[INFO] Pip executable: {pip_exe}")
    
    # Check if executables exist
    if not python_exe.exists():
        print(f"[ERROR] Python executable not found at {python_exe}")
        return
    if not pip_exe.exists():
        print(f"[ERROR] Pip executable not found at {pip_exe}")
        return
    
    # Install/upgrade pip
    print("[INFO] Upgrading pip in virtual environment")
    result = subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], 
                           capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Failed to upgrade pip: {result.stderr}")
        return
    else:
        print(f"[SUCCESS] Pip upgraded: {result.stdout.strip()}")
    
    # Install required packages
    print(f"[INFO] Installing packages: {REQUIRED_PACKAGES}")
    result = subprocess.run([str(pip_exe), "install"] + REQUIRED_PACKAGES, 
                           capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Failed to install packages: {result.stderr}")
        return
    else:
        print(f"[SUCCESS] Packages installed: {result.stdout}")
    
    print("[INFO] Virtual environment setup complete")
    
    # Test if packages can be imported in the venv
    print("[INFO] Testing package imports in virtual environment...")
    for package in REQUIRED_PACKAGES:
        # Map package names to import names
        import_name = package
        if package == "pyyaml":
            import_name = "yaml"
        
        test_cmd = [str(python_exe), "-c", f"import {import_name}; print(f'{import_name} imported successfully')"]
        result = subprocess.run(test_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[SUCCESS] {package} -> {result.stdout.strip()}")
        else:
            print(f"[ERROR] {package} -> {result.stderr.strip()}")

def test_current_environment():
    """Test if packages are available in the current environment."""
    print("\n" + "="*50)
    print("Testing current Python environment:")
    print(f"Python executable: {sys.executable}")
    
    packages = ["tqdm", "requests", "anndata", "yaml"]
    for package in packages:
        try:
            if package == "yaml":
                import yaml
            elif package == "tqdm":
                import tqdm
            elif package == "requests":
                import requests
            elif package == "anndata":
                import anndata
            print(f"[SUCCESS] {package} is available in current environment")
        except ImportError as e:
            print(f"[ERROR] {package} not available in current environment: {e}")

def main():
    print("Virtual Environment Test Script")
    print("="*50)
    
    # Test current environment first
    test_current_environment()
    
    # Set up and test virtual environment
    print("\n" + "="*50)
    print("Setting up virtual environment:")
    venv_path = Path("test_venv")
    setup_virtual_env(venv_path)
    
    print(f"\n[INFO] Virtual environment created at: {venv_path.absolute()}")
    print(f"[INFO] To manually activate: source {venv_path}/bin/activate")
    print(f"[INFO] To clean up: rm -rf {venv_path}")

if __name__ == "__main__":
    main()
