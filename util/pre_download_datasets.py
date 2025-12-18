#!/usr/bin/env python
"""
Helper script to pre-download and subset datasets sequentially.

This script reads the nextflow.config file and runs download.py for each dataset
sequentially. This pre-populates the cache with subsetted data, reducing memory
footprint when running the main Nextflow pipeline in parallel.

Usage:
    python bin/util/pre_download_datasets.py [--config nextflow.config] [--cache_dir cache]
    
    Or activate the conda environment first:
    conda activate <env>  # from envs/download.yml
    python bin/util/pre_download_datasets.py
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def parse_nextflow_config(config_path: Path) -> dict:
    """
    Parse nextflow.config to extract params.
    
    Returns a dictionary with parsed parameters.
    """
    params = {}
    
    if not config_path.exists():
        print(f"[WARN] Config file not found: {config_path}")
        return params
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Extract params.datasets
    datasets_match = re.search(r'params\.datasets\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if datasets_match:
        datasets_str = datasets_match.group(1)
        # Parse the list, handling quotes and whitespace
        datasets = [
            d.strip().strip("'\"") 
            for d in re.findall(r"['\"]([^'\"]+)['\"]", datasets_str)
        ]
        params['datasets'] = datasets
    
    # Extract params.cache_dir
    cache_dir_match = re.search(r'params\.cache_dir\s*=\s*[\'\"]([^\'\"]+)[\'\"]', content)
    if cache_dir_match:
        params['cache_dir'] = cache_dir_match.group(1)
    
    # Extract params.apply_ontology_mapping
    mapping_match = re.search(r'params\.apply_ontology_mapping\s*=\s*(true|false)', content)
    if mapping_match:
        params['apply_ontology_mapping'] = mapping_match.group(1).lower() == 'true'
    
    return params


def find_conda_env_path(env_name: str = "envs/download.yml") -> Path | None:
    """
    Find the path to the conda environment file.
    
    Returns the absolute path to the environment file, or None if not found.
    """
    # Try to find the environment file relative to the script location
    # Script is at util/pre_download_datasets.py, so go up one level to repo root
    script_dir = Path(__file__).parent.parent  # util/ -> repo root
    env_path = script_dir / env_name
    
    if env_path.exists():
        return env_path.resolve()
    
    # Try current directory
    env_path = Path(env_name)
    if env_path.exists():
        return env_path.resolve()
    
    return None


def run_download_with_conda(
    dataset: str,
    cache_dir: Path,
    cfg_yaml: Path,
    apply_ontology_mapping: bool,
    conda_env_path: Path | None = None,
    work_dir: Path | None = None
) -> bool:
    """
    Run download.py for a single dataset using conda environment.
    
    Returns True if successful, False otherwise.
    """
    # Determine paths
    # Script is at util/pre_download_datasets.py, so go up one level to repo root
    script_dir = Path(__file__).parent.parent  # util/ -> repo root
    download_script = script_dir / "bin" / "download.py"
    
    if not download_script.exists():
        print(f"[ERROR] download.py not found at {download_script}")
        return False
    
    if not cfg_yaml.exists():
        print(f"[ERROR] Config YAML not found: {cfg_yaml}")
        return False
    
    # Create temporary work directory for this dataset
    if work_dir is None:
        work_dir = cache_dir / f"_temp_{dataset}"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Output path in work directory
    output_path = work_dir / f"{dataset}_raw.h5ad"
    
    # Build command
    cmd = [
        sys.executable,
        str(download_script),
        "--dataset", dataset,
        "--cfg_yaml", str(cfg_yaml),
        "--cache_dir", str(cache_dir),
        "--out", str(output_path),
        "--apply_ontology_mapping", str(apply_ontology_mapping)
    ]
    
    print(f"\n{'='*80}")
    print(f"[INFO] Downloading dataset: {dataset}")
    print(f"[INFO] Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            cwd=str(script_dir),
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        print(f"\n[OK] Successfully downloaded and processed {dataset}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to download {dataset}: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error while downloading {dataset}: {e}")
        return False
    finally:
        # Clean up temporary work directory
        if work_dir.exists() and work_dir.name.startswith("_temp_"):
            try:
                import shutil
                shutil.rmtree(work_dir)
            except Exception as e:
                print(f"[WARN] Could not clean up {work_dir}: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pre-download and subset datasets sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults from nextflow.config
  python bin/util/pre_download_datasets.py
  
  # Specify custom config file
  python bin/util/pre_download_datasets.py --config custom.config
  
  # Override cache directory
  python bin/util/pre_download_datasets.py --cache_dir /path/to/cache
        """
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("nextflow.config"),
        help="Path to nextflow.config file (default: nextflow.config)"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help="Override cache directory from config"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Override datasets list from config"
    )
    parser.add_argument(
        "--apply_ontology_mapping",
        type=lambda x: x.lower() == 'true',
        default=None,
        help="Override apply_ontology_mapping from config"
    )
    parser.add_argument(
        "--conda_env",
        type=Path,
        default=None,
        help="Path to conda environment file (default: envs/download.yml)"
    )
    
    args = parser.parse_args()
    
    # Parse config
    print(f"[INFO] Reading configuration from {args.config}")
    config_params = parse_nextflow_config(args.config)
    
    # Use command-line overrides or config values
    datasets = args.datasets or config_params.get('datasets', [])
    cache_dir = args.cache_dir or Path(config_params.get('cache_dir', 'cache'))
    apply_ontology_mapping = (
        args.apply_ontology_mapping 
        if args.apply_ontology_mapping is not None 
        else config_params.get('apply_ontology_mapping', True)
    )
    
    if not datasets:
        print("[ERROR] No datasets specified. Please set params.datasets in nextflow.config or use --datasets")
        sys.exit(1)
    
    print(f"[INFO] Datasets to download: {datasets}")
    print(f"[INFO] Cache directory: {cache_dir}")
    print(f"[INFO] Apply ontology mapping: {apply_ontology_mapping}")
    
    # Find conda environment
    conda_env_path = args.conda_env or find_conda_env_path()
    if conda_env_path:
        print(f"[INFO] Using conda environment: {conda_env_path}")
        print(f"[INFO] Note: Make sure the conda environment is activated or use 'conda env create -f {conda_env_path}'")
    else:
        print(f"[WARN] Conda environment file not found. Make sure download.py dependencies are available.")
    
    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine datasets directory
    # Script is at util/pre_download_datasets.py, so go up one level to repo root
    script_dir = Path(__file__).parent.parent  # util/ -> repo root
    datasets_dir = script_dir / "datasets"
    
    if not datasets_dir.exists():
        print(f"[ERROR] Datasets directory not found: {datasets_dir}")
        sys.exit(1)
    
    # Run download for each dataset sequentially
    success_count = 0
    failed_datasets = []
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\n{'#'*80}")
        print(f"Processing dataset {i}/{len(datasets)}: {dataset}")
        print(f"{'#'*80}")
        
        cfg_yaml = datasets_dir / f"{dataset}.yml"
        
        if not cfg_yaml.exists():
            print(f"[ERROR] Config file not found: {cfg_yaml}")
            failed_datasets.append(dataset)
            continue
        
        success = run_download_with_conda(
            dataset=dataset,
            cache_dir=cache_dir,
            cfg_yaml=cfg_yaml,
            apply_ontology_mapping=apply_ontology_mapping,
            conda_env_path=conda_env_path
        )
        
        if success:
            success_count += 1
        else:
            failed_datasets.append(dataset)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully processed: {success_count}/{len(datasets)} datasets")
    
    if failed_datasets:
        print(f"Failed datasets: {', '.join(failed_datasets)}")
        sys.exit(1)
    else:
        print("All datasets processed successfully!")
        print(f"\n[INFO] Cached datasets are available in: {cache_dir}")
        print("[INFO] You can now run the main Nextflow pipeline with reduced memory footprint.")


if __name__ == "__main__":
    main()

