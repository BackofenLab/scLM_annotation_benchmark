from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_subset_cli_exposes_required_arguments():
    result = run_cli("bin/subset.py", "--help")

    assert result.returncode == 0, result.stderr
    assert "--subset_size" in result.stdout
    assert "--qc_dir" in result.stdout


def test_ontology_map_cli_exposes_required_arguments():
    result = run_cli("bin/ontology_map.py", "--help")

    assert result.returncode == 0, result.stderr
    assert "--cache_dir" in result.stdout
    assert "--apply" in result.stdout


def test_download_cli_no_longer_exposes_ontology_mapping_flag():
    result = run_cli("bin/download.py", "--help")

    assert result.returncode == 0, result.stderr
    assert "--apply_ontology_mapping" not in result.stdout
