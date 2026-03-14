from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dataset_configs_no_longer_define_subset_size():
    dataset_files = sorted((REPO_ROOT / "datasets").glob("*.yml"))
    assert dataset_files

    offenders = []
    for dataset_file in dataset_files:
        config = yaml.safe_load(dataset_file.read_text()) or {}
        if "subset_size" in config:
            offenders.append(dataset_file.name)

    assert offenders == []


def test_main_workflow_uses_split_data_processes():
    workflow_text = (REPO_ROOT / "main.nf").read_text()

    for snippet in [
        "process download {",
        "process preprocess {",
        "process ontology_mapping {",
        "process subset {",
        "resolveSubsetSize(",
        "ontology_out = ontology_mapping(ontology_input)",
        "subset_out = subset(subset_input)",
    ]:
        assert snippet in workflow_text

    download_section = workflow_text.split("process download {", 1)[1].split("process preprocess {", 1)[0]
    assert "--apply_ontology_mapping" not in download_section


def test_nextflow_config_defines_subset_categories():
    config_text = (REPO_ROOT / "nextflow.config").read_text()

    assert "params.subset_categories" in config_text
    assert "params.min_cells_per_type" in config_text
    assert "'openrouter': 1000" in config_text


def test_readme_documents_new_pipeline_and_cache_invalidation():
    readme_text = (REPO_ROOT / "README.md").read_text()

    for snippet in [
        "`subset_categories`",
        "`min_cells_per_type`",
        "cache/subset_qc/<dataset>/sub<size>/",
        "rm -f cache/*_sub*.h5ad",
        "apply_ontology_mapping",
    ]:
        assert snippet in readme_text
