# Data and code of the paper

This folder holds everything that is specific to the paper: the stored predictions of the
benchmark, the result tables, the data behind the supplementary tables, the two additional
analyses, and the scripts that draw the figures and write the tables. Run all commands from the
repository folder, in the evaluation environment (`envs/evaluation.yml`).

## Contents

| Path | Content |
| --- | --- |
| `predictions/` | the answers of every run of the benchmark: 21 methods in two settings and the chance reference, four datasets, test sets of 1,000 and 10,000 cells |
| `benchmark.params.yml` | pipeline parameters that score the stored predictions |
| `results/benchmark_metrics.csv` | metrics per dataset, model and test set size |
| `results/per_class_scores.csv` | recall and ontology scores per true cell type and run |
| `results/confusion_pairs.csv` | cells per pair of true and predicted cell type and run |
| `results/rare_type_outcomes.csv` | recall per method and cell type with the type's share of the full dataset |
| `results/statistics/` | the summary statistics quoted in the text |
| `data/dataset_facts.csv`, `data/model_facts.csv` | facts about datasets and models with their sources, for the supplementary tables |
| `data/references.bib` | the references these tables cite |
| `data/cell_type_distributions/` | cells per cell type in the full datasets and in the 1,000-cell test sets |
| `data/openrouter_activity.csv` | one row per request to OpenRouter in the benchmark: model, provider, tokens, cost, latency, finish reason; `.audit.md` describes how it was assembled |
| `mapping_audit/` | the record of the manual review of the mapping table |
| `qc_plots/` | quality control plots of the four datasets |
| `experiments/` | the prompt-context ablation and the imperfect-list analysis (see `experiments/README.md`) |
| `scripts/` | figures, tables and statistics |
| `tests/` | tests of the scripts |

## Reproducing the results

1. **Result tables.** Score the stored predictions with the pipeline. No method runs, because all
   predictions are stored.

   ```bash
   nextflow run main.nf -profile local -params-file paper/benchmark.params.yml
   ```

   `results_paper/benchmark_metrics.csv`, `per_class_scores.csv` and `confusion_pairs.csv` are
   then identical to the files in `paper/results/`. The text similarity needs the SapBERT encoder,
   which the run downloads into `cache/`.

2. **Rare cell types.**

   ```bash
   python paper/scripts/rare_type_table.py
   ```

3. **Figures, tables and statistics.**

   ```bash
   python paper/scripts/figures.py --dpi 600 --figures 2 s1_harmonization s2_answer_types \
       s3_metric_agreement s6_cost_performance
   python paper/scripts/figures.py --dpi 300 --figures 3 4 s4_context_per_dataset \
       s5_context_supporting s7_lists_per_dataset s8_ontology_scores s9_rare_full \
       s10_rare_answer_types s11_hfra s12_hiha s13_zeisel s14_human_pancreas
   python paper/scripts/supplementary_tables.py
   python paper/scripts/ablation_tables.py
   python paper/scripts/dataset_table.py
   python paper/scripts/model_table.py
   python paper/scripts/extract_statistics.py
   ```

   Figures are written to `paper/figures/` as PDF and PNG (five of them at 600 dpi, as in the
   paper), tables to `paper/tables/`, statistics to `paper/results/statistics/`. The figure
   modules are `figures_benchmark.py` (Figure 2, S1 to S3), `figures_experiments.py` (Figure 3,
   S4 to S8) and `figures_rare_types.py` (Figure 4, S9 to S14); `methods.py` holds the method
   families and names that figures and tables share.

Figure 1 was manually created and is therefore not generated here.

## Notes

- The test sets, the clustering samples and the converted Zeisel file are on Zenodo
  (doi:10.5281/zenodo.24590958).
- The predictions and model answers of the two additional analyses are on Zenodo
  (`additional_analyses_data.zip`, doi:10.5281/zenodo.24590958), together with what scoring them
  needs; see `experiments/README.md`. Their prompts are not stored, as the scripts rebuild them.
