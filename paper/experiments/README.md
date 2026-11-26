# Prompt-context ablation and imperfect candidate lists

Two analyses beside the benchmark, both on its 1,000-cell test sets of the four datasets with three
seeds (1234, 5678, 9012), and both scored with the benchmark's mapping and scoring scripts.

**Prompt-context ablation** (study tree `results/`). Each arm is a model with a prompt template.
The modes vary the prompt: M1 gives only the genes (a cell's 200 most expressed genes, or a
cluster's 25 marker genes), M2 adds the list of the dataset's cell types, M3 tissue and species,
M4 both, and M6 is M1 with the genes in random order. The `hvg` gene variant takes a cell's genes
from a pool of 5,000 highly variable genes (M1 and M4). The cluster arms use Leiden clusters at
the resolutions 0.5 and 1.0. `results/context_ablation.csv` holds these arms (`config.yml` also
defines A5, A6 and A7, which are not in the table):

| Arm | Model | Prompts |
|---|---|---|
| `A1_llm_percell` | openai/gpt-oss-120b (OpenRouter) | per cell, the benchmark's LLM template |
| `A2_base_percell` | google/gemma-2-2b (local) | per cell, the template of A1 |
| `A3_scllm_percell` | vandijklab/C2S-Scale-Gemma-2-2B (local) | per cell, the C2S-Scale template |
| `A4_llm_percluster` | openai/gpt-oss-120b (OpenRouter) | per cluster, from its marker genes |
| `A8_gemini_percluster` | google/gemini-3.1-pro-preview (OpenRouter) | the prompts of A4 |

**Imperfect candidate lists** (study tree `results_lists/`). Mode M2 with other lists than the
exact one: E (exact), EU (exact, "uncertain" allowed), L1 (a parent or child added for half of the
types), L2-2 and L2-5 (2 or 5 absent types added), L3 (a quarter of the types removed, "uncertain"
allowed) and L4 (L1, L2-5 and L3 together). The arms are A1, A3 and A8, four baselines that get
the list as a label constraint (B1 SingleR, B2 CellTypist, B3 scType, B4 SCSA; not EU), and two
single-cell language models that compare each cell with the listed names (S1 mmContext, S2
LangCell; not EU), run with the benchmark's wrappers `mmcontext_blind` and `langcell_blind`.

## Contents

| Path | Content |
|---|---|
| `config.yml` | datasets, arms, modes, gene variants, list conditions and scoring |
| `src/` | the scripts; each one describes itself at its top |
| `slurm/` | scripts that run the local arms and the scLM arms on a SLURM cluster |
| `results/context_ablation.csv` | result table of the ablation |
| `results/spend.csv` | spend of the OpenRouter arms at genes + list per run, summed over their distinct calls from the response caches (Figure S6) |
| `results_lists/runs.csv` | result table of the list analysis, one row per run |
| `results_lists/lists/` | the drawn lists, one file per dataset, condition and seed |

Of the study trees, only the last four entries are in the repository: prompts, model answers,
caches, clusters and scored runs are not, nor is the study's mapping table in `cache/`.

## Environments

- `env.yml`: `hvg.py`, `sentences.py`, `cluster.py`, `run_openrouter.py` and `aggregate.py`.
- `env-local.yml`, on a machine with a GPU: `run_local.py` (`env-local.lock.yml` is its export).
- The pipeline's evaluation environment (`envs/evaluation.yml`): `score.py`, `review_mappings.py`,
  `lists.py` and `run_reference.py`, which runs each baseline in its own environment (`--env`).

## Rerunning

Rerunning the runners calls every model again and costs money: `run_openrouter.py` sends billed
requests to OpenRouter (it needs `OPENROUTER_API_KEY`), `run_local.py` runs the checkpoints from
Hugging Face on a GPU, and `--dry-run` reports what a runner would call. Run all commands from
`paper/experiments/`, as the paths in `config.yml` are relative to it. The inputs are the test
sets, clustering contexts and SingleR references in `../../cache/datasets/`.

First copy the benchmark's mapping table, `../../resources/cell_type_ontology_mappings.csv`, and
the pinned Cell Ontology release, `../../cache/cl-basic_2026-06-08.obo` (downloaded by a pipeline
run), into `cache/`. Before scoring, `python src/review_mappings.py --seed-from-predictions
<study tree>/predictions --apply` adds the answers that the table lacks and resolves them by its
rules. `score.py` adds answers that are still missing as unreviewed rows. It skips every run that
has metrics and maps and scores the runs below the coverage threshold again; remove
`<study tree>/scored/` to score all runs again.

Prompt-context ablation:

```bash
python src/hvg.py                       # HVG pools in ../../cache/hvg/
python src/sentences.py                 # sentence lengths, results/sentence_stats.csv
python src/sentences.py --materialize   # prompt files of the per-cell arms
python src/cluster.py                   # clusters and the prompt files of A4
for f in results/prompts/*__A4_*; do cp "$f" "${f/A4_llm_percluster/A8_gemini_percluster}"; done
python src/run_openrouter.py --arms A1_llm_percell A4_llm_percluster A8_gemini_percluster
python src/run_local.py --arms A2_base_percell A3_scllm_percell
python src/score.py --study ablation --python <python of the evaluation environment>
python src/aggregate.py                 # results/context_ablation.csv
```

Imperfect candidate lists, after the ablation's `cluster.py` (A8 uses its clusters at 1.0):

```bash
python src/lists.py --write             # results_lists/lists/
python src/sentences.py --materialize --study lists
python src/cluster.py --study lists
python src/run_openrouter.py --study lists --arms A1_llm_percell A8_gemini_percluster
python src/run_local.py --study lists --arms A3_scllm_percell
(cd ../.. && sbatch --export=ALL,MODEL=mmcontext_blind,ARM=S1_mmcontext paper/experiments/slurm/run_sclm_lists.sbatch)
(cd ../.. && sbatch --export=ALL,MODEL=langcell_blind,ARM=S2_langcell paper/experiments/slurm/run_sclm_lists.sbatch)
python src/run_reference.py singler --env <environment>   # and celltypist, sctype, scsa
python src/score.py --study lists --python <python of the evaluation environment>
python src/lists.py --score             # results_lists/runs.csv and confusion_pairs.csv
```

`slurm/` runs the local arms on a SLURM cluster: `ship_to_gpu.sh` copies code, configuration and
the local arms' prompt files and creates the environment, `submit_shards.sh <arm> [dataset ...]`
submits one job per dataset, `fetch_from_gpu.sh` copies predictions, caches and logs back, and
`cleanup_gpu.sh` removes the study's files from the cluster; `STUDY=lists` switches the first three
to the list analysis. The ssh alias (`HOST`, default `gpu`), extra ssh options such as a jump host
(`SSH_OPTS`), the remote folder (`REMOTE_DIR`, default `context_ablation` in the home folder) and
the GPU node that `submit_shards.sh` checks (`NODE`) are variables; the model weights go to
`/scratch/global/$USER/hf`. `run_sclm_lists.sbatch` runs the scLM arms in a checkout of the whole
repository on a GPU node (see its header).
