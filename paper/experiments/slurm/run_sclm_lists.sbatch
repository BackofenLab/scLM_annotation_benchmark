#!/bin/bash
# The scLM arms of the imperfect-list analysis (S1_mmcontext, S2_langcell): one embedding-based
# wrapper of the benchmark on the 1,000-cell test sets, once per drawn list in
# paper/experiments/results_lists/lists/, which reaches the wrapper through CANDIDATE_TYPES_FILE.
# A list equal to the dataset's E list of the same seed (EU, which differs only in the "uncertain"
# option that these models cannot give) is skipped, and so is a run whose predictions exist.
#
# Submit from the repository root, with the wrapper's environment (envs/models/<MODEL>.yml) active
# and its prepare.sh run, after a pipeline run has put the test sets into cache/datasets/subset/
# (Zeisel through human orthologs). Without SLURM, run it with bash from the repository root.
#
#   mkdir -p paper/experiments/slurm/logs
#   sbatch --export=ALL,MODEL=mmcontext_blind,ARM=S1_mmcontext paper/experiments/slurm/run_sclm_lists.sbatch
#   sbatch --export=ALL,MODEL=langcell_blind,ARM=S2_langcell paper/experiments/slurm/run_sclm_lists.sbatch
#
# One job per model: mmContext writes <input>_pp.h5ad next to its input, so two of its jobs must
# not run at once. DATASETS (separated by colons) limits the datasets.
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --job-name=sclm_lists
#SBATCH --output=paper/experiments/slurm/logs/sclm_lists_%j.out
#SBATCH --error=paper/experiments/slurm/logs/sclm_lists_%j.err
set -euo pipefail

: "${MODEL:?set MODEL, mmcontext_blind or langcell_blind}"
: "${ARM:?set ARM, S1_mmcontext or S2_langcell}"
DATASETS="${DATASETS:-HFRA:HIHA:human_pancreas:Zeisel}"
LISTS=paper/experiments/results_lists/lists
OUT_ROOT=paper/experiments/results_lists/predictions
cd "${SLURM_SUBMIT_DIR:-.}"

for DS in ${DATASETS//:/ }; do
  IN="cache/datasets/subset/${DS}_sub1000_stratified.h5ad"
  [ "$DS" = Zeisel ] && IN="cache/datasets/subset/Zeisel_sub1000_stratified_human_orthologs.h5ad"
  for L in "$LISTS/${DS}"__*.txt; do
    STEM=$(basename "$L" .txt); SEED=${STEM##*__}; COND=${STEM#*__}; COND=${COND%__*}
    [ "$COND" != E ] && cmp -s "$L" "$LISTS/${DS}__E__${SEED}.txt" && continue
    OUT="$OUT_ROOT/$ARM/$DS/$COND/default/$SEED/predictions.csv"
    [ -f "$OUT" ] && continue
    echo "[$(date '+%F %T')] $DS $COND $SEED -> $OUT"
    CANDIDATE_TYPES_FILE="$PWD/$L" bash "models/$MODEL/src/run.sh" "$IN" "$OUT" "datasets/$DS.yml" \
      cache/models
  done
done
echo "[$(date '+%F %T')] $ARM done"
