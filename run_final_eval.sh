#!/usr/bin/env bash
# Final evaluation over ALL cached predictions, using the cached cell_type_ontology_mappings.csv.
# Runs the two eval stages directly (no Nextflow, no model conda envs):
#   1) bin/map_predictions.py  -> <stem>_mapped.csv (+ .stats.json) into a temp dir
#   2) bin/evaluate.py         -> results/<dataset>_<model>.json (+ scored CSVs, confusion PDFs)
# Then aggregates all JSONs with bin/aggregate.py.
#
# One subset size per (model,dataset) is chosen exactly like the Nextflow pipeline
# (resolveSubsetSize): model name containing 'openrouter' or 'langcell_blind' -> 1000, else 10000.
set -uo pipefail

PROJECT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT"

# ---- config (edit as needed) ----
PY="${PY:-/Users/<user>/miniforge3/envs/evaluate/bin/python}"
OUTDIR="${OUTDIR:-results}"
STRATEGY="${STRATEGY:-stratified}"
USE_TEXT_ENCODER="${USE_TEXT_ENCODER:-false}"   # set to true to also compute SapBERT similarity
MIN_SCORING_COVERAGE="${MIN_SCORING_COVERAGE:-0.9}"  # min trusted-mapping coverage to emit metrics (else NaN)
TEXT_ENCODER_MODEL="${TEXT_ENCODER_MODEL:-cambridgeltl/SapBERT-from-PubMedBERT-fulltext}"
FORCE_REMAP="${FORCE_REMAP:-false}"             # set to true to ignore the *_mapped.csv cache and re-map
EXCLUDE_PATTERNS="${EXCLUDE_PATTERNS:-}"         # space-separated substrings; prediction files whose
                                                # basename contains any are skipped (e.g. "llama-3.1-8b-instruct")
MAKE_PLOTS="${MAKE_PLOTS:-true}"                 # true => also render bin/plot_model_comparison.py figures
# ---------------------------------

export KMP_DUPLICATE_LIB_OK=TRUE PYTHONWARNINGS=ignore MPLBACKEND=Agg
export PYTHONPATH="$PROJECT/bin:${PYTHONPATH:-}"

LOGDIR="$OUTDIR/logs"
MAPDIR="$OUTDIR/_mapped_work"
mkdir -p "$OUTDIR" "$LOGDIR" "$MAPDIR"

# Mirror resolveSubsetSize(model, subset_categories) from main.nf / nextflow.config.
resolve_size() {
  local m
  m="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  case "$m" in
    *openrouter*|*langcell_blind*) echo 1000 ;;
    *) echo 10000 ;;
  esac
}

PRED_ROOT="cache/models/predictions"
jsons=()
echo "Scanning $PRED_ROOT ..."
for model_dir in "$PRED_ROOT"/*/; do
  model="$(basename "$model_dir")"
  pref="$(resolve_size "$model")"
  alt=$([ "$pref" = 1000 ] && echo 10000 || echo 1000)
  for ds_dir in "$model_dir"*/; do
    [ -d "$ds_dir" ] || continue
    dataset="$(basename "$ds_dir")"
    # Locate the predictions for this (model,dataset). Prefer the size-tagged dir picked like
    # the pipeline (resolve_size), then the other size, then predictions directly under the
    # dataset dir (agent models use this irregular layout), then any subdir -- "take what is
    # available".
    pdir=""
    for cand in "${ds_dir}${pref}_${STRATEGY}" "${ds_dir}${alt}_${STRATEGY}" "${ds_dir%/}"; do
      if compgen -G "$cand/predictions*.csv" > /dev/null 2>&1; then pdir="$cand"; break; fi
    done
    if [ -z "$pdir" ]; then
      for cand in "$ds_dir"*/; do
        if compgen -G "$cand/predictions*.csv" > /dev/null 2>&1; then pdir="${cand%/}"; break; fi
      done
    fi
    [ -z "$pdir" ] && { echo "[skip] $model x $dataset: no predictions"; continue; }

    # Enumerate the real prediction files, EXCLUDING the persisted *_mapped.csv cache that now
    # lives next to them (otherwise predictions*.csv would re-glob them into predictions_mapped_mapped.csv).
    pred_files=()
    for p in "$pdir"/predictions*.csv; do
      bn="$(basename "$p")"
      case "$bn" in *_mapped.csv) continue;; esac
      skip=false
      for pat in $EXCLUDE_PATTERNS; do
        case "$bn" in *"$pat"*) skip=true; break;; esac
      done
      [ "$skip" = true ] && { echo "  [excl] $bn"; continue; }
      [ -e "$p" ] && pred_files+=("$p")
    done
    [ "${#pred_files[@]}" -eq 0 ] && { echo "[skip] $model x $dataset: no (non-mapped) predictions"; continue; }
    preds="$(printf '%s,' "${pred_files[@]}")"; preds="${preds%,}"
    work="$MAPDIR/${dataset}_${model}"; mkdir -p "$work"; rm -f "$work"/*_mapped.csv "$work"/*_mapped.stats.json 2>/dev/null
    log="$LOGDIR/${dataset}_${model}.log"
    echo "[run ] $model x $dataset  (size ${pdir##*/})"

    # Stage 1: map (normalize + review-filter + ontology-map), with the same sha-keyed cache as
    # the Nextflow map_predictions process. Reuse a cached <base>_mapped.csv (next to the
    # predictions) only when its stats + .mappings.sha exist and the mappings CSV is unchanged.
    # Editing cell_type_ontology_mappings.csv changes its sha and forces a re-map; FORCE_REMAP=true
    # ignores the cache entirely.
    sha_now=$(sha256sum cache/cell_type_ontology_mappings.csv 2>/dev/null | awk '{print $1}' || echo "NO_MAPPINGS")
    need_map=false
    for p in "${pred_files[@]}"; do
      base="$(basename "$p" .csv)"
      if [ "$FORCE_REMAP" != "true" ] && \
         [ -f "$pdir/${base}_mapped.csv" ] && \
         [ -f "$pdir/${base}_mapped.stats.json" ] && \
         [ -f "$pdir/${base}_mapped.mappings.sha" ] && \
         [ "$(cat "$pdir/${base}_mapped.mappings.sha")" = "$sha_now" ]; then
        cp "$pdir/${base}_mapped.csv" "$work/${base}_mapped.csv"
        cp "$pdir/${base}_mapped.stats.json" "$work/${base}_mapped.stats.json"
      else
        need_map=true
      fi
    done

    if [ "$need_map" = "true" ]; then
      if ! "$PY" bin/map_predictions.py --pred "$preds" --model "$model" --dataset "$dataset" \
            --out-dir "$work" --cache_dir cache --apply_ontology_mapping true > "$log" 2>&1; then
        echo "[FAIL map ] $model x $dataset  (see $log)"; continue
      fi
      # Persist freshly-mapped outputs next to the predictions + stamp the mappings sha.
      for f in "$work"/*_mapped.csv "$work"/*_mapped.stats.json; do
        [ -e "$f" ] && cp "$f" "$pdir/"
      done
      for m in "$work"/*_mapped.csv; do
        [ -e "$m" ] || continue
        b="$(basename "$m" _mapped.csv)"
        printf '%s' "$sha_now" > "$pdir/${b}_mapped.mappings.sha"
      done
    else
      echo "[cache] $model x $dataset: mapped predictions up-to-date" >> "$log"
    fi

    mapped="$(printf '%s,' "$work"/*_mapped.csv)"; mapped="${mapped%,}"
    # Stage 2: score
    te_args=(--use_text_encoder "$USE_TEXT_ENCODER")
    [ "$USE_TEXT_ENCODER" = "true" ] && te_args+=(--text_encoder_model "$TEXT_ENCODER_MODEL")
    if "$PY" bin/evaluate.py --pred "$mapped" --model "$model" \
          --out "$OUTDIR/${dataset}_${model}.json" --cache_dir cache --max-unmapped-ratio 0.99 \
          --min-scoring-coverage "$MIN_SCORING_COVERAGE" "${te_args[@]}" >> "$log" 2>&1; then
      echo "[done] $model x $dataset"
      jsons+=("$OUTDIR/${dataset}_${model}.json")
    else
      echo "[FAIL score] $model x $dataset  (see $log)"
    fi
  done
done

echo
echo "Produced ${#jsons[@]} JSON files in $OUTDIR."

# Aggregate all JSONs into CSV + plots (datasets ymls help parse dataset/model names).
ymls="$(printf '%s ' datasets/*.yml)"
echo "Aggregating ..."
"$PY" bin/aggregate.py --input "${jsons[@]}" --output "$OUTDIR" --level 0 --datasets $ymls \
  > "$LOGDIR/_aggregate.log" 2>&1 && echo "Aggregated -> $OUTDIR/aggregated_by_dataset_model.csv" \
  || echo "Aggregation failed (see $LOGDIR/_aggregate.log)"

# Comparison figures (bin/plot_model_comparison.py). Set MAKE_PLOTS=false to skip.
if [ "$MAKE_PLOTS" = "true" ]; then
  echo "Generating comparison plots ..."
  plot_args=(--input "$OUTDIR/aggregated_by_dataset_model.csv" --outdir "$OUTDIR/plots")
  # The script's default metric set excludes the text-encoder metrics, so request them explicitly
  # (alongside the default curated set) only when they were actually computed.
  if [ "$USE_TEXT_ENCODER" = "true" ]; then
    plot_args+=(--metrics balanced_accuracy accuracy rada_avg_similarity lin_avg_similarity \
      id_accuracy balanced_id_accuracy macro_f1 weighted_f1 \
      text_emb_norm_avg_raw text_emb_norm_avg_mapped text_emb_mapping_lift)
  fi
  "$PY" bin/plot_model_comparison.py "${plot_args[@]}" > "$LOGDIR/_plots.log" 2>&1 \
    && echo "Comparison plots -> $OUTDIR/plots/" \
    || echo "Plotting failed (see $LOGDIR/_plots.log)"
fi
