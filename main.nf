nextflow.enable.dsl=2

import org.yaml.snakeyaml.Yaml

def loadYaml(String path) {
    def yaml = new Yaml()
    return yaml.load(new File(path).text)
}

// Resolve the subset size(s) for a model based on pattern matching. A category value may be a
// single size (e.g. 1000) or a list of sizes (e.g. [1000, 10000]); a model is run once per
// resolved size. Always returns a List<Integer>.
def resolveSubsetSizes(String model, Map categories) {
    def hit = categories.find { entry ->
        entry.key != 'default' && model.toLowerCase().contains(entry.key.toLowerCase())
    }
    def value = hit ? hit.value : categories.get('default', 10000)
    return (value instanceof List) ? value : [value]
}

def isHumanOrganism(def organismVal) {
    def org = (organismVal ?: 'Homo sapiens').toString().trim().toLowerCase()
    return org == 'homo sapiens' || org == 'human'
}

def normalizeOrganismKey(def organismVal) {
    return (organismVal ?: 'Homo sapiens').toString().trim().toLowerCase()
}

def modelAllowedForOrganism(String model, def organismVal, Map blacklistByOrganism) {
    def orgKey = normalizeOrganismKey(organismVal)
    def blockedModels = (blacklistByOrganism ?: [:]).get(orgKey, [])
    return !blockedModels.collect { it.toString() }.contains(model)
}

// Expected prediction filenames produced by a model's run.sh.
// Most models write a single predictions.csv. OpenRouter LLM models write one
// predictions_${SLUG}.csv per id in openrouter_LLM.model_names (slug = id with
// '/' -> '--'), so the cache check needs the exact filename list to detect
// when the user changes which OpenRouter ids to run.
def expectedPredictionFiles(String model) {
    if (!model.toLowerCase().contains('openrouter')) {
        return ['predictions.csv']
    }
    def cfgPath = "models/${model}/src/model_cfg.yml"
    def cfgFile = new File(cfgPath)
    if (!cfgFile.exists()) {
        log.warn "OpenRouter model ${model}: ${cfgPath} not found; falling back to predictions.csv"
        return ['predictions.csv']
    }
    def cfg
    try {
        cfg = new Yaml().load(cfgFile.text)
    } catch (Exception e) {
        log.warn "OpenRouter model ${model}: could not parse ${cfgPath} (${e.message}); falling back to predictions.csv"
        return ['predictions.csv']
    }
    def root = cfg?.openrouter_LLM ?: [:]
    def names = root.model_names ?: (root.model_name ? [root.model_name] : [])
    if (!names) {
        log.warn "OpenRouter model ${model}: no model_names in ${cfgPath}; falling back to predictions.csv"
        return ['predictions.csv']
    }
    return names.collect { name -> "predictions_${name.toString().replace('/', '--')}.csv" }
}

process download {
    container params.container_image
    conda 'envs/download.yml'
    storeDir "${params.cache_dir}/datasets/raw"

    input:
    tuple val(dataset), path(cfg_yaml)

    tag { dataset }

    output:
    tuple(path("${dataset}_raw.h5ad"), val(dataset))

    script:
    """
    set -euo pipefail
    download.py \
      --dataset  "${dataset}" \
      --cfg_yaml "${cfg_yaml}" \
      --out      "${dataset}_raw.h5ad"
    """
}

process download_from_zenodo {
    container params.container_image
    conda 'envs/download.yml'
    storeDir "${params.cache_dir}/datasets/subset"

    input:
    tuple val(dataset), val(subset_size), val(strategy), val(annotation_key), path(manifest)

    tag { "${dataset}xsub${subset_size}x${strategy}" }

    output:
    tuple(path("${dataset}_sub${subset_size}_${strategy}.h5ad"), val(dataset), val(subset_size), val(strategy))

    script:
    """
    set -euo pipefail
    fetch_zenodo_subset.py \\
      --manifest "${manifest}" \\
      --dataset "${dataset}" \\
      --subset_size ${subset_size} \\
      --strategy "${strategy}" \\
      --annotation_key "${annotation_key}" \\
      --output "${dataset}_sub${subset_size}_${strategy}.h5ad"
    """
}

process preprocess {
    conda 'envs/download.yml'
    storeDir "${params.cache_dir}/datasets/preprocessed"

    input:
    tuple path(raw_file), val(dataset), path(cfg_yaml), val(cache_dir), val(annotation_key), val(apply_mapping)

    tag { dataset }

    output:
    tuple(path("${dataset}_pp.h5ad"), val(dataset), val(annotation_key))

    script:
    """
    set -euo pipefail
    preprocess.py \\
      --input "${raw_file}" \\
      --output "${dataset}_pp.h5ad" \\
      --config "${cfg_yaml}" \\
      --summary "${dataset}_pp_summary.txt" \\
      --cache_dir "${cache_dir}" \\
      --annotation_key "${annotation_key}" \\
      --apply_ontology_mapping ${apply_mapping} \\
      --dataset "${dataset}"
    """
}

process subset {
    conda 'envs/download.yml'
    storeDir "${params.cache_dir}/datasets/subset"

    input:
    tuple path(pp_file), val(dataset), path(cfg_yaml), val(cache_dir), val(annotation_key), val(subset_size), val(min_cells_per_type), val(strategy)

    tag { "${dataset}xsub${subset_size}x${strategy}" }

    output:
    tuple(path("${dataset}_sub${subset_size}_${strategy}.h5ad"), val(dataset), val(subset_size), val(strategy))

    script:
    """
    set -euo pipefail
    subset.py \\
      --input "${pp_file}" \\
      --output "${dataset}_sub${subset_size}_${strategy}.h5ad" \\
      --annotation_key "${annotation_key}" \\
      --subset_size ${subset_size} \\
      --dataset "${dataset}" \\
      --qc_dir "${cache_dir}/subset_qc/${dataset}/sub${subset_size}_${strategy}" \\
      --min_cells_per_type ${min_cells_per_type} \\
      --strategy "${strategy}"
    """
}

process extract_cell_types {
    container params.container_image
    conda 'envs/cell_types.yml'

    input:
        path(h5ad_files)
        path(cfg_files)
        val(cache_dir)  // list of files

    script:
        """
        if [ ! -f ${cache_dir}/cell_types.csv ]; then
            extract_cell_types.py \
                --h5ads ${h5ad_files.join(' ')} \
                --cfgs ${cfg_files.join(' ')} \
                --out ${cache_dir}/cell_types.csv
        else
            echo "Cell type collection exists, if you want to rerun, delete cache/cell_types.csv"
        fi
        """
}

process prepare_model {
    container params.container_image
    conda 'git git-lfs'
    
    input:
    tuple val(model), path(model_dir)

    tag { model }

    output:
    tuple val(model), path(model_dir)

    script:
    """
    if [ ! -d "${model_dir}" ]; then
        echo "Model directory ${model_dir} does not exist"
        exit 1
    fi
    if [ -f "${model_dir}/src/prepare.sh" ]; then
        echo "Running preparation script for ${model}..."
        cd ${model_dir}/src
        bash prepare.sh
    else
        echo "No preparation needed for ${model}"
    fi
    """
}

process run_model {
    container params.container_image
    secret 'OPENROUTER_API_KEY'

    input:
    tuple path(data_file), path(model_dir), val(dataset), path(cfg_yaml), val(model), val(cache_dir), val(subset_size), val(strategy), val(expected_files)

    tag { "${dataset}x${model}x${strategy}" }

    conda { "envs/models/${model}.yml" }

    when:
    // NOTE: `when:` is evaluated before task inputs are staged into the work dir.
    // So we must resolve config via project-relative path, not `cfg_yaml` (which would be a staged filename).
    def cfg = loadYaml("datasets/${dataset}.yml")
    def organism = cfg?.get('organism')
    def blacklist = params.model_blacklist_by_organism ?: [:]
    modelAllowedForOrganism(model, organism, blacklist)

    output:
    tuple(path("results/*/*/*/predictions*.csv"), val(dataset), val(model), val(cache_dir), val(subset_size), val(strategy))

    script:
    """
    export API_KEY="\$OPENROUTER_API_KEY"
    echo "Subset size of ${dataset}: ${subset_size} (strategy: ${strategy})"

    cache_base="${cache_dir}/models/predictions/${model}/${dataset}/${subset_size}_${strategy}"
    results_dir=./results/${model}/${dataset}/${subset_size}_${strategy}

    mkdir -p \${results_dir}

    read -ra EXPECTED_FILES <<< "${expected_files}"

    # Pre-stage any cached files into results_dir so run.sh sees them and
    # can skip the corresponding slugs (relevant for OpenRouter where one
    # task produces multiple predictions_<slug>.csv files; non-OpenRouter
    # models have a single expected file so this is a no-op or a full hit).
    all_cached=true
    for f in "\${EXPECTED_FILES[@]}"; do
        if [ -f "\${cache_base}/\$f" ]; then
            cp "\${cache_base}/\$f" "\${results_dir}/\$f"
        else
            all_cached=false
        fi
    done

    if [ "\$all_cached" = "false" ]; then
        mkdir -p \${cache_base}
        bash "${model_dir}/src/run.sh" \\
            "${data_file}" \\
            \${results_dir}/predictions.csv \\
            "${cfg_yaml}" \\
            "${cache_dir}/models"
        # Persist only the expected files into the cache.
        for f in "\${EXPECTED_FILES[@]}"; do
            cp "\${results_dir}/\$f" "\${cache_base}/\$f"
        done
    else
        echo "Cell type prediction already exists for ${dataset} x ${model} (subset ${subset_size}, strategy ${strategy})."
        echo "To rerun, delete files in: \${cache_base}/"
    fi
    """
}

// Stage 1 of evaluation: normalize + review-filter + ontology-map predictions, producing a
// cached <base>_mapped.csv (+ sidecar stats.json). Cached next to the predictions and skipped
// on rerun unless the predictions or the mappings CSV change (sha staleness check). This is the
// only stage that touches the stateful cell_type_ontology_mappings.csv; maxForks 1 serializes
// the append-new-labels side-effect so parallel tasks can't lost-update the shared CSV.
process map_predictions {
    container params.container_image
    conda 'envs/evaluation.yml'
    errorStrategy 'ignore'
    maxForks 1

    input:
    tuple path(predictions), val(dataset), val(model), val(cache_dir), val(subset_size), val(strategy), val(apply_ontology_mapping)

    tag { "${dataset}x${model}" }

    output:
    tuple(path("mapped/*_mapped.csv"), path("mapped/*_mapped.stats.json"), val(dataset), val(model), val(cache_dir), val(subset_size))

    script:
    """
    set -euo pipefail
    cache_base="${cache_dir}/models/predictions/${model}/${dataset}/${subset_size}_${strategy}"
    mappings="${cache_dir}/cell_type_ontology_mappings.csv"
    mkdir -p mapped
    sha_now=\$(sha256sum "\$mappings" 2>/dev/null | awk '{print \$1}' || echo "NO_MAPPINGS")

    # Reuse cached mapped CSVs when the CSV + its stats sidecar are present AND the mappings
    # CSV is unchanged (sha match).
    need_map=false
    read -ra PREDS <<< "${predictions}"
    for p in "\${PREDS[@]}"; do
        base=\$(basename "\$p" .csv)
        if [ -f "\${cache_base}/\${base}_mapped.csv" ] && \\
           [ -f "\${cache_base}/\${base}_mapped.stats.json" ] && \\
           [ -f "\${cache_base}/\${base}_mapped.mappings.sha" ] && \\
           [ "\$(cat "\${cache_base}/\${base}_mapped.mappings.sha")" = "\$sha_now" ]; then
            cp "\${cache_base}/\${base}_mapped.csv" "mapped/\${base}_mapped.csv"
            cp "\${cache_base}/\${base}_mapped.stats.json" "mapped/\${base}_mapped.stats.json"
        else
            need_map=true
        fi
    done

    if [ "\$need_map" = "true" ]; then
        pred_files=\$(echo ${predictions} | tr ' ' ',')
        map_predictions.py --pred "\$pred_files" --model ${model} --dataset ${dataset} \\
            --out-dir mapped --cache_dir ${cache_dir} --apply_ontology_mapping ${apply_ontology_mapping}
        mkdir -p "\${cache_base}"
        for f in mapped/*_mapped.csv mapped/*_mapped.stats.json; do
            [ -e "\$f" ] && cp "\$f" "\${cache_base}/"
        done
        for m in mapped/*_mapped.csv; do
            b=\$(basename "\$m" _mapped.csv)
            printf '%s' "\$sha_now" > "\${cache_base}/\${b}_mapped.mappings.sha"
        done
    else
        echo "Mapped predictions up-to-date for ${dataset} x ${model} (subset ${subset_size}, ${strategy})."
    fi
    """
}

// Stage 2 of evaluation: pure scoring on the pre-mapped predictions (loads the CL ontology
// once). Re-running this is cheap and never re-does ontology mapping.
process score_predictions {
    container params.container_image
    conda 'envs/evaluation.yml'
    errorStrategy 'ignore'

    input:
    tuple path(mapped), path(stats), val(dataset), val(model), val(cache_dir), val(subset_size)

    tag { "${dataset}x${model}" }

    publishDir(path: 'results/', mode: 'copy', overwrite: true)

    output:
    path "*.json"
    path "**/*.pdf", optional: true

    script:
    """
    pred_files=\$(echo ${mapped} | tr ' ' ',')
    evaluate.py --pred "\$pred_files" --model ${model} --out ${dataset}_${model}__sub${subset_size}.json --cache_dir ${cache_dir} --subset-size ${subset_size} --max-unmapped-ratio 0.99 --min-scoring-coverage ${params.min_scoring_coverage} --use_text_encoder ${params.use_text_encoder} --text_encoder_model ${params.text_encoder_model}
    """
}

process combine_results {
    container params.container_image
    conda 'envs/evaluation.yml'

    publishDir(path: 'results/', mode: 'copy', overwrite: true)

    input:
    path(json_path)
    file(datasets)

    output:
    tuple path("aggregated*.csv"), path("aggregated_plot*.png")

    script:
    """
    aggregate.py --input ${json_path} --output ./ --level 0 --datasets ${datasets}
    """
}

workflow {
    // --- Upfront Zenodo manifest validation ---

    if (params.download_from_zenodo) {
        def zenodo_manifest = loadYaml("datasets/zenodo_manifest.yml")
        def zenodo_available = zenodo_manifest.files.keySet()
        def zenodo_sizes = params.models.collectMany { resolveSubsetSizes(it, params.subset_categories) }.unique()
        def missing = []
        params.datasets.each { ds ->
            zenodo_sizes.each { sz ->
                def fname = "${ds}_sub${sz}_${params.subset_strategy}.h5ad".toString()
                if (!zenodo_available.contains(fname)) missing << fname
            }
        }
        if (missing) {
            log.error "download_from_zenodo=true but these files are not in datasets/zenodo_manifest.yml:"
            missing.each { log.error "  - ${it}" }
            log.error "Either disable download_from_zenodo, or adjust datasets / subset_categories / subset_strategy."
            System.exit(1)
        }
    }

    // --- Model preparation (runs first so unique_sizes is available to both data pipelines) ---

    def hasOpenRouter = (params.models ?: []).any { it.toLowerCase().contains('openrouter') }
    if (hasOpenRouter) {
        log.info "OpenRouter model(s) selected. The API key is read via the 'secret' directive on run_model."
        log.info "If not yet set, run:  nextflow secrets set OPENROUTER_API_KEY 'sk-or-v1-...'"
        log.info "(Tasks will fail with 'Required secret is missing: OPENROUTER_API_KEY' if not set.)"
    }

    Channel
        .fromList(params.models)
        .map { model -> tuple(model, file("models/${model}")) }
        .set { prepare_input }

    prepared_models = prepare_model(prepare_input)

    Channel
        .fromList(params.models)
        .flatMap { model -> resolveSubsetSizes(model, params.subset_categories).collect { sz -> tuple(model, sz) } }
        .set { model_sizes }

    model_sizes
        .map { model, size -> size }
        .unique()
        .set { unique_sizes }

    // --- Data pipeline: either download→preprocess→subset OR fetch from Zenodo ---

    if (params.download_from_zenodo) {
        Channel
            .fromList(params.datasets)
            .combine(unique_sizes)
            .map { ds, sz ->
                def cfg = loadYaml("datasets/${ds}.yml")
                tuple(ds, sz, params.subset_strategy, cfg.annotation_key, file("datasets/zenodo_manifest.yml"))
            }
            .set { zenodo_input }

        subset_out = download_from_zenodo(zenodo_input)
    } else {
        Channel
            .fromList(params.datasets)
            .map { dataset -> tuple(dataset, file("datasets/${dataset}.yml")) }
            .set { download_input }

        download_out = download(download_input)

        download_out
            .map { raw_file, dataset ->
                def cfg = loadYaml("datasets/${dataset}.yml")
                def apply_mapping = cfg.containsKey('apply_ontology_mapping') ? cfg.apply_ontology_mapping : params.apply_ontology_mapping
                tuple(raw_file, dataset, file("datasets/${dataset}.yml"), file(params.cache_dir), cfg.annotation_key, apply_mapping)
            }
            .set { preprocess_input }

        preprocess_out = preprocess(preprocess_input)

        preprocess_out
            .combine(unique_sizes)
            .map { pp_file, dataset, annotation_key, subset_size ->
                tuple(pp_file, dataset, file("datasets/${dataset}.yml"), file(params.cache_dir), annotation_key, subset_size, params.min_cells_per_type, params.subset_strategy)
            }
            .set { subset_input }

        subset_out = subset(subset_input)
    }

    // --- Combine subsets with models ---

    model_sizes.set { model_size_pairs }

    subset_out
        .combine(prepared_models)
        .combine(model_size_pairs)
        .filter { data_file, dataset, data_subset_size, data_strategy, model_name, model_dir, model_name2, model_subset_size ->
            model_name2 == model_name && data_subset_size == model_subset_size
        }
        .map { data_file, dataset, data_subset_size, data_strategy, model_name, model_dir, model_name2, model_subset_size ->
            def expected = expectedPredictionFiles(model_name).join(' ')
            tuple(data_file, model_dir, dataset, file("datasets/${dataset}.yml"), model_name,
                  file(params.cache_dir), data_subset_size, data_strategy, expected)
        }
        .set { model_input }

    predictions = run_model(model_input)

    // --- Evaluation (two cached stages: map -> score) ---

    Channel.value(params.apply_ontology_mapping)
            .set { apply_mapping_ch }

    // Stage 1: normalize + review-filter + ontology-map (cached, skipped on rerun).
    map_in = predictions.combine(apply_mapping_ch)
                        .map { pred, dataset, model, cache_dir, size, strat, apply_flag ->
                            tuple(pred, dataset, model, cache_dir, size, strat, apply_flag)
                        }
    mapped = map_predictions(map_in)

    // Stage 2: pure scoring on the mapped predictions (its input tuple is exactly map's output).
    eval_out = score_predictions(mapped)

    Channel
        .fromList(params.datasets)
        .map { dataset -> file("datasets/${dataset}.yml") }
        .collect()
        .set { datasets }

    combine_results(eval_out[0].collect(), datasets)
}

workflow.onComplete {
    def warningFile = file("${params.cache_dir}/ONTOLOGY_MAPPING_WARNING.txt")
    if (warningFile.exists()) {
        println ""
        println "=" * 70
        println "  ONTOLOGY MAPPING WARNING: new cell-type mappings were discovered and added"
        println "  to cell_type_ontology_mappings.csv with reviewed=False."
        println "  Details: ${warningFile.toString()}"
        println "  ACTION: review the new mappings and set 'reviewed=True'. Until then, models below"
        println "  the coverage floor (see 'reviewed_fraction' in aggregated_by_dataset_model.csv)"
        println "  report NaN metrics. After reviewing, re-run with -resume to score them."
        println "=" * 70
        println ""

        def timestamp = new Date().format('yyyy-MM-dd_HH-mm-ss')
        def archivedWarningFile = file("${params.cache_dir}/ONTOLOGY_MAPPING_WARNING_${timestamp}.txt")
        warningFile.renameTo(archivedWarningFile)
        println "Warning file archived as: ${archivedWarningFile.toString()}"
    }

    println "Pipeline completed at: ${new Date()}"
    if (workflow.success) {
        println "Pipeline completed successfully!"
    } else {
        println "Pipeline completed with errors"
    }
}
