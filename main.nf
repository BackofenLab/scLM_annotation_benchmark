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

// Container image for a given model. R-based annotators (SingleR tiers, scType) need
// R/Bioconductor, absent from the Python-only params.container_image, so they run in
// params.r_container_image (built from envs/Dockerfile.r); every other model keeps the default.
// Only consulted under the container profiles (docker/apptainer/helix); ignored under conda.
def containerFor(String model) {
    def rModels = (params.r_container_models ?: []) as List
    return rModels.contains(model) ? params.r_container_image : params.container_image
}

// True if a dataset declares a CELLxGENE reference for SingleR (singler_reference_source: cellxgene).
// Such datasets get their reference fetched by fetch_singler_reference and passed to the R method via
// SINGLER_REF_H5AD; datasets without the flag (default "celldex") use the built-in celldex path.
def singlerRefIsCellxgene(String dataset) {
    def cfg = loadYaml("datasets/${dataset}.yml")
    return (cfg?.get('singler_reference_source') ?: 'celldex').toString() == 'cellxgene'
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

// Shared task body for the model-running processes. `run_model` (non-OpenRouter) and
// `run_model_openrouter` use this byte-for-byte; the only difference is that the
// OpenRouter variant declares `secret 'OPENROUTER_API_KEY'` and prepends the
// `export API_KEY=...` line, so the key is only ever required for OpenRouter tasks.
def runModelScript(data_file, model_dir, dataset, cfg_yaml, model, cache_dir, subset_size, strategy, expected_files) {
    return """
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
        # Persist agentic-blueprint provenance alongside predictions when the model
        # produced it (the edited blueprint + its coverage + the scenario the agent
        # saw). No-op for models that don't write these files.
        for extra in blueprint.produced.py coverage.json scenario_config.jsonc; do
            [ -f "\${results_dir}/\$extra" ] && cp "\${results_dir}/\$extra" "\${cache_base}/\$extra" || true
        done
    else
        echo "Cell type prediction already exists for ${dataset} x ${model} (subset ${subset_size}, strategy ${strategy})."
        echo "To rerun, delete files in: \${cache_base}/"
    fi
    """
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

// Fetch a prebuilt clustering context (<dataset>_context.h5ad) from the pinned Zenodo record,
// the context analogue of download_from_zenodo. Lets the Zenodo fast-path skip the heavy
// download->preprocess->build_context chain for cluster-level models. storeDir caches into the
// same datasets/context/ dir that build_context writes to, so the two paths are interchangeable.
process fetch_zenodo_context {
    container params.container_image
    conda 'envs/download.yml'
    storeDir "${params.cache_dir}/datasets/context"

    input:
    tuple val(dataset), val(annotation_key), path(manifest)

    tag { "${dataset}xcontext" }

    output:
    tuple(val(dataset), path("${dataset}_context.h5ad"))

    script:
    """
    set -euo pipefail
    fetch_zenodo_subset.py \\
      --manifest "${manifest}" \\
      --dataset "${dataset}" \\
      --filename "${dataset}_context.h5ad" \\
      --annotation_key "${annotation_key}" \\
      --output "${dataset}_context.h5ad"
    """
}

// Fetch + normalize a CELLxGENE Census dataset into a SingleR reference h5ad (for datasets whose YAML
// sets singler_reference_source: cellxgene). Output shares the query's convention (Ensembl var_names,
// log1p X, gene_names, counts layer) so predict.R reads it with the same zellkonverter call. Runs in
// the Python env (the R container lacks cellxgene_census); cached so the 3 sizes/reruns skip it.
process fetch_singler_reference {
    container params.container_image
    conda 'envs/download.yml'
    storeDir "${params.cache_dir}/datasets/reference"

    input:
    tuple val(dataset), val(census_id), val(census_version), val(label_key), val(species)

    tag { "${dataset}xsinglerref" }

    output:
    tuple(val(dataset), path("${dataset}_singler_ref.h5ad"))

    script:
    """
    set -euo pipefail
    prepare_singler_reference.py \\
      --census_id "${census_id}" \\
      --census_version "${census_version}" \\
      --label_key "${label_key}" \\
      --species "${species}" \\
      --cap_per_label ${params.singler_reference_cap_per_label} \\
      --out "${dataset}_singler_ref.h5ad"
    """
}

process preprocess {
    conda 'envs/download.yml'
    storeDir "${params.cache_dir}/datasets/preprocessed"
    // Preprocessing depends on Ensembl BioMart (gene-symbol -> Ensembl ID) for symbol-based
    // datasets, which is intermittently flaky; a transient bad response should retry, not abort
    // the whole run. storeDir means already-preprocessed datasets are never re-run.
    errorStrategy 'retry'
    maxRetries 2

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

// Non-OpenRouter models. Identical to run_model_openrouter except it does NOT declare
// `secret 'OPENROUTER_API_KEY'`, so these tasks run without any OpenRouter key being set.
process run_model {
    container { containerFor(model) }   // R models (SingleR tiers) -> r_container_image; others unchanged
    // Conda env creation can transiently fail under contention (e.g. LinkError when tasks sharing
    // one env prefix race to build it) or on flaky package downloads. Retry a few times so the
    // env, once built by the winning task, is reused; after that, skip this one task rather than
    // aborting the whole run, so other models/datasets still complete.
    errorStrategy { task.attempt <= 3 ? 'retry' : 'ignore' }
    maxRetries 3

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
    runModelScript(data_file, model_dir, dataset, cfg_yaml, model, cache_dir, subset_size, strategy, expected_files)
}

// OpenRouter LLM models. Declares the secret so the key is injected securely (never
// written into .command.sh / .nextflow.log) and re-exports it as API_KEY for run.sh.
// Because the `secret` directive is static per process, isolating it here is what makes
// the key required ONLY for OpenRouter tasks.
process run_model_openrouter {
    container params.container_image
    secret 'OPENROUTER_API_KEY'

    input:
    tuple path(data_file), path(model_dir), val(dataset), path(cfg_yaml), val(model), val(cache_dir), val(subset_size), val(strategy), val(expected_files)

    tag { "${dataset}x${model}x${strategy}" }

    conda { "envs/models/${model}.yml" }

    when:
    def cfg = loadYaml("datasets/${dataset}.yml")
    def organism = cfg?.get('organism')
    def blacklist = params.model_blacklist_by_organism ?: [:]
    modelAllowedForOrganism(model, organism, blacklist)

    output:
    tuple(path("results/*/*/*/predictions*.csv"), val(dataset), val(model), val(cache_dir), val(subset_size), val(strategy))

    script:
    'export API_KEY="$OPENROUTER_API_KEY"\n' + runModelScript(data_file, model_dir, dataset, cfg_yaml, model, cache_dir, subset_size, strategy, expected_files)
}

// Builds the clustering CONTEXT for cluster-level annotators: a union-seeded, floored-proportional
// sample of the FULL preprocessed data (up to context_max_size) that is guaranteed to contain every
// eval-subset cell. Cached under datasets/context (storeDir) so it is built once and reused.
process build_context {
    container params.container_image
    conda 'envs/download.yml'
    storeDir "${params.cache_dir}/datasets/context"
    tag { dataset }

    input:
    tuple val(dataset), path(full_pp), path(subset_files), val(annotation_key)

    output:
    tuple val(dataset), path("${dataset}_context.h5ad")

    script:
    """
    python ${projectDir}/bin/build_context.py \\
        --full ${full_pp} \\
        --subsets ${subset_files} \\
        --output ${dataset}_context.h5ad \\
        --annotation_key "${annotation_key}" \\
        --max_size ${params.context_max_size} \\
        --min_cells_per_type ${params.context_min_cells_per_type} \\
        --qc_csv ${dataset}_context_qc.csv
    """
}

// Cluster-level annotators (sctype_blind, scsa_blind) run with a clustering context. Identical to
// run_model except it stages the context h5ad and exports CONTEXT_H5AD, which the model's predict
// step reads to cluster+score on the context and emit only the eval cells. Kept separate so the
// context is only wired for models that understand it.
process run_model_cluster {
    container { containerFor(model) }   // scType -> r_container_image; SCSA (Python) keeps the default
    // See run_model: retry transient conda env-creation failures, then skip rather than abort.
    errorStrategy { task.attempt <= 3 ? 'retry' : 'ignore' }
    maxRetries 3

    input:
    tuple path(data_file), path(model_dir), val(dataset), path(cfg_yaml), val(model), val(cache_dir), val(subset_size), val(strategy), val(expected_files), path(context_file)

    tag { "${dataset}x${model}x${strategy}" }

    conda { "envs/models/${model}.yml" }

    when:
    def cfg = loadYaml("datasets/${dataset}.yml")
    def organism = cfg?.get('organism')
    def blacklist = params.model_blacklist_by_organism ?: [:]
    modelAllowedForOrganism(model, organism, blacklist)

    output:
    tuple(path("results/*/*/*/predictions*.csv"), val(dataset), val(model), val(cache_dir), val(subset_size), val(strategy))

    script:
    "export CONTEXT_H5AD=\"\$(realpath ${context_file})\"\n" + runModelScript(data_file, model_dir, dataset, cfg_yaml, model, cache_dir, subset_size, strategy, expected_files)
}

// SingleR with a custom CELLxGENE reference. Identical to run_model except it stages the fetched
// reference h5ad and exports SINGLER_REF_H5AD, which predict.R reads to take the cellxgene path
// (labels = the reference's cell_type column). Kept separate so the reference is wired only for
// singler_blind on datasets that declare a cellxgene reference; celldex SingleR uses run_model.
process run_model_singler_ref {
    container { containerFor(model) }
    // See run_model: retry transient conda env-creation failures, then skip rather than abort.
    errorStrategy { task.attempt <= 3 ? 'retry' : 'ignore' }
    maxRetries 3

    input:
    tuple path(data_file), path(model_dir), val(dataset), path(cfg_yaml), val(model), val(cache_dir), val(subset_size), val(strategy), val(expected_files), path(reference_file)

    tag { "${dataset}x${model}x${strategy}" }

    conda { "envs/models/${model}.yml" }

    when:
    def cfg = loadYaml("datasets/${dataset}.yml")
    def organism = cfg?.get('organism')
    def blacklist = params.model_blacklist_by_organism ?: [:]
    modelAllowedForOrganism(model, organism, blacklist)

    output:
    tuple(path("results/*/*/*/predictions*.csv"), val(dataset), val(model), val(cache_dir), val(subset_size), val(strategy))

    script:
    "export SINGLER_REF_H5AD=\"\$(realpath ${reference_file})\"\n" + runModelScript(data_file, model_dir, dataset, cfg_yaml, model, cache_dir, subset_size, strategy, expected_files)
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
    // Serialize so the shared evaluation.yml conda env is built once (by the first task) and
    // reused, rather than parallel tasks racing to create the same env prefix (LinkError). Scoring
    // is fast, so the serialization cost is negligible. (map_predictions already runs maxForks 1.)
    maxForks 1

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

    def _needsContext = (params.models ?: []).any { ((params.cluster_context_models ?: []) as List).contains(it) }
    def _fetchContext = params.fetch_context_from_zenodo && _needsContext
    if (params.download_from_zenodo || _fetchContext) {
        def zenodo_manifest = loadYaml("datasets/zenodo_manifest.yml")
        def zenodo_available = zenodo_manifest.files.keySet()
        def missing = []
        if (params.download_from_zenodo) {
            def zenodo_sizes = params.models.collectMany { resolveSubsetSizes(it, params.subset_categories) }.unique()
            params.datasets.each { ds ->
                zenodo_sizes.each { sz ->
                    def fname = "${ds}_sub${sz}_${params.subset_strategy}.h5ad".toString()
                    if (!zenodo_available.contains(fname)) missing << fname
                }
            }
        }
        if (_fetchContext) {
            params.datasets.each { ds ->
                def fname = "${ds}_context.h5ad".toString()
                if (!zenodo_available.contains(fname)) missing << fname
            }
        }
        if (missing) {
            log.error "download_from_zenodo/fetch_context_from_zenodo: these files are not in datasets/zenodo_manifest.yml:"
            missing.each { log.error "  - ${it}" }
            log.error "Either disable the Zenodo fetch(es), or add the files to the manifest (and the record)."
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

    // Cluster-level annotators (sctype/scsa) additionally need a clustering context built from the
    // FULL data. `needsContext` gates the extra download→preprocess + build_context work.
    def clusterModels = (params.cluster_context_models ?: []) as List
    def needsContext = (params.models ?: []).any { clusterModels.contains(it) }

    // --- Data pipeline: either download→preprocess→subset OR fetch from Zenodo ---
    // `preprocess_full` (pp_file, dataset, annotation_key) is the FULL preprocessed data, used only
    // to build the clustering context. It is produced on the download→preprocess path, and also on
    // the Zenodo fast-path when a cluster-level model is selected (eval subsets still come from
    // Zenodo; only the context needs the full data). storeDir makes the extra work skip if cached.

    // Helper: full download → preprocess for a dataset channel of (dataset, cfg_file).
    def toPreprocessInput = { ch ->
        ch.map { raw_file, dataset ->
            def cfg = loadYaml("datasets/${dataset}.yml")
            def apply_mapping = cfg.containsKey('apply_ontology_mapping') ? cfg.apply_ontology_mapping : params.apply_ontology_mapping
            tuple(raw_file, dataset, file("datasets/${dataset}.yml"), file(params.cache_dir), cfg.annotation_key, apply_mapping)
        }
    }

    // Context source is INDEPENDENT of subset source:
    //   fetch_context_from_zenodo=true  -> download <dataset>_context.h5ad from Zenodo (fast path)
    //   fetch_context_from_zenodo=false -> build it locally (this is how the contexts are produced
    //                                      in the first place, to then upload to Zenodo).
    // The full download->preprocess runs only if we build the SUBSETS locally OR build the CONTEXTS.
    def buildContext = needsContext && !params.fetch_context_from_zenodo
    def needFullPreprocess = (!params.download_from_zenodo) || buildContext

    preprocess_full = Channel.empty()
    if (needFullPreprocess) {
        Channel
            .fromList(params.datasets)
            .map { dataset -> tuple(dataset, file("datasets/${dataset}.yml")) }
            .set { download_input }
        preprocess_full = preprocess(toPreprocessInput(download(download_input)))
    }

    // --- Eval subsets: fetch from Zenodo, or subset the full preprocessed data locally ---
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
        preprocess_full
            .combine(unique_sizes)
            .map { pp_file, dataset, annotation_key, subset_size ->
                tuple(pp_file, dataset, file("datasets/${dataset}.yml"), file(params.cache_dir), annotation_key, subset_size, params.min_cells_per_type, params.subset_strategy)
            }
            .set { subset_input }
        subset_out = subset(subset_input)
    }

    // --- Clustering context: fetch prebuilt from Zenodo, or build locally from full + subsets ---
    context_by_dataset = Channel.empty()
    if (needsContext && params.fetch_context_from_zenodo) {
        Channel
            .fromList(params.datasets)
            .map { ds -> tuple(ds, loadYaml("datasets/${ds}.yml").annotation_key, file("datasets/zenodo_manifest.yml")) }
            .set { ctx_fetch_input }
        context_by_dataset = fetch_zenodo_context(ctx_fetch_input)
    } else if (buildContext) {
        subset_out
            .map { f, ds, sz, strat -> tuple(ds, f) }
            .groupTuple()
            .set { subsets_by_dataset }

        preprocess_full
            .map { pp, ds, ak -> tuple(ds, pp, ak) }
            .join(subsets_by_dataset)
            .map { ds, pp, ak, subs -> tuple(ds, pp, subs, ak) }
            .set { context_build_input }

        context_by_dataset = build_context(context_build_input)
    }

    // --- SingleR custom reference: fetch a CELLxGENE reference h5ad per dataset that declares one ---
    // (singler_reference_source: cellxgene). One reference per dataset, shared across sizes; storeDir
    // caches it. celldex-referenced datasets (e.g. HIHA) produce nothing here and use the celldex path.
    reference_by_dataset = Channel.empty()
    if ((params.models ?: []).contains('singler_blind')) {
        Channel
            .fromList(params.datasets)
            .filter { ds -> singlerRefIsCellxgene(ds) }
            .map { ds ->
                def cfg = loadYaml("datasets/${ds}.yml")
                def ver = (cfg.singler_reference_version ?: cfg.census_version ?: 'stable').toString()
                def label = (cfg.singler_reference_label_key ?: 'cell_type').toString()
                def species = (cfg.species ?: cfg.organism ?: 'human').toString()
                tuple(ds, cfg.singler_reference_id.toString(), ver, label, species)
            }
            .set { singler_ref_fetch_input }
        reference_by_dataset = fetch_singler_reference(singler_ref_fetch_input)
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

    // Route models: OpenRouter (needs the secret), cluster-level
    // (needs the clustering context), else the default run_model. `clusterModels` is resolved
    // above from params.cluster_context_models.
    model_input
        .branch { data_file, model_dir, dataset, cfg_yaml, model, cache_dir, size, strategy, expected ->
            openrouter:   model.toLowerCase().contains('openrouter')
            cluster:      clusterModels.contains(model)
            singler_ref:  model == 'singler_blind' && singlerRefIsCellxgene(dataset)
            other:        true
        }
        .set { model_input_branched }

    // Cluster-level models: attach their dataset's context h5ad (cartesian + filter by dataset,
    // since each dataset has exactly one context but many model/size tasks).
    model_input_branched.cluster
        .map { data_file, model_dir, dataset, cfg_yaml, model, cache_dir, size, strategy, expected ->
            tuple(dataset, data_file, model_dir, cfg_yaml, model, cache_dir, size, strategy, expected) }
        .combine(context_by_dataset)
        .filter { ds, df, md, cfg, m, c, sz, st, exp, ctx_ds, ctx_file -> ds == ctx_ds }
        .map { ds, df, md, cfg, m, c, sz, st, exp, ctx_ds, ctx_file ->
            tuple(df, md, ds, cfg, m, c, sz, st, exp, ctx_file) }
        .set { cluster_input }

    // SingleR custom-reference models: attach their dataset's fetched reference h5ad (same
    // cartesian + filter-by-dataset shape as the cluster context above; one ref per dataset).
    model_input_branched.singler_ref
        .map { data_file, model_dir, dataset, cfg_yaml, model, cache_dir, size, strategy, expected ->
            tuple(dataset, data_file, model_dir, cfg_yaml, model, cache_dir, size, strategy, expected) }
        .combine(reference_by_dataset)
        .filter { ds, df, md, cfg, m, c, sz, st, exp, ref_ds, ref_file -> ds == ref_ds }
        .map { ds, df, md, cfg, m, c, sz, st, exp, ref_ds, ref_file ->
            tuple(df, md, ds, cfg, m, c, sz, st, exp, ref_file) }
        .set { singler_ref_input }

    predictions = run_model_openrouter(model_input_branched.openrouter)
                      .mix(run_model(model_input_branched.other))
                      .mix(run_model_cluster(cluster_input))
                      .mix(run_model_singler_ref(singler_ref_input))

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
