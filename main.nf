nextflow.enable.dsl=2

import org.yaml.snakeyaml.Yaml

def loadYaml(String path) {
    def yaml = new Yaml()
    return yaml.load(new File(path).text)
}

// Resolve the subset size for a model based on pattern matching
def resolveSubsetSize(String model, Map categories) {
    for (entry in categories) {
        if (entry.key != 'default' && model.toLowerCase().contains(entry.key.toLowerCase())) {
            return entry.value
        }
    }
    return categories.get('default', 10000)
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
    
    input:
    tuple path(data_file), path(model_dir), val(dataset), path(cfg_yaml), val(model), val(cache_dir), val(subset_size), val(strategy)

    tag { "${dataset}x${model}x${strategy}" }

    conda { "envs/models/${model}.yml" }

    output:
    tuple(path("results/*/*/*/predictions*.csv"), val(dataset), val(model), val(cache_dir))

    script:
    def apiKeyExport = params.openrouter_api_key ? "export API_KEY='${params.openrouter_api_key}'" : ''
    """
    ${apiKeyExport}
    echo "Subset size of ${dataset}: ${subset_size} (strategy: ${strategy})"

    cache_base="${cache_dir}/models/predictions/${model}/${dataset}/${subset_size}_${strategy}"
    results_dir=./results/${model}/${dataset}/${subset_size}_${strategy}

    mkdir -p \${results_dir}

    if ! ls \${cache_base}/predictions*.csv 1>/dev/null 2>&1; then
        mkdir -p \${cache_base}
        bash "${model_dir}/src/run.sh" \
            "${data_file}" \
            \${results_dir}/predictions.csv \
            "${cfg_yaml}" \
            "${cache_dir}/models"
        cp \${results_dir}/predictions*.csv \${cache_base}/
    else
        cp \${cache_base}/predictions*.csv \${results_dir}/
        echo "Cell type prediction already exists for ${dataset} x ${model} (subset ${subset_size}, strategy ${strategy})."
        echo "To rerun, delete: \${cache_base}/predictions*.csv"
    fi
    """
}

process evaluate_annotation {
    container params.container_image
    conda 'envs/evaluation.yml'
    errorStrategy 'ignore'

    input:
    tuple path(predictions), val(dataset), val(model), val(cache_dir), val(apply_ontology_mapping)

    tag { "${dataset}x${model}" }

    publishDir(path: 'results/', mode: 'copy', overwrite: true)

    output:
    path "*.json"
    path "**/*.pdf", optional: true

    script:
    """
    pred_files=\$(echo ${predictions} | tr ' ' ',')
    evaluate.py --pred "\$pred_files" --out ${dataset}_${model}.json --cache_dir ${cache_dir} --apply_ontology_mapping ${apply_ontology_mapping} --max-unmapped-ratio 0.99
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
        def zenodo_sizes = params.models.collect { resolveSubsetSize(it, params.subset_categories) }.unique()
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
    if (hasOpenRouter && !params.openrouter_api_key) {
        log.warn "OpenRouter model(s) selected but no API key configured. Set the key via one of:"
        log.warn "  1. export OPENROUTER_API_KEY=sk-or-v1-...  (recommended, read by nextflow.config)"
        log.warn "  2. --openrouter_api_key VALUE  on the nextflow command line"
    }

    Channel
        .fromList(params.models)
        .map { model -> tuple(model, file("models/${model}")) }
        .set { prepare_input }

    prepared_models = prepare_model(prepare_input)

    Channel
        .fromList(params.models)
        .map { model -> tuple(model, resolveSubsetSize(model, params.subset_categories)) }
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
            tuple(data_file, model_dir, dataset, file("datasets/${dataset}.yml"), model_name, file(params.cache_dir), data_subset_size, data_strategy)
        }
        .set { model_input }

    predictions = run_model(model_input)

    // --- Evaluation ---

    Channel.value(params.apply_ontology_mapping)
            .set { apply_mapping_ch }

    predictions_with_flag = predictions.combine(apply_mapping_ch)
                                     .map { pred, dataset, model, cache_dir, apply_flag ->
                                         tuple(pred, dataset, model, cache_dir, apply_flag)
                                     }

    eval_out = evaluate_annotation(predictions_with_flag)

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
        println "  ONTOLOGY MAPPING WARNING: New cell type mappings detected!"
        println "  Please check: ${warningFile.toString()}"
        println "  ACTION REQUIRED: Review and verify mappings before using results!"
        println "=" * 70
        println ""

        def timestamp = new Date().format('yyyy-MM-dd_HH-mm-ss')
        def archivedWarningFile = file("${params.cache_dir}/ONTOLOGY_MAPPING_WARNING_${timestamp}.txt")
        warningFile.renameTo(archivedWarningFile)
        println "Warning file archived as: ${archivedWarningFile.toString()}"
    }

    def reviewFile = file("${params.cache_dir}/ONTOLOGY_REVIEW_PENDING.txt")
    if (reviewFile.exists()) {
        println ""
        println "=" * 70
        println "  ONTOLOGY REVIEW PENDING: One or more eval tasks were skipped"
        println "  because >10% of predictions had unreviewed mappings."
        println "  Details: ${reviewFile.toString()}"
        println "  ACTION: Review mappings, set 'reviewed=True', then run with -resume."
        println "=" * 70
        println ""

        def reviewTimestamp = new Date().format('yyyy-MM-dd_HH-mm-ss')
        def archivedReviewFile = file("${params.cache_dir}/ONTOLOGY_REVIEW_PENDING_${reviewTimestamp}.txt")
        reviewFile.renameTo(archivedReviewFile)
        println "Pending-review file archived as: ${archivedReviewFile.toString()}"
    }

    println "Pipeline completed at: ${new Date()}"
    if (workflow.success) {
        println "Pipeline completed successfully!"
    } else {
        println "Pipeline completed with errors"
    }
}
