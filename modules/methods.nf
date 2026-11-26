// Processes that run the annotation methods.

include { containerFor; environmentFor; methodSpec; testSetLabel } from './functions'

// Shell command that writes a dataset's own list of expected cell types to
// expected_cell_types.txt, one name per line; empty without a list.
def listedFile(listed) {
    def quoted = listed.collect { "'" + it.replace("'", "'\\''") + "'" }.join(' ')
    return listed ? "printf '%s\\n' ${quoted} > expected_cell_types.txt" : ''
}

// Shell script of a method run. Prediction files that the store already holds are passed to
// the method, which skips them, and all prediction files are copied to the store afterwards.
def methodScript(method, method_dir, dataset, size, test_set, dataset_cfg, context, reference,
                 constraint, cluster_files, expected, listed, store, cache_dir) {
    def exports = [
        context    ? "export CONTEXT_H5AD=\"\$(realpath ${context})\"" : '',
        reference  ? "export SINGLER_REF_H5AD=\"\$(realpath ${reference})\"" : '',
        constraint ? "export LABEL_CONSTRAINT=\"\$(realpath ${constraint})\"" : '',
        methodSpec(method).needs.candidate_types ? "export CANDIDATE_TYPES_FILE=\"${params.candidate_types}\"" : '',
        cluster_files ? "export CLUSTERS_CSV=\"\$(realpath ${dataset}_${size}.clusters.csv)\"" : '',
        cluster_files ? "export MARKERS_CSV=\"\$(realpath ${dataset}_${size}.markers.csv)\"" : '',
        listedFile(listed),
        listed ? "export EXPECTED_TYPES_FILE=\"\$(realpath expected_cell_types.txt)\"" : '',
    ].findAll().join('\n    ')
    """
    set -euo pipefail
    store="${store}/${method}/${dataset}/${testSetLabel(size)}"
    mkdir -p out "\$store" "${cache_dir}/models"
    export METHOD_DIR="\$(realpath ${method_dir})"
    ${exports}
    for f in ${expected}; do
        if [ -f "\$store/\$f" ]; then cp "\$store/\$f" "out/\$f"; fi
    done
    bash "${method_dir}/src/run.sh" "${test_set}" out/predictions.csv "${dataset_cfg}" "${cache_dir}/models"
    for f in ${expected}; do
        cp "out/\$f" "\$store/\$f"
    done
    for f in out/usage_*.csv; do
        if [ -e "\$f" ]; then cp "\$f" "\$store/"; fi
    done
    """
}

// One-time setup of a method, for example fetching its code and weights.
process prepare_method {
    container params.container_image
    conda 'git git-lfs'
    // The two settings of a method can share one src folder.
    maxForks 1
    tag { method }

    input:
    tuple val(method), path(method_dir)

    output:
    tuple val(method), path(method_dir)

    script:
    """
    set -euo pipefail
    if [ -f "${method_dir}/src/prepare.sh" ]; then
        cd "${method_dir}/src"
        bash prepare.sh
    fi
    """
}

// The labels that count for the expected cell types of a test set, for constrained baselines.
// The expected cell types are those of the labels, or the dataset's own list where it applies.
process constrain_labels {
    container params.container_image
    conda "${projectDir}/envs/evaluation.yml"
    tag { "${dataset} ${size}" }

    input:
    tuple val(dataset), val(size), path(test_set), path(dataset_cfg), val(listed)
    val cache_dir
    val mappings
    val mappings_version

    output:
    tuple val(dataset), val(size), path("${dataset}_sub${size}.constraint.tsv")

    script:
    """
    set -euo pipefail
    ${listedFile(listed)}
    constrain_labels.py --input ${test_set} --config ${dataset_cfg} --cache_dir ${cache_dir} \\
        --mappings ${mappings} --out ${dataset}_sub${size}.constraint.tsv \\
        ${listed ? '--expected expected_cell_types.txt' : ''}
    """
}

process run_method {
    container { containerFor(method) }
    conda { environmentFor(method) }
    // Building an environment can fail when several tasks need the same one at once.
    errorStrategy { task.attempt <= 3 ? 'retry' : 'ignore' }
    maxRetries 3
    tag { "${dataset} ${method} ${size}" }

    input:
    tuple val(method), val(dataset), val(size), path(method_dir), path(test_set),
          path(dataset_cfg), path(context), path(reference), path(constraint),
          path(cluster_files), val(expected), val(listed)
    val store
    val cache_dir

    output:
    tuple val(method), val(dataset), val(size), path("out/predictions*.csv")

    script:
    methodScript(method, method_dir, dataset, size, test_set, dataset_cfg, context, reference,
                 constraint, cluster_files, expected, listed, store, cache_dir)
}

// The same for methods that call the OpenRouter API. The key is a Nextflow secret, so it is
// written neither to the logs nor to the task scripts.
process run_method_with_key {
    container { containerFor(method) }
    conda { environmentFor(method) }
    secret 'OPENROUTER_API_KEY'
    tag { "${dataset} ${method} ${size}" }

    input:
    tuple val(method), val(dataset), val(size), path(method_dir), path(test_set),
          path(dataset_cfg), path(context), path(reference), path(constraint),
          path(cluster_files), val(expected), val(listed)
    val store
    val cache_dir

    output:
    tuple val(method), val(dataset), val(size), path("out/predictions*.csv")

    script:
    'export API_KEY="$OPENROUTER_API_KEY"\n' + methodScript(
        method, method_dir, dataset, size, test_set, dataset_cfg, context, reference, constraint,
        cluster_files, expected, listed, store, cache_dir)
}
