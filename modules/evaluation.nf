// Processes that map the answers to the Cell Ontology and score them.

process map_predictions {
    container params.container_image
    conda "${projectDir}/envs/evaluation.yml"
    // Tasks that add rows to the mapping table must not run at the same time.
    maxForks 1
    tag { "${dataset} ${method} ${size}" }
    publishDir "${params.outdir}/annotations/${dataset}/${method}/sub${size}", mode: 'copy',
        pattern: 'mapped/*_annotations.*', saveAs: { it.replaceFirst('mapped/', '') }

    input:
    tuple val(method), val(dataset), val(size), path(predictions)
    val cache_dir
    val mappings
    val mappings_version
    val append

    output:
    tuple val(method), val(dataset), val(size), path("mapped/*_mapped.csv"), emit: mapped
    path "mapped/*_annotations.*", emit: annotations

    script:
    """
    set -euo pipefail
    map_predictions.py --pred ${[predictions].flatten().join(',')} --method ${method} --dataset ${dataset} \\
        --out-dir mapped --mappings ${mappings} --cache_dir ${cache_dir} --append ${append}
    """
}

process score_predictions {
    container params.container_image
    conda "${projectDir}/envs/evaluation.yml"
    tag { "${dataset} ${method} ${size}" }
    publishDir "${params.outdir}/evaluation/${dataset}", mode: 'copy',
        saveAs: { it.replaceFirst('scores/', '') }

    input:
    tuple val(method), val(dataset), val(size), path(mapped)
    val cache_dir
    val mappings
    val mappings_version

    output:
    path "scores/*"

    script:
    """
    set -euo pipefail
    evaluate.py --pred ${[mapped].flatten().join(',')} --method ${method} --dataset ${dataset} \\
        --subset-size ${size} --out-dir scores --mappings ${mappings} --cache_dir ${cache_dir} \\
        --min-scoring-coverage ${params.min_scoring_coverage} \\
        --use_text_encoder ${params.use_text_encoder} \\
        --text_encoder_model ${params.text_encoder_model}
    """
}

process combine_results {
    container params.container_image
    conda "${projectDir}/envs/evaluation.yml"
    publishDir params.outdir, mode: 'copy'

    input:
    path score_files

    output:
    path "benchmark_metrics.csv"
    path "per_class_scores.csv", optional: true
    path "confusion_pairs.csv", optional: true
    path "plots/*", optional: true

    script:
    """
    set -euo pipefail
    aggregate.py --input *.json --output .
    plot_results.py --input benchmark_metrics.csv --output plots
    """
}
