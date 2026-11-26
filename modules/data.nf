// Processes that provide the test sets and the additional inputs of some methods. Data built
// here are kept in a cache folder named after the fingerprint of their settings (settings_id).

process download {
    container params.container_image
    conda "${projectDir}/envs/download.yml"
    storeDir { "${params.cache_dir}/datasets/raw/${settings_id}" }
    tag { dataset }

    input:
    tuple val(dataset), path(dataset_cfg), val(settings_id)

    output:
    tuple val(dataset), path("${dataset}_raw.h5ad")

    script:
    """
    set -euo pipefail
    download.py --dataset "${dataset}" --cfg_yaml "${dataset_cfg}" --out "${dataset}_raw.h5ad"
    """
}

process preprocess {
    container params.container_image
    conda "${projectDir}/envs/download.yml"
    storeDir { "${params.cache_dir}/datasets/preprocessed/${settings_id}" }
    // The gene annotation service that maps symbols to Ensembl ids fails now and then.
    errorStrategy 'retry'
    maxRetries 2
    tag { dataset }

    input:
    tuple val(dataset), path(raw_file), path(dataset_cfg), val(annotation_key), val(map_labels),
          val(labels_optional), val(organism), val(biomart_species), val(settings_id)
    val cache_dir
    val mappings
    val append

    output:
    tuple val(dataset), path("${dataset}_pp.h5ad")

    script:
    """
    set -euo pipefail
    preprocess.py \\
        --input "${raw_file}" \\
        --output "${dataset}_pp.h5ad" \\
        --config "${dataset_cfg}" \\
        --summary "${dataset}_pp_summary.txt" \\
        --cache_dir "${cache_dir}" \\
        --mappings "${mappings}" \\
        --annotation_key "${annotation_key}" \\
        --apply_ontology_mapping ${map_labels} \\
        --append_new_labels ${append} \\
        --labels_optional ${labels_optional} \\
        --organism "${organism}" \\
        --species "${biomart_species}" \\
        --dataset "${dataset}"
    """
}

process subset {
    container params.container_image
    conda "${projectDir}/envs/download.yml"
    storeDir { "${params.cache_dir}/datasets/subset/${settings_id}" }
    tag { "${dataset} ${size}" }

    input:
    tuple val(dataset), val(size), path(preprocessed), val(annotation_key), val(settings_id)
    val cache_dir

    output:
    tuple val(dataset), val(size), path("${dataset}_sub${size}_${params.subset_strategy}.h5ad")

    script:
    """
    set -euo pipefail
    subset.py \\
        --input "${preprocessed}" \\
        --output "${dataset}_sub${size}_${params.subset_strategy}.h5ad" \\
        --annotation_key "${annotation_key}" \\
        --subset_size ${size} \\
        --dataset "${dataset}" \\
        --qc_dir "${cache_dir}/subset_qc/${dataset}/sub${size}_${params.subset_strategy}" \\
        --min_cells_per_type ${params.min_cells_per_type} \\
        --strategy "${params.subset_strategy}"
    """
}

// Test set from the Zenodo record listed in datasets/zenodo_manifest.yml.
process fetch_test_set {
    container params.container_image
    conda "${projectDir}/envs/download.yml"
    storeDir "${params.cache_dir}/datasets/subset"
    tag { "${dataset} ${size}" }

    input:
    tuple val(dataset), val(size), val(annotation_key), path(manifest)

    output:
    tuple val(dataset), val(size), path("${dataset}_sub${size}_${params.subset_strategy}.h5ad")

    script:
    """
    set -euo pipefail
    fetch_zenodo_subset.py \\
        --manifest "${manifest}" \\
        --dataset "${dataset}" \\
        --subset_size ${size} \\
        --strategy "${params.subset_strategy}" \\
        --annotation_key "${annotation_key}" \\
        --output "${dataset}_sub${size}_${params.subset_strategy}.h5ad"
    """
}

// Test set with its genes replaced by their one-to-one human orthologs, kept beside the test set
// (no settings_id for a test set from the Zenodo record).
process orthologize {
    container params.container_image
    conda "${projectDir}/envs/download.yml"
    storeDir { "${params.cache_dir}/datasets/subset" + (settings_id ? "/${settings_id}" : '') }
    tag { "${dataset} ${size}" }

    input:
    tuple val(dataset), val(size), path(test_set), path(orthologs), val(settings_id)

    output:
    tuple val(dataset), val(size), path("${test_set.baseName}_human_orthologs.h5ad")

    script:
    """
    set -euo pipefail
    orthologize.py --input "${test_set}" --orthologs "${orthologs}" \\
        --output "${test_set.baseName}_human_orthologs.h5ad"
    """
}

// Clustering sample for the methods that annotate clusters: cells of the full dataset, up to
// context_max_size, that include every cell of the dataset's test sets. The table of the cells
// per type in the dataset, the test sets and the sample is kept beside it.
process build_context {
    container params.container_image
    conda "${projectDir}/envs/download.yml"
    storeDir { "${params.cache_dir}/datasets/context/${settings_id}" }
    tag { dataset }

    input:
    tuple val(dataset), path(preprocessed), path(test_sets), val(annotation_key), val(settings_id)

    output:
    tuple val(dataset), path("${dataset}_context.h5ad"), emit: context
    path("${dataset}_context_qc.csv"), emit: qc

    script:
    """
    set -euo pipefail
    build_context.py \\
        --full ${preprocessed} \\
        --subsets ${test_sets} \\
        --output ${dataset}_context.h5ad \\
        --annotation_key "${annotation_key}" \\
        --max_size ${params.context_max_size} \\
        --min_cells_per_type ${params.context_min_cells_per_type} \\
        --qc_csv ${dataset}_context_qc.csv
    """
}

// Clustering sample from the Zenodo record.
process fetch_context {
    container params.container_image
    conda "${projectDir}/envs/download.yml"
    storeDir "${params.cache_dir}/datasets/context"
    tag { dataset }

    input:
    tuple val(dataset), val(annotation_key), path(manifest)

    output:
    tuple val(dataset), path("${dataset}_context.h5ad")

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

// Reference for SingleR from CELLxGENE Census, for a dataset that names one.
process fetch_singler_reference {
    container params.container_image
    conda "${projectDir}/envs/download.yml"
    storeDir "${params.cache_dir}/datasets/reference"
    tag { dataset }

    input:
    tuple val(dataset), val(census_id), val(census_version), val(label_key), val(species)

    output:
    tuple val(dataset), path("${dataset}_singler_ref.h5ad")

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

// Clusters and their marker genes for methods that annotate clusters. A drawn test set is
// clustered within the clustering sample that contains it; all cells of a dataset are clustered
// as they are. A dataset with cluster_key in its yml brings its own clusters.
process cluster_cells {
    container params.container_image
    conda "${projectDir}/envs/clustering.yml"
    tag { "${dataset} ${size}" }
    publishDir "${params.outdir}/clusters/${dataset}", mode: 'copy'

    input:
    tuple val(dataset), val(size), path(cells), path(data), val(batch_key), val(cluster_key)

    output:
    tuple val(dataset), val(size), path("${dataset}_${size}.clusters.csv"),
          path("${dataset}_${size}.markers.csv"), path("${dataset}_${size}.json")

    script:
    def options = [
        "--n_top_genes ${params.cluster_n_top_genes}",
        params.cluster_scale ? '--scale' : '--no-scale',
        batch_key ? "--batch_key '${batch_key}'" : '',
        params.cluster_integration != 'none' ? "--integration ${params.cluster_integration}" : '',
        cluster_key ? "--cluster_key '${cluster_key}'" : '',
    ].findAll().join(' ')
    """
    set -euo pipefail
    cluster_cells.py --data ${data ?: cells} --cells ${cells} --out ${dataset}_${size} \\
        --resolution ${params.cluster_resolution} --n_pcs ${params.cluster_n_pcs} \\
        --n_neighbors ${params.cluster_n_neighbors} --n_marker_genes ${params.cluster_marker_genes} \\
        --rank_method ${params.cluster_rank_method} --seed ${params.cluster_seed} ${options}
    """
}
