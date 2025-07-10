nextflow.enable.dsl=2

process download {
    output:
    path "data/*/*.h5ad"

    script:
    """
    download.py --dataset ${params.dataset} --subtype ${params.subtype} --out data/${params.dataset}/${params.subtype}.h5ad
    """
}

process embed {
    input:
    tuple path(data_file), path(model_dir), val(dataset), val(model)

    output:
    path "results/*/*/embedding.npy"

    script:
    """
    mkdir -p results/${dataset}/${model}
    bash ${model_dir}/src/run.sh ${data_file} ./results/${dataset}/${model}/embedding.npy
    """
}

process annotate {
    input:
    path "results/${dataset}/${model}/embedding.npy"
    path "data/${dataset}.h5ad"

    output:
    path "results/${dataset}/${model}/predictions.csv"

    script:
    """
    python bin/annotate.py --embedding results/${dataset}/${model}/embedding.npy --adata data/${dataset}.h5ad --out results/${dataset}/${model}/predictions.csv
    """
}

process evaluate_annotation {
    input:
    path "results/${dataset}/${model}/predictions.csv"
    path "data/${dataset}.h5ad"

    output:
    path "results/${dataset}/${model}/evaluation.json"

    script:
    """
    python bin/evaluate.py --pred results/${dataset}/${model}/predictions.csv --adata data/${dataset}.h5ad --out results/${dataset}/${model}/evaluation.json
    """
}

workflow {
    Channel.of([params.dataset, params.subtype]).set{dataset_input}
    dataset = download()
    dataset
        .map { data_file ->
            tuple(data_file, file("3_scGenerativeCapabilities/${params.model}"), params.dataset, params.model)
        }
        .set { embed_input }
    embed(embed_input)

}
