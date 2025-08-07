nextflow.enable.dsl=2

process download {
    conda 'python=3.11 tqdm requests anndata pyyaml'

    input:
    tuple path(cfg_dir), path(data_root)

    output:
    path "data/*sub*.h5ad"

    script:
    """
    download.py \
        --dataset   ${params.dataset} \
        --cfg-dir   ${cfg_dir} \
        --out       data/${params.dataset}.h5ad \
        --data-root ${data_root}
    """
}

process run_model {
    conda 'git'

    input:
    tuple path(data_file), path(model_dir), val(dataset), val(model)

    output:
    path "results/*/*/predictions.csv"  // Wildcard pattern for multiple output types

    script:
    """
    mkdir -p results/${dataset}/${model}
    bash ${model_dir}/src/run.sh ${data_file} ./results/${dataset}/${model}/predictions.csv
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
    python bin/evaluate.py --pred results/${dataset}/${model}/predictions.csv --adata data/${dataset}.h5ad --out results/${dataset}/${model}
    """
}

workflow {
    dataset = download(tuple(file('datasets'), file(params.data_root)))
    
    // Run the model (all models should output predictions.csv)
    predictions = run_model(dataset.map { data_file ->
        tuple(data_file, file("models/${params.model}"), params.dataset, params.model)
    })
}
