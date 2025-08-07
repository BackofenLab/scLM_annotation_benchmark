nextflow.enable.dsl=2

process download {
    conda 'python=3.11 tqdm requests anndata pyyaml'

    input:
    path cfg_dir

    output:
    path "data/*.h5ad"

    script:
    """
    download.py \
        --dataset   ${params.dataset} \
        --cfg-dir   ${cfg_dir} \
        --out       data/${params.dataset}.h5ad \
        --data-root cache/data
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
    dataset = download(file('datasets'))
    
    // Run the model (all models should output predictions.csv)
    predictions = run_model(dataset.map { data_file ->
        tuple(data_file, file("models/${params.model}"), params.dataset, params.model)
    })
}
