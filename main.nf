nextflow.enable.dsl=2

process download {
    conda 'python=3.11 tqdm requests anndata pyyaml'

    input:
    tuple path(cfg_dir), val(data_cache)

    output:
    path "data/*.h5ad"

    script:
    """
    download.py \
        --dataset   ${params.dataset} \
        --cfg-dir   ${cfg_dir} \
        --out       data/${params.dataset}.h5ad \
        --data_cache ${data_cache}
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
    conda 'pandas numpy scikit-learn matplotlib seaborn'

    input:
    tuple path(predictions), val(dataset), val(model)

    output:
    path "results/*/*/evaluation.json"

    script:
    """
    mkdir -p results/${dataset}/${model}
    evaluate.py --pred ${predictions} --out results/${dataset}/${model}/ 
    """
}

workflow {
    dataset = download(tuple(file('datasets'), params.data_cache))
    
    // Run the model (all models should output predictions.csv)
    predictions = run_model(dataset.map { data_file ->
        tuple(data_file, file("models/${params.model}"), params.dataset, params.model)
    })

    evaluate_annotation(predictions.map { pred ->
        tuple(pred, params.dataset, params.model)
    })
}
