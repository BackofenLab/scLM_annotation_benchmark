nextflow.enable.dsl=2

process download {
    conda 'python=3.11 tqdm requests anndata pyyaml'

    input:
    tuple path(cfg_dir), val(dataset), val(data_cache)

    output:
    tuple(path("data/*.h5ad"), val(dataset))

    script:
    """
    download.py \
        --dataset   ${dataset} \
        --cfg-dir   ${cfg_dir} \
        --out       data/${dataset}.h5ad \
        --data_cache ${data_cache}
    """
}

process run_model {
    conda 'git'

    input:
    tuple path(data_file), path(model_dir), val(dataset), val(model)

    output:
    tuple(path("results/*/*/predictions.csv"), val(dataset), val(model))  // Wildcard pattern for multiple output types

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
    Channel
        .from(params.datasets)
        .map { dataset -> tuple(file('datasets'), dataset, params.data_cache) }
        .set { download_input }

    dataset = download(download_input)
    
    dataset
        .combine(Channel.from(params.models))
        .map { data_file, dataset, model ->
            tuple(data_file, file("models/${model}"), dataset, model)
        }
        .set { model_input }
    // Run the model (all models should output predictions.csv)
    predictions = run_model(model_input)

    evaluate_annotation(predictions)
}
