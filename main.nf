nextflow.enable.dsl=2

process download {
    conda 'envs/download.yml'

    input:
    tuple val(dataset), val(data_cache), path(cfg_yaml)

    output:
    tuple(path("data/*.h5ad"), val(dataset), path(cfg_yaml))

    script:
    """
    set -euo pipefail
    # Call the script via PATH but force the *Conda* python interpreter
    download.py \
      --dataset   "${dataset}" \
      --cfg_yaml  "${cfg_yaml}" \
      --out       "data/${dataset}.h5ad" \
      --data_cache "${data_cache}"
    """
}

process prepare_model {
    input:
    tuple val(model), path(model_dir)

    output:
    tuple val(model), path(model_dir)

    script:
    """
    # Check if prepare.sh exists and run it if it does
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
    input:
    tuple path(data_file), path(model_dir), val(dataset), path(cfg_yaml), val(model)

    conda { "envs/models/${model}.yml" }

    output:
    tuple(path("results/*/*/predictions.csv"), val(dataset), val(model))

    script:
    """
    mkdir -p results/${dataset}/${model}
    bash ${model_dir}/src/run.sh ${data_file} ./results/${dataset}/${model}/predictions.csv ${cfg_yaml}
    """
}

process evaluate_annotation {
    conda 'envs/evaluation.yml'

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
        .map { dataset -> tuple(dataset, file(params.data_cache), file("datasets/${dataset}.yml")) }
        .set { download_input }

    download_out = download(download_input)
    
    // Prepare models (handles setup like git cloning, etc.)
    Channel
        .from(params.models)
        .map { model -> tuple(model, file("models/${model}")) }
        .set { prepare_input }
    
    prepared_models = prepare_model(prepare_input)
    
    // Combine data with prepared models
    download_out
        .combine(prepared_models)
        .map { data_file, dataset, cfg_yaml, model, model_dir ->
            tuple(data_file, model_dir, dataset, cfg_yaml, model)
        }
        .set { model_input }
    
    // Run the model (all models should output predictions.csv)
    predictions = run_model(model_input)

    evaluate_annotation(predictions)
}
