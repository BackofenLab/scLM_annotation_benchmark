nextflow.enable.dsl=2

process download {
    conda 'envs/download.yml'

    input:
    tuple val(dataset), val(cache_dir), path(cfg_yaml)

    output:
    tuple(path("data/*.h5ad"), val(dataset), path(cfg_yaml), val(cache_dir))

    script:
    """
    set -euo pipefail
    # Call the script via PATH but force the *Conda* python interpreter
    download.py \
      --dataset   "${dataset}" \
      --cfg_yaml  "${cfg_yaml}" \
      --out       "data/${dataset}.h5ad" \
      --cache_dir "${cache_dir}/download"
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
    tuple path(data_file), path(model_dir), val(dataset), path(cfg_yaml), val(model), val(cache_dir)

    conda { "envs/models/${model}.yml" }

    output:
    tuple(path("results/*/*/predictions*.csv"), val(dataset), val(model), val(cache_dir))

    script:
    """
    mkdir -p results/${dataset}/${model}
    mkdir -p ${cache_dir}/models
    bash ${model_dir}/src/run.sh ${data_file} ./results/${dataset}/${model}/predictions.csv ${cfg_yaml} ${cache_dir}/models
    """
}

process evaluate_annotation {
    conda 'envs/evaluation.yml'

    input:
    tuple path(predictions), val(dataset), val(model), val(cache_dir)

    publishDir(path: 'results/', mode: 'copy', overwrite: 'false')

    output:
    path "*.json"

    script:
    """
    # Convert the prediction files to a comma-separated string
    pred_files=\$(echo ${predictions} | tr ' ' ',')
    evaluate.py --pred "\$pred_files" --out ${dataset}_${model}.json --cache_dir ${cache_dir}/evaluation
    """
}

process combine_results {
    conda 'envs/evaluation.yml'

    publishDir(path: 'results/', mode: 'copy', overwrite: 'false')

    input:
    path(json_path)

    output:
    tuple path("aggregated*.csv"), path("aggregated_plot*.png")

    script:
    """
    # Aggregate results:
    aggregate.py --input ${json_path} --output ./
    """
}

workflow {
    Channel
        .from(params.datasets)
        .map { dataset -> tuple(dataset, file(params.cache_dir), file("datasets/${dataset}.yml")) }
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
        .map { data_file, dataset, cfg_yaml, cache_dir, model, model_dir ->
            tuple(data_file, model_dir, dataset, cfg_yaml, model, cache_dir)
        }
        .set { model_input }
    
    // Run the model (all models should output predictions.csv)
    predictions = run_model(model_input)

    jsons = evaluate_annotation(predictions)

    combine_results(jsons.collect())
}

workflow.onComplete {
    // Check for ontology mapping warnings
    def warningFile = file("${params.cache_dir}/ONTOLOGY_MAPPING_WARNING.txt")
    if (warningFile.exists()) {
        println ""
        println "🚨" * 50
        println "⚠️  ONTOLOGY MAPPING WARNING: New cell type mappings detected!"
        println "📄 Please check: ${warningFile.toString()}"
        println "⚠️  ACTION REQUIRED: Review and verify mappings before using results!"
        println "🚨" * 50
        println ""
    }
    
    // Standard completion message
    println "Pipeline completed at: ${new Date()}"
    if (workflow.success) {
        println "✅ Pipeline completed successfully!"
    } else {
        println "❌ Pipeline completed with errors"
    }
}
