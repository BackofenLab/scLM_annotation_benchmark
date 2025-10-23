nextflow.enable.dsl=2

process download {
    conda 'envs/download.yml'

    input:
    tuple val(dataset), val(cache_dir), path(cfg_yaml)

    // Tag appears after inputs; you can use `dataset`
    tag { dataset }

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
      --cache_dir "${cache_dir}"
    """
}

process extract_cell_types {
    conda 'envs/cell_types.yml'

    input:
        path(h5ad_files)
        path(cfg_files)
        val(cache_dir)  // list of files

    script:
        """
        if [ ! -f ${cache_dir}/cell_types.csv ]; then
            extract_cell_types.py \
                --h5ads ${h5ad_files.join(' ')} \
                --cfgs ${cfg_files.join(' ')} \
                --out ${cache_dir}/cell_types.csv
        else
            echo "Cell type collection exists, if you want to rerun, delete cache/cell_types.csv"
        fi
        """
}

process prepare_model {
    input:
    tuple val(model), path(model_dir)

    // Use the model name in the tag
    tag { model }

    output:
    tuple val(model), path(model_dir)

    script:
    """
    # check if the model_dir exists
    if [ ! -d "${model_dir}" ]; then
        echo "Model directory ${model_dir} does not exist"
        exit 1
    fi
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

    // Same tag format as run_model
    tag { "${dataset}×${model}" }

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

process run_model_throttled {
    input:
    tuple path(data_file), path(model_dir), val(dataset), path(cfg_yaml), val(model), val(cache_dir)

    // Same tag format as run_model
    tag { "${dataset}×${model}" }

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

    // Tag with dataset and model for easy tracing
    tag { "${dataset}×${model}" }

    publishDir(path: 'results/', mode: 'copy', overwrite: 'false')

    output:
    path "*.json"

    script:
    """
    # Convert the prediction files to a comma-separated string
    pred_files=\$(echo ${predictions} | tr ' ' ',')
    evaluate.py --pred "\$pred_files" --out ${dataset}_${model}.json --cache_dir ${cache_dir}
    """
}

process combine_results {
    conda 'envs/evaluation.yml'

    publishDir(path: 'results/', mode: 'copy', overwrite: 'false')

    input:
    path(json_path)
    file(datasets)

    output:
    tuple path("aggregated*.csv"), path("aggregated_plot*.png")

    script:
    """
    # Aggregate results (only level 0 for now):
    aggregate.py --input ${json_path} --output ./ --level 0 --datasets ${datasets}
    """
}

workflow {
    Channel
        .fromList(params.datasets)
        .map { dataset -> tuple(dataset, file(params.cache_dir), file("datasets/${dataset}.yml")) }
        .set { download_input }

    download_out = download(download_input)

    Channel.fromPath(params.cache_dir)
            .set { cache_dir_ch }

    download_out.map { h5ad_file, dataset, cfg_yaml, cache_dir -> h5ad_file }
                .collect()
                .set { h5ad_file_ch }

    download_out.map { h5ad_file, dataset, cfg_yaml, cache_dir -> cfg_yaml }
                .collect()
                .set { config_file_ch }

    cell_types = extract_cell_types(h5ad_file_ch, config_file_ch, cache_dir_ch)

    // Prepare models (handles setup like git cloning, etc.)
    Channel
        .fromList(params.models)
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
    
    // Run the models (all models should output predictions.csv)
    // OpenRouter LLMs are request-limited, so we currently run them in a separate lane via run_model_throttled,
    // while all other models can be run in parallel via run_model
    def throttled_in = model_input.filter { t -> (params.throttled_models ?: []).contains(t[4]) }
    def normal_in    = model_input.filter { t -> !(params.throttled_models ?: []).contains(t[4]) }
    predictions_fast = run_model(normal_in)
    predictions_slow = run_model_throttled(throttled_in)
    predictions      = predictions_fast.mix(predictions_slow)

    jsons = evaluate_annotation(predictions)
    Channel
        .fromList(params.datasets)
        .map { dataset -> file("datasets/${dataset}.yml") }
        .set { datasets }

    combine_results(jsons.collect(), datasets)
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
        
        // Archive the warning file with timestamp to prevent showing it again
        def timestamp = new Date().format('yyyy-MM-dd_HH-mm-ss')
        def archivedWarningFile = file("${params.cache_dir}/ONTOLOGY_MAPPING_WARNING_${timestamp}.txt")
        warningFile.renameTo(archivedWarningFile)
        println "📁 Warning file archived as: ${archivedWarningFile.toString()}"
    }
    
    // Standard completion message
    println "Pipeline completed at: ${new Date()}"
    if (workflow.success) {
        println "✅ Pipeline completed successfully!"
    } else {
        println "❌ Pipeline completed with errors"
    }
}
