// Annotation: prepare the data, run the methods and map their answers to the Cell Ontology.
// Runs whose predictions are already in the prediction store are not run again.

include {
    absolutePath; datasetConfig; methodSpec; subsetSizes; predictionFiles; predictionsDir;
    storedPredictions; orthologTable; mappingsVersion
} from '../modules/functions'
include {
    download; preprocess; subset; fetch_test_set; orthologize; build_context; fetch_context;
    fetch_singler_reference
} from '../modules/data'
include { prepare_method; constrain_labels; run_method; run_method_with_key } from '../modules/methods'
include { map_predictions } from '../modules/evaluation'

def usesCensusReference(String dataset) {
    return (datasetConfig(dataset).singler_reference_source ?: 'celldex').toString() == 'cellxgene'
}

def checkZenodoFiles(List names) {
    def available = new org.yaml.snakeyaml.Yaml()
        .load(new File("${projectDir}/datasets/zenodo_manifest.yml").text).files.keySet()
    def missing = names.findAll { !available.contains(it) }
    if (missing) {
        error "Not in datasets/zenodo_manifest.yml: ${missing.join(', ')}. " +
              "Build these files locally (--download_from_zenodo false, " +
              "--fetch_context_from_zenodo false) or add them to the record."
    }
}

workflow ANNOTATE {
    take:
    append    // whether answers that are new are added to the mapping table

    main:
    def cache_dir = absolutePath(params.cache_dir)
    def mappings = absolutePath(params.mappings)
    def manifest = file("${projectDir}/datasets/zenodo_manifest.yml")
    def strategy = params.subset_strategy

    // Every run is a method on a dataset at a test set size.
    def runs = []
    params.models.each { method ->
        params.datasets.each { dataset ->
            subsetSizes(method).each { size -> runs << [method, dataset, size] }
        }
    }
    def stored = runs.findAll { method, dataset, size -> storedPredictions(method, dataset, size) }
    def pending = runs - stored
    log.info "${runs.size()} runs: ${stored.size()} with stored predictions, ${pending.size()} to run"

    def needs = { run, what -> methodSpec(run[0]).needs[what] as boolean }
    def test_set_keys = pending.collect { it[1..2] }.unique()
    def ortholog_keys = pending
        .findAll { needs(it, 'human_genes') && orthologTable(it[1]) }.collect { it[1..2] }.unique()
    def context_datasets = pending.findAll { needs(it, 'clustering_context') }.collect { it[1] }.unique()
    def reference_datasets = pending
        .findAll { needs(it, 'singler_reference') && usesCensusReference(it[1]) }
        .collect { it[1] }.unique()
    def constraint_keys = pending.findAll { needs(it, 'label_constraint') }.collect { it[1..2] }.unique()

    def fetch_contexts = context_datasets && params.fetch_context_from_zenodo
    def build_contexts = context_datasets && !params.fetch_context_from_zenodo
    def build_datasets = (params.download_from_zenodo ? [] : test_set_keys.collect { it[0] }) +
        (build_contexts ? context_datasets : [])
    checkZenodoFiles(
        (params.download_from_zenodo
            ? test_set_keys.collect { "${it[0]}_sub${it[1]}_${strategy}.h5ad".toString() } : []) +
        (fetch_contexts ? context_datasets.collect { "${it}_context.h5ad".toString() } : []))

    // Full datasets, only where a test set or a clustering sample is built locally.
    raw = Channel.fromList(build_datasets.unique())
        .branch { dataset ->
            local: datasetConfig(dataset).path
            remote: true
        }
    downloaded = download(raw.remote.map { dataset -> tuple(dataset, file("${projectDir}/datasets/${dataset}.yml")) })
    local_files = raw.local.map { dataset -> tuple(dataset, file("${projectDir}/${datasetConfig(dataset).path}")) }
    preprocessed = preprocess(
        downloaded.mix(local_files).map { dataset, raw_file ->
            def cfg = datasetConfig(dataset)
            def map_labels = cfg.containsKey('apply_ontology_mapping') ? cfg.apply_ontology_mapping : true
            tuple(dataset, raw_file, file("${projectDir}/datasets/${dataset}.yml"), cfg.annotation_key, map_labels)
        },
        cache_dir, mappings)

    // Test sets.
    if (params.download_from_zenodo) {
        test_sets = fetch_test_set(Channel.fromList(test_set_keys).map { dataset, size ->
            tuple(dataset, size, datasetConfig(dataset).annotation_key, manifest) })
    } else {
        test_sets = subset(
            Channel.fromList(test_set_keys).combine(preprocessed, by: 0).map { dataset, size, file ->
                tuple(dataset, size, file, datasetConfig(dataset).annotation_key) },
            cache_dir)
    }
    ortholog_sets = orthologize(
        Channel.fromList(ortholog_keys).map { dataset, size -> tuple([dataset, size], file(orthologTable(dataset))) }
            .combine(test_sets.map { dataset, size, file -> tuple([dataset, size], file) }, by: 0)
            .map { key, table, file -> tuple(key[0], key[1], file, table) })

    // Additional inputs of some methods.
    if (fetch_contexts) {
        contexts = fetch_context(Channel.fromList(context_datasets).map { dataset ->
            tuple(dataset, datasetConfig(dataset).annotation_key, manifest) })
    } else {
        contexts = build_context(
            Channel.fromList(build_contexts ? context_datasets : [])
                .combine(preprocessed, by: 0)
                .combine(test_sets.map { dataset, size, file -> tuple(dataset, file) }.groupTuple(), by: 0)
                .map { dataset, full, sets -> tuple(dataset, full, sets, datasetConfig(dataset).annotation_key) })
    }
    references = fetch_singler_reference(Channel.fromList(reference_datasets).map { dataset ->
        def cfg = datasetConfig(dataset)
        tuple(dataset, cfg.singler_reference_id.toString(),
              (cfg.singler_reference_version ?: cfg.census_version ?: 'stable').toString(),
              (cfg.singler_reference_label_key ?: 'cell_type').toString(),
              (cfg.species ?: cfg.organism ?: 'human').toString())
    })
    constraints = constrain_labels(
        Channel.fromList(constraint_keys).map { dataset, size -> tuple([dataset, size], file("${projectDir}/datasets/${dataset}.yml")) }
            .combine(test_sets.map { dataset, size, file -> tuple([dataset, size], file) }, by: 0)
            .map { key, cfg, file -> tuple(key[0], key[1], file, cfg) },
        cache_dir, mappings, mappingsVersion())

    // Each pending run with the inputs it needs. An input that a run does not need is an
    // empty list, which Nextflow stages as no file.
    methods = Channel.fromList(pending.collect { it[0] }.unique())
        .map { method -> tuple(method, file("${projectDir}/models/${method}")) }
        .branch { method, dir ->
            with_setup: new File("${dir}/src/prepare.sh").exists()
            other: true
        }
    prepared = prepare_method(methods.with_setup).mix(methods.other)
    none = Channel.of(tuple('none', []))
    plain = test_sets.map { dataset, size, file -> tuple("${dataset}:${size}:plain".toString(), file) }
    human = ortholog_sets.map { dataset, size, file -> tuple("${dataset}:${size}:human".toString(), file) }

    method_inputs = Channel.fromList(pending)
        .map { method, dataset, size -> tuple(method, [method: method, dataset: dataset, size: size]) }
        .combine(prepared, by: 0)
        .map { method, run, method_dir ->
            def genes = needs([method, run.dataset], 'human_genes') && orthologTable(run.dataset) ? 'human' : 'plain'
            tuple("${run.dataset}:${run.size}:${genes}".toString(), run + [method_dir: method_dir]) }
        .combine(plain.mix(human), by: 0)
        .map { key, run, test_set ->
            tuple(needs([run.method], 'clustering_context') ? run.dataset : 'none', run + [test_set: test_set]) }
        .combine(contexts.mix(none), by: 0)
        .map { key, run, context ->
            def wanted = needs([run.method], 'singler_reference') && usesCensusReference(run.dataset)
            tuple(wanted ? run.dataset : 'none', run + [context: context]) }
        .combine(references.mix(none), by: 0)
        .map { key, run, reference ->
            def wanted = needs([run.method], 'label_constraint')
            tuple(wanted ? "${run.dataset}:${run.size}".toString() : 'none', run + [reference: reference]) }
        .combine(constraints.map { dataset, size, file -> tuple("${dataset}:${size}".toString(), file) }.mix(none), by: 0)
        .map { key, run, constraint ->
            tuple(run.method, run.dataset, run.size, run.method_dir, run.test_set,
                  file("${projectDir}/datasets/${run.dataset}.yml"), run.context, run.reference,
                  constraint, predictionFiles(run.method).join(' ')) }
        .branch { run ->
            with_key: methodSpec(run[0]).needs.api_key
            other: true
        }

    predicted = run_method(method_inputs.other, predictionsDir(), cache_dir)
        .mix(run_method_with_key(method_inputs.with_key, predictionsDir(), cache_dir))
    from_store = Channel.fromList(stored).map { method, dataset, size ->
        tuple(method, dataset, size, storedPredictions(method, dataset, size).collect { file(it) }) }

    map_predictions(predicted.mix(from_store), cache_dir, mappings, mappingsVersion(), append)

    emit:
    mapped = map_predictions.out.mapped
}
