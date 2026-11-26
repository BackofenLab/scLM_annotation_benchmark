// Helpers that read the dataset and method configurations.

import org.yaml.snakeyaml.Yaml

def loadYaml(path) {
    def file = new File(path.toString())
    return file.exists() ? (new Yaml().load(file.text) ?: [:]) : [:]
}

def absolutePath(path) {
    def file = new File(path.toString())
    return file.isAbsolute() ? file.path : new File(launchDir.toString(), path.toString()).path
}

def datasetConfig(String dataset) {
    def path = "${projectDir}/datasets/${dataset}.yml"
    if (!new File(path).exists()) {
        error "Dataset '${dataset}': ${path} not found"
    }
    return loadYaml(path)
}

// The column of a dataset's labels: its annotation_key, else cell_type, as in the wrappers.
def labelKey(String dataset) {
    return (datasetConfig(dataset).annotation_key ?: 'cell_type').toString()
}

// The method's description in models/<method>/method.yml, with 'needs' always present.
def methodSpec(String method) {
    def dir = "${projectDir}/models/${method}"
    if (!new File("${dir}/src/run.sh").exists()) {
        error "Method '${method}': ${dir}/src/run.sh not found"
    }
    def spec = loadYaml("${dir}/method.yml")
    spec.needs = spec.needs ?: [:]
    return spec
}

// Test set sizes at which a method runs: the requested sizes up to the method's limit. 'all'
// stands for all cells of a dataset, which a method with a limit does not get.
def runSizes(String method, List sizes) {
    def limit = methodSpec(method).max_subset_size
    return limit ? sizes.findAll { it != 'all' && (it as int) <= (limit as int) } : sizes
}

// Name of a test set in folder names: <size>_<strategy>, or 'all' for all cells.
def testSetLabel(size) {
    return size == 'all' ? 'all' : "${size}_${params.subset_strategy}".toString()
}

// Prediction files of one run of a method: one file, or one per served model.
def predictionFiles(String method) {
    def served = methodSpec(method).served_models
    if (!served) {
        return ['predictions.csv']
    }
    def models = loadYaml("${projectDir}/models/${method}/${served}").models ?: []
    if (!models) {
        error "Method '${method}': no 'models' list in models/${method}/${served}"
    }
    return models.collect { "predictions_${it.toString().replace('/', '--')}.csv".toString() }
}

def predictionsDir() {
    return absolutePath(params.predictions_dir ?: "${params.outdir}/predictions")
}

// The folder of a run in the prediction store.
def storeFolder(String method, String dataset, size) {
    return "${predictionsDir()}/${method}/${dataset}/${testSetLabel(size)}".toString()
}

// The stored prediction files of a run, or null if one of them is missing.
def storedPredictions(String method, String dataset, size) {
    def files = predictionFiles(method).collect { new File(storeFolder(method, dataset, size), it) }
    return files.every { it.exists() } ? files.collect { it.path } : null
}

// The note at the end of a run on the tasks that failed in every attempt and were left out.
def reportLeftOut() {
    def left_out = workflow.stats.ignoredCount
    if (left_out) {
        log.warn "${left_out} task(s) failed in every attempt and were left out, so their " +
                 "results are missing. .nextflow.log names them."
    }
}

// The species of a dataset in three spellings: species (human or mouse), as the methods read it;
// organism, as stored with the data; and the Ensembl BioMart name, under which gene symbols are
// looked up. The configuration's species sets all three. organism and preprocessing.species,
// where given, set theirs directly, for example for other organisms.
def speciesNames(String dataset) {
    def cfg = datasetConfig(dataset)
    def organisms = [human: 'Homo sapiens', mouse: 'Mus musculus']
    def given = cfg.organism?.toString()?.trim()
    def species = cfg.species?.toString()?.trim()?.toLowerCase() ?:
        (organisms.find { it.value.equalsIgnoreCase(given ?: '') }?.key ?:
         (given ?: 'human').toLowerCase())
    def organism = given ?: organisms[species]
    if (!organism) {
        error "datasets/${dataset}.yml: species '${species}' is neither human nor mouse; set " +
              "organism and preprocessing.species as well"
    }
    def words = organism.toLowerCase().tokenize()
    def biomart = (cfg.preprocessing?.species ?: words[0][0] + words[-1]).toString()
    return [species: species, organism: organism, biomart: biomart]
}

// Table of one-to-one human orthologs for the dataset's organism, or null for human data.
def orthologTable(String dataset) {
    def table = (params.ortholog_tables ?: [:])[speciesNames(dataset).organism.toLowerCase()]
    return table ? "${projectDir}/${table}" : null
}

def containerFor(String method) {
    return methodSpec(method).container == 'r' ? params.r_container_image : params.container_image
}

def environmentFor(String method) {
    return "${projectDir}/envs/models/${method}.yml"
}

// Content hash of the mapping table, so that a changed table re-runs the steps that read it.
def mappingsVersion() {
    def file = new File(absolutePath(params.mappings))
    return file.exists() ? file.bytes.md5() : 'none'
}

// The data file of a dataset that is not downloaded (path, absolute or relative to the
// repository), or null.
def localDataFile(String dataset) {
    def path = datasetConfig(dataset).path?.toString()
    if (!path) {
        return null
    }
    return new File(path).isAbsolute() ? new File(path) : new File("${projectDir}/${path}")
}

// The settings that the preprocessed data of a dataset depend on: where the data come from (a
// local file by path, size and modification time) and how they are preprocessed.
def dataSettings(String dataset, boolean annotation_mode) {
    def cfg = datasetConfig(dataset)
    def local = localDataFile(dataset)
    def names = speciesNames(dataset)
    return [
        source: local ? [path: local.path, size: local.length(), modified: local.lastModified()] :
            cfg.subMap(['source', 'census_dataset_id', 'census_version', 'url']),
        preprocessing: cfg.preprocessing ?: [:],
        annotation_key: labelKey(dataset),
        apply_ontology_mapping: cfg.containsKey('apply_ontology_mapping') ? cfg.apply_ontology_mapping : true,
        labels_optional: annotation_mode && !cfg.annotation_key,
        organism: names.organism,
        species: names.biomart,
    ]
}

// A short fingerprint of settings. Cached data are kept in a folder named after the fingerprint
// of the settings they depend on, so that changed settings build new files.
def fingerprint(settings) {
    return groovy.json.JsonOutput.toJson(settings).md5().take(12)
}
