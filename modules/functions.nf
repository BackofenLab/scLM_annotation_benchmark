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

// The stored prediction files of a run, or null if one of them is missing.
def storedPredictions(String method, String dataset, size) {
    def dir = "${predictionsDir()}/${method}/${dataset}/${testSetLabel(size)}"
    def files = predictionFiles(method).collect { new File(dir, it) }
    return files.every { it.exists() } ? files.collect { it.path } : null
}

// Table of one-to-one human orthologs for the dataset's organism, or null for human data.
def orthologTable(String dataset) {
    def organism = (datasetConfig(dataset).organism ?: 'Homo sapiens').toString().trim().toLowerCase()
    def table = (params.ortholog_tables ?: [:])[organism]
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
