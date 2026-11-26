// Benchmark: annotate the test sets with every selected method, then score the annotations
// against the datasets' labels.
//
//     nextflow run main.nf -profile local

nextflow.enable.dsl = 2

include { absolutePath } from './modules/functions'
include { ANNOTATE } from './workflows/annotate'
include { EVALUATE } from './workflows/evaluate'

workflow {
    def sizes = params.subset_sizes instanceof List ? params.subset_sizes : [params.subset_sizes]
    if (sizes.any { !it.toString().isInteger() }) {
        error "subset_sizes must be numbers; annotate.nf annotates all cells"
    }
    ANNOTATE(params.append_new_answers, sizes.collect { it as int })
    EVALUATE(ANNOTATE.out.mapped)
}

workflow.onComplete {
    def table = new File(absolutePath(params.mappings))
    def open = table.exists() ? table.readLines().count { it.startsWith('False;') } : 0
    if (open) {
        log.info "${open} answer(s) in ${table} are not reviewed yet. They count as wrong, and a " +
                 "run with too few reviewed answers has no metrics. Review them and run again " +
                 "with -resume."
    }
}
