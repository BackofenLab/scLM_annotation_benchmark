// Annotation only: annotate the cells of every selected dataset with every selected method and
// map the answers to the Cell Ontology. All cells are annotated, or a subsample of
// annotation_subset_size cells. The mapping table is not changed. A dataset's
// expected_cell_types, if given, are the list of the constrained methods.
//
//     nextflow run annotate.nf -profile local

nextflow.enable.dsl = 2

include { reportLeftOut } from './modules/functions'
include { ANNOTATE } from './workflows/annotate'

workflow {
    ANNOTATE(false, params.annotation_subset_size ? [params.annotation_subset_size as int] : ['all'], true)
}

workflow.onComplete {
    reportLeftOut()
}
