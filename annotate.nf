// Annotation only: annotate the test sets with every selected method and map the answers to
// the Cell Ontology. The mapping table is not changed.
//
//     nextflow run annotate.nf -profile local

nextflow.enable.dsl = 2

include { ANNOTATE } from './workflows/annotate'

workflow {
    ANNOTATE(false)
}
