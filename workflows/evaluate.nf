// Evaluation: score the mapped predictions and combine the runs into tables and plots.

include { absolutePath; mappingsVersion } from '../modules/functions'
include { score_predictions; combine_results } from '../modules/evaluation'

workflow EVALUATE {
    take:
    mapped    // method, dataset, test set size, mapped prediction files

    main:
    scores = score_predictions(mapped, absolutePath(params.cache_dir),
                               absolutePath(params.mappings), mappingsVersion())
    combine_results(scores.flatten().collect())
}
