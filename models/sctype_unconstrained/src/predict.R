#!/usr/bin/env Rscript
# Annotate the cells of a test set with scType.
#
# Clusters the cells (scran SNN graph and Louvain), scores the marker sets of ScTypeDB with
# sctype_score(), gives every cluster its top-scoring cell type and writes cell_id, true_labels
# and predicted_labels for every test cell to a CSV. Called by run.sh.

suppressPackageStartupMessages({
  library(optparse); library(yaml); library(zellkonverter); library(SingleCellExperiment)
  library(scran); library(scater); library(igraph); library(dplyr)
  library(openxlsx); library(HGNChelper)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option("--input"), make_option("--output"), make_option("--markers"),
  make_option("--config"), make_option("--sctype_dir")
)))

# The dataset YAML is read in R, so the environment needs no Python.
`%||%` <- function(a, b) if (is.null(a) || is.na(a) || identical(a, "")) b else a
cfg <- yaml::read_yaml(opt$config)
opt$tissue <- as.character(cfg$sctype_tissue %||% "")
opt$annotation_key <- as.character(cfg$annotation_key %||% "cell_type")

# scType's helper functions from the sc-type repository (see prepare.sh)
source(file.path(opt$sctype_dir, "gene_sets_prepare.R"))
source(file.path(opt$sctype_dir, "sctype_score_.R"))

# With CONTEXT_H5AD, a larger sample that contains the test cells (bin/build_context.py) is
# clustered and scored, so that rare cell types can form their own cluster and the z-scores
# reflect the whole sample. Every test cell gets the label of its context cluster. Without it,
# the test set itself is clustered.
ctx_path <- Sys.getenv("CONTEXT_H5AD", "")
cluster_src <- if (nzchar(ctx_path) && file.exists(ctx_path)) ctx_path else opt$input
if (identical(cluster_src, opt$input)) {
  cat("scType: clustering the eval subset itself (no CONTEXT_H5AD set).\n")
} else {
  cat(sprintf("scType: clustering context '%s', scoring eval cells from '%s'.\n", ctx_path, opt$input))
}

# reader = "R" reads the file with rhdf5. The default Python reader needs basilisk and fails on
# the float32 CSR matrices of these files (dgRMatrix type error).
sce <- readH5AD(cluster_src, reader = "R")
# X holds log-normalized values: use it as logcounts.
if ("X" %in% assayNames(sce)) assayNames(sce)[assayNames(sce) == "X"] <- "logcounts"
if (!"logcounts" %in% assayNames(sce)) assay(sce, "logcounts") <- assay(sce, 1)

# Gene symbols are in rowData$gene_names (var_names are Ensembl IDs). In upper case, mouse
# symbols match the human symbols of ScTypeDB for most one-to-one orthologs.
sym <- if ("gene_names" %in% colnames(rowData(sce))) as.character(rowData(sce)$gene_names) else rownames(sce)
rownames(sce) <- make.unique(toupper(sym))

# The cells and true labels of the output come from the test set.
sce_eval <- if (identical(cluster_src, opt$input)) sce else readH5AD(opt$input, reader = "R")
out_cells <- colnames(sce_eval)
true_eval <- if (opt$annotation_key %in% colnames(colData(sce_eval))) {
  setNames(as.character(colData(sce_eval)[[opt$annotation_key]]), out_cells)
} else setNames(rep("NA", length(out_cells)), out_cells)

# Clustering with the settings of the scType tutorial in Seurat (2,000 highly variable genes,
# k = 20, Louvain resolution 0.8), using scran's SNN graph and igraph's Louvain instead of Seurat.
set.seed(0)
dec <- modelGeneVar(sce)
hvg <- getTopHVGs(dec, n = min(2000L, nrow(sce)))
sce <- runPCA(sce, ncomponents = min(30L, length(hvg) - 1L), subset_row = hvg)
g <- buildSNNGraph(sce, use.dimred = "PCA", k = 20)
clusters <- as.character(igraph::cluster_louvain(g, resolution = 0.8)$membership)

# Marker scores on z-scores per gene, clipped to [-10, 10] like Seurat's ScaleData (scale.max).
gs <- gene_sets_prepare(opt$markers, opt$tissue)
# Constrained mode (LABEL_CONSTRAINT): only the cell types that count for one of the true cell types
# are scored (bin/util/label_constraint.R), so the marker weights are recomputed over them. The
# clustering above does not depend on the gene sets and stays identical.
constraint <- Sys.getenv("LABEL_CONSTRAINT", "")
kept <- NULL
if (nzchar(constraint)) {
  source(file.path(opt$sctype_dir, "..", "..", "..", "bin", "util", "label_constraint.R"))
  kept <- constrain(names(gs$gs_positive), constraint, "scType", true_eval)
  if (length(kept) < 2) stop("fewer than 2 cell types count for the listed cell types")
  gs <- lapply(gs, function(s) s[names(s) %in% names(kept)])
}
# Only the marker genes are made dense and scaled. scale() works per gene, so their z-scores
# are the same as when all genes are scaled, and sctype_score() uses only these genes. The dense
# matrix of cells by marker genes fits in memory for a context of 100,000 cells.
marker_genes <- intersect(unique(c(unlist(gs$gs_positive), unlist(gs$gs_negative))), rownames(sce))
scaled <- t(scale(t(as.matrix(assay(sce, "logcounts")[marker_genes, , drop = FALSE]))))
scaled[is.na(scaled)] <- 0
scaled[scaled > 10] <- 10; scaled[scaled < -10] <- -10
es.max <- sctype_score(scRNAseqData = scaled, scaled = TRUE,
                       gs = gs$gs_positive, gs2 = gs$gs_negative)

# Every clustered cell gets the top-scoring cell type of its cluster.
cells <- colnames(sce)
assign <- setNames(rep("Unknown", length(cells)), cells)
for (cl in unique(clusters)) {
  idx <- which(clusters == cl)
  es.cl <- sort(rowSums(es.max[, idx, drop = FALSE]), decreasing = TRUE)
  top <- names(es.cl)[1]
  # As in scType, a cluster whose top score is below a quarter of its cell number is "Unknown".
  if (as.numeric(es.cl[1]) < length(idx) / 4) top <- "Unknown"
  assign[idx] <- top
}

# Output only the test cells. The context contains every test cell (bin/build_context.py).
missing <- setdiff(out_cells, names(assign))
if (length(missing) > 0)
  cat(sprintf("scType: WARNING %d/%d eval cells absent from the clustering context; labeled Unknown.\n",
              length(missing), length(out_cells)))
pred <- assign[out_cells]; pred[is.na(pred)] <- "Unknown"
if (!is.null(kept)) pred <- ifelse(pred %in% names(kept), kept[pred], pred)

out <- data.frame(cell_id = out_cells, true_labels = unname(true_eval[out_cells]),
                  predicted_labels = unname(pred), stringsAsFactors = FALSE)
if (!is.null(kept)) out$reachable <- out$true_labels %in% kept
write.csv(out, opt$output, row.names = FALSE)
cat(sprintf("scType: wrote %d predictions (%d cells clustered into %d clusters, tissue='%s') to %s\n",
            nrow(out), length(cells), length(unique(clusters)), opt$tissue, opt$output))
