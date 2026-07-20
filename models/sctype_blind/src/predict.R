#!/usr/bin/env Rscript
# scType per-cluster cell-type annotation.
#
# Reads the pipeline's subset h5ad, clusters cells (scran SNN graph + Louvain), scores the
# ScTypeDB marker signatures per cluster with sctype_score(), assigns each cluster its top
# cell type, and propagates that label to all its cells (scType's intended usage). Emits the
# pipeline contract: cell_id, true_labels, predicted_labels (one row per cell).

suppressPackageStartupMessages({
  library(optparse); library(yaml); library(zellkonverter); library(SingleCellExperiment)
  library(scran); library(scater); library(igraph); library(dplyr)
  library(openxlsx); library(HGNChelper)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option("--input"), make_option("--output"), make_option("--markers"),
  make_option("--config"), make_option("--sctype_dir")
)))

# Per-dataset settings are read from the dataset YAML in R (keeps the env R-only).
`%||%` <- function(a, b) if (is.null(a) || is.na(a) || identical(a, "")) b else a
cfg <- yaml::read_yaml(opt$config)
opt$tissue <- as.character(cfg$sctype_tissue %||% "")
opt$organism <- as.character(cfg$organism %||% cfg$species %||% "human")
opt$annotation_key <- as.character(cfg$annotation_key %||% "cell_type")

# scType helper functions fetched by prepare.sh
source(file.path(opt$sctype_dir, "gene_sets_prepare.R"))
source(file.path(opt$sctype_dir, "sctype_score_.R"))

# Optional clustering context: when CONTEXT_H5AD points to a larger (up to 100k) h5ad that
# CONTAINS the eval subset, scType clusters and scores markers on that context, so rare cell
# types can form their own cluster and z-scores use the full composition; each context cluster's
# label is then propagated to the eval cells it contains. Without it, scType clusters the eval
# subset itself (original behavior). The scored set is always exactly the eval subset.
ctx_path <- Sys.getenv("CONTEXT_H5AD", "")
cluster_src <- if (nzchar(ctx_path) && file.exists(ctx_path)) ctx_path else opt$input
if (identical(cluster_src, opt$input)) {
  cat("scType: clustering the eval subset itself (no CONTEXT_H5AD set).\n")
} else {
  cat(sprintf("scType: clustering context '%s', scoring eval cells from '%s'.\n", ctx_path, opt$input))
}

# reader="R" reads the HDF5 directly in R (rhdf5) — avoids basilisk/python and the float32-CSR
# dgRMatrix dtype error the default (python) reader hits on this pipeline's h5ads.
sce <- readH5AD(cluster_src, reader = "R")
# The pipeline's adata.X is log1p-normalized -> treat as logcounts.
if ("X" %in% assayNames(sce)) assayNames(sce)[assayNames(sce) == "X"] <- "logcounts"
if (!"logcounts" %in% assayNames(sce)) assay(sce, "logcounts") <- assay(sce, 1)

# Gene symbols live in rowData$gene_names (var_names are Ensembl IDs). Uppercase so that
# mouse symbols approximately match ScTypeDB's human symbols (conserved 1:1 orthologs).
sym <- if ("gene_names" %in% colnames(rowData(sce))) as.character(rowData(sce)$gene_names) else rownames(sce)
rownames(sce) <- make.unique(toupper(sym))

# Eval cells to emit + their ground-truth labels always come from the eval subset, so the scored
# set is exactly the fixed subset regardless of the clustering context.
sce_eval <- if (identical(cluster_src, opt$input)) sce else readH5AD(opt$input, reader = "R")
out_cells <- colnames(sce_eval)
true_eval <- if (opt$annotation_key %in% colnames(colData(sce_eval))) {
  setNames(as.character(colData(sce_eval)[[opt$annotation_key]]), out_cells)
} else setNames(rep("NA", length(out_cells)), out_cells)

# --- Cluster: SNN graph + Louvain, matching the scType tutorial's Seurat defaults
#     (FindNeighbors k=20, FindClusters Louvain resolution=0.8, 2000 HVGs). scran's SNN + igraph
#     Louvain is the SCE-native equivalent used to avoid a heavy Seurat dependency. ---
set.seed(0)
dec <- modelGeneVar(sce)
hvg <- getTopHVGs(dec, n = min(2000L, nrow(sce)))
sce <- runPCA(sce, ncomponents = min(30L, length(hvg) - 1L), subset_row = hvg)
g <- buildSNNGraph(sce, use.dimred = "PCA", k = 20)
clusters <- as.character(igraph::cluster_louvain(g, resolution = 0.8)$membership)

# --- scType marker scoring (z-score per gene, clipped at +-10 like Seurat ScaleData scale.max) ---
gs <- gene_sets_prepare(opt$markers, opt$tissue)
# Restrict to marker genes BEFORE densifying + scaling. scale() is per-gene independent, so the
# z-scores of the marker genes are identical whether the whole matrix or only those genes are
# scaled, and sctype_score() subselects to exactly these genes internally anyway. This keeps the
# dense intermediate at n_cells x n_markers (~hundreds of genes) instead of n_cells x n_genes,
# which is what makes a large clustering context (up to 100k cells) fit in memory.
marker_genes <- intersect(unique(c(unlist(gs$gs_positive), unlist(gs$gs_negative))), rownames(sce))
scaled <- t(scale(t(as.matrix(assay(sce, "logcounts")[marker_genes, , drop = FALSE]))))
scaled[is.na(scaled)] <- 0
scaled[scaled > 10] <- 10; scaled[scaled < -10] <- -10
es.max <- sctype_score(scRNAseqData = scaled, scaled = TRUE,
                       gs = gs$gs_positive, gs2 = gs$gs_negative)

# --- Per-cluster top type -> propagate to cells (over the clustering source) ---
cells <- colnames(sce)
assign <- setNames(rep("Unknown", length(cells)), cells)
for (cl in unique(clusters)) {
  idx <- which(clusters == cl)
  es.cl <- sort(rowSums(es.max[, idx, drop = FALSE]), decreasing = TRUE)
  top <- names(es.cl)[1]
  # scType low-confidence guard: total score below n_cells/4 -> Unknown.
  if (as.numeric(es.cl[1]) < length(idx) / 4) top <- "Unknown"
  assign[idx] <- top
}

# Emit only the eval-subset cells, propagating their context cluster's label. Every eval cell
# should be contained in the clustering source (the context is union-constructed to guarantee it).
missing <- setdiff(out_cells, names(assign))
if (length(missing) > 0)
  cat(sprintf("scType: WARNING %d/%d eval cells absent from the clustering context -> Unknown.\n",
              length(missing), length(out_cells)))
pred <- assign[out_cells]; pred[is.na(pred)] <- "Unknown"

out <- data.frame(cell_id = out_cells, true_labels = unname(true_eval[out_cells]),
                  predicted_labels = unname(pred), stringsAsFactors = FALSE)
write.csv(out, opt$output, row.names = FALSE)
cat(sprintf("scType: wrote %d predictions (%d cells clustered into %d clusters, tissue='%s') to %s\n",
            nrow(out), length(cells), length(unique(clusters)), opt$tissue, opt$output))
