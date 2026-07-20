#!/usr/bin/env Rscript
# SingleR reference-correlation cell-type annotation.
#
# Annotates each cell by Spearman correlation of its transcriptome to a celldex built-in
# reference (chosen per dataset via `singler_reference` in the dataset YAML). Emits the
# pipeline contract: cell_id, true_labels, predicted_labels (one row per cell).
#
# celldex/ExperimentHub downloads are cached under the pipeline cache dir ($4), so the first
# run fetches the reference and later runs are offline-stable.

suppressPackageStartupMessages({
  library(optparse); library(yaml); library(zellkonverter); library(SingleCellExperiment)
  library(SingleR); library(celldex); library(ExperimentHub)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option("--input"), make_option("--output"),
  make_option("--config"), make_option("--cache_dir"),
  make_option("--label_tier")
)))

`%||%` <- function(a, b) if (is.null(a) || is.na(a) || identical(a, "")) b else a
cfg <- yaml::read_yaml(opt$config)
ref_name <- as.character(cfg$singler_reference %||% "HumanPrimaryCellAtlasData")
annotation_key <- as.character(cfg$annotation_key %||% "cell_type")

# Reference source: "celldex" (built-in, loaded by name) or "cellxgene" (a fetched reference h5ad
# whose path arrives via the SINGLER_REF_H5AD env var, exported by the run_model_singler_ref
# process). Default celldex so datasets without the flag behave exactly as before.
ref_source <- as.character(cfg$singler_reference_source %||% "celldex")
ref_label_key <- as.character(cfg$singler_reference_label_key %||% "cell_type")
ref_path <- Sys.getenv("SINGLER_REF_H5AD", "")

# celldex references carry three label tiers, whose granularity is reference-specific (e.g. Monaco
# 10/29/28, MouseRNAseq 18/28/22, HPCA 36/157/66 for main/fine/ont). `ont` holds Cell Ontology IDs,
# which the evaluator resolves without the mappings CSV and is the best-scoring tier, so it is the
# default when neither --label_tier nor cfg$singler_reference_tier is set. (Only used on the celldex
# path; the cellxgene path has a single label level.)
tier <- as.character(opt$label_tier %||% cfg$singler_reference_tier %||% "ont")
if (!tier %in% c("main", "fine", "ont"))
  stop("--label_tier must be one of: main, fine, ont (got '", tier, "')")
label_col <- paste0("label.", tier)

if (!is.null(opt$cache_dir) && nzchar(opt$cache_dir)) {
  hub <- file.path(opt$cache_dir, "ExperimentHub")
  dir.create(hub, recursive = TRUE, showWarnings = FALSE)
  setExperimentHubOption("CACHE", hub)
}

# reader="R" reads the HDF5 directly in R (rhdf5) — avoids basilisk/python and the float32-CSR
# dgRMatrix dtype error the default (python) reader hits on this pipeline's h5ads.
sce <- readH5AD(opt$input, reader = "R")
# The pipeline's adata.X is log1p-normalized -> treat as logcounts (SingleR wants log-normalized).
if ("X" %in% assayNames(sce)) assayNames(sce)[assayNames(sce) == "X"] <- "logcounts"
if (!"logcounts" %in% assayNames(sce)) assay(sce, "logcounts") <- assay(sce, 1)

# Drop reference cells whose label is NA/empty (no usable training signal — label.ont in particular
# is unset for some celldex cells); otherwise SingleR would correlate against an "NA" class.
drop_na_labels <- function(r, labs) {
  keep <- !is.na(labs) & nzchar(labs)
  list(ref = r[, keep], labels = labs[keep])
}

if (identical(ref_source, "cellxgene") && nzchar(ref_path)) {
  # --- Custom CELLxGENE reference (fetched by bin/prepare_singler_reference.py) ---
  # The reference h5ad shares the query's convention (Ensembl var_names, log1p X). Match genes on
  # Ensembl IDs: keep rownames as-is (Ensembl) on BOTH query and reference; SingleR intersects on
  # rownames. This avoids the celldex symbol-case handling entirely.
  message("SingleR: custom cellxgene reference ", ref_path, " (label '", ref_label_key, "')")
  ref <- readH5AD(ref_path, reader = "R")
  if ("X" %in% assayNames(ref)) assayNames(ref)[assayNames(ref) == "X"] <- "logcounts"
  if (!"logcounts" %in% assayNames(ref)) assay(ref, "logcounts") <- assay(ref, 1)
  if (!ref_label_key %in% colnames(colData(ref)))
    stop("Reference h5ad '", ref_path, "' has no obs column '", ref_label_key, "'")
  d <- drop_na_labels(ref, as.character(colData(ref)[[ref_label_key]]))
  ref <- d$ref; ref_labels <- d$labels
  pred <- SingleR(test = sce, ref = ref, labels = ref_labels,
                  assay.type.test = "logcounts", assay.type.ref = "logcounts")
  ref_desc <- paste0("cellxgene:", basename(ref_path))
} else {
  # --- celldex built-in reference (default), e.g. MonacoImmuneData(), MouseRNAseqData() ---
  # celldex refs are keyed by gene SYMBOL; put native-case symbols on the query rownames so they
  # match (human refs = UPPER symbols, mouse = Title-case, as stored per organism upstream).
  sym <- if ("gene_names" %in% colnames(rowData(sce))) as.character(rowData(sce)$gene_names) else rownames(sce)
  rownames(sce) <- make.unique(sym)
  ref <- get(ref_name, asNamespace("celldex"))()
  if (!label_col %in% colnames(colData(ref)))
    stop("Reference '", ref_name, "' has no column '", label_col, "'")
  d <- drop_na_labels(ref, as.character(colData(ref)[[label_col]]))
  ref <- d$ref; ref_labels <- d$labels
  pred <- SingleR(test = sce, ref = ref, labels = ref_labels, assay.type.test = "logcounts")
  ref_desc <- paste0(ref_name, " (tier=", tier, ")")
}

true <- if (annotation_key %in% colnames(colData(sce))) {
  as.character(colData(sce)[[annotation_key]])
} else rep("NA", ncol(sce))
lab <- as.character(pred$labels); lab[is.na(lab)] <- "Unknown"

out <- data.frame(cell_id = colnames(sce), true_labels = true,
                  predicted_labels = lab, stringsAsFactors = FALSE)
write.csv(out, opt$output, row.names = FALSE)
cat(sprintf("SingleR: wrote %d predictions (ref=%s, %d ref labels -> %d predicted) to %s\n",
            nrow(out), ref_desc, length(unique(ref_labels)), length(unique(lab)), opt$output))
