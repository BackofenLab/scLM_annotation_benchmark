#!/usr/bin/env Rscript
# Annotate the cells of a test set with SingleR.
#
# SingleR labels every cell by Spearman correlation with a labelled reference: the celldex
# reference named by singler_reference in the dataset YAML or, with
# singler_reference_source: "cellxgene", the reference h5ad in SINGLER_REF_H5AD
# (bin/prepare_singler_reference.py). Writes cell_id, true_labels and predicted_labels to a CSV.
# celldex downloads are cached in <cache_dir>/ExperimentHub, so only the first run needs the
# network. Called by run.sh.

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

# Reference source: "celldex" (loaded by name, the default) or "cellxgene" (the h5ad in
# SINGLER_REF_H5AD, which the pipeline sets).
ref_source <- as.character(cfg$singler_reference_source %||% "celldex")
ref_label_key <- as.character(cfg$singler_reference_label_key %||% "cell_type")
ref_path <- Sys.getenv("SINGLER_REF_H5AD", "")

# celldex references have three label columns, label.main, label.fine and label.ont, whose number
# of labels depends on the reference. label.ont holds Cell Ontology IDs, which the evaluation
# resolves without the mapping table; it is the default unless --label_tier or
# singler_reference_tier is set. A cellxgene reference has a single label column.
tier <- as.character(opt$label_tier %||% cfg$singler_reference_tier %||% "ont")
if (!tier %in% c("main", "fine", "ont"))
  stop("--label_tier must be one of: main, fine, ont (got '", tier, "')")
label_col <- paste0("label.", tier)

if (!is.null(opt$cache_dir) && nzchar(opt$cache_dir)) {
  hub <- file.path(opt$cache_dir, "ExperimentHub")
  dir.create(hub, recursive = TRUE, showWarnings = FALSE)
  setExperimentHubOption("CACHE", hub)
}

# reader = "R" reads the file with rhdf5. The default Python reader needs basilisk and fails on
# the float32 CSR matrices of these files (dgRMatrix type error).
sce <- readH5AD(opt$input, reader = "R")
# X holds log-normalized values, which SingleR expects: use it as logcounts.
if ("X" %in% assayNames(sce)) assayNames(sce)[assayNames(sce) == "X"] <- "logcounts"
if (!"logcounts" %in% assayNames(sce)) assay(sce, "logcounts") <- assay(sce, 1)

true <- if (annotation_key %in% colnames(colData(sce))) {
  as.character(colData(sce)[[annotation_key]])
} else rep("NA", ncol(sce))

# drop_na_labels removes the reference cells without a label (label.ont is missing for some
# celldex cells), which SingleR would otherwise treat as a class "NA".
# Constrained mode (LABEL_CONSTRAINT): the reference also keeps only the labels that count for one
# of the true cell types (bin/util/label_constraint.R). SingleR chooses its marker genes among the
# labels it is given, so this is a run of its own, not a filter on the unconstrained answers.
constraint <- Sys.getenv("LABEL_CONSTRAINT", "")
if (nzchar(constraint)) {
  here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
  source(file.path(here, "..", "..", "..", "bin", "util", "label_constraint.R"))
}
drop_na_labels <- function(r, labs) {
  keep <- !is.na(labs) & nzchar(labs)
  kept <- NULL
  if (nzchar(constraint)) {
    kept <- constrain(unique(labs[keep]), constraint, "SingleR", true)
    if (length(kept) < 2) stop("fewer than 2 reference labels count for the listed cell types")
    keep <- keep & labs %in% names(kept)
  }
  list(ref = r[, keep], labels = labs[keep], kept = kept)
}

if (identical(ref_source, "cellxgene") && nzchar(ref_path)) {
  # CELLxGENE reference (bin/prepare_singler_reference.py). Test set and reference both have
  # Ensembl IDs as rownames and log-normalized X; SingleR matches the genes by rownames.
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
  # celldex reference, e.g. MonacoImmuneData or MouseRNAseqData. Its rownames are gene symbols
  # (upper case for human, title case for mouse), so the rownames of the test set become its
  # gene symbols (rowData$gene_names) in their original case.
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

lab <- as.character(pred$labels); lab[is.na(lab)] <- "Unknown"
if (!is.null(d$kept)) lab <- unname(ifelse(lab %in% names(d$kept), d$kept[lab], lab))

out <- data.frame(cell_id = colnames(sce), true_labels = true,
                  predicted_labels = lab, stringsAsFactors = FALSE)
if (!is.null(d$kept)) out$reachable <- out$true_labels %in% d$kept
write.csv(out, opt$output, row.names = FALSE)
cat(sprintf("SingleR: wrote %d predictions (ref=%s, %d ref labels, %d predicted) to %s\n",
            nrow(out), ref_desc, length(unique(ref_labels)), length(unique(lab)), opt$output))
