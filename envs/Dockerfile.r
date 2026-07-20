# Compact R/Bioconductor image for the R-based annotators only: SingleR (main/fine/ont tiers) and
# scType. The main sc_lfm image is Python-only; rather than bloat it with R, these models run here
# via a per-model `container` directive (see containerFor() in main.nf). Every other process keeps
# params.container_image, so this does not touch the rest of the pipeline's container story.
#
# Pinned to Bioconductor 3.20 / R 4.4 to match envs/models/singler_blind.yml + sctype_blind.yml
# (bioconductor-singler 2.8.0, scran 1.34.0 live in BioC 3.20). Build + push, then set
# params.r_container_image to the pushed tag:
#   docker build -f envs/Dockerfile.r -t <registry>/sc_lfm_r:<tag> envs
#   docker push <registry>/sc_lfm_r:<tag>
FROM bioconductor/bioconductor_docker:RELEASE_3_20

# Bioconductor deps: SingleR + celldex (references), scran/scater (scType clustering), zellkonverter
# + rhdf5/HDF5Array (readH5AD(reader="R") reads the pipeline's .h5ad in pure R, no basilisk/python).
RUN R -e "BiocManager::install(c( \
      'SingleR','celldex','SingleCellExperiment','SummarizedExperiment', \
      'scran','scater','zellkonverter','rhdf5','HDF5Array','ExperimentHub' \
    ), update=FALSE, ask=FALSE, version='3.20')" \
 && R -e "if (!all(c('SingleR','scran','zellkonverter') %in% rownames(installed.packages()))) quit(status=1)"

# CRAN deps used by predict.R / run.sh (arg parsing, YAML, marker DB, clustering helpers).
RUN R -e "install.packages(c('optparse','yaml','openxlsx','HGNChelper','igraph','dplyr'), \
      repos='https://cloud.r-project.org')" \
 && R -e "if (!all(c('optparse','HGNChelper','openxlsx') %in% rownames(installed.packages()))) quit(status=1)"
