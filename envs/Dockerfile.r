# R and Bioconductor image for the methods written in R (SingleR and scType); the main image
# sc_lfm has no R. Methods whose method.yml sets `container: r` run in this image (containerFor()
# in modules/functions.nf); all other processes use params.container_image.
#
# Bioconductor 3.20 with R 4.4, as in envs/models/singler_unconstrained.yml and
# sctype_unconstrained.yml (bioconductor-singler 2.8.0 and scran 1.34.0 belong to Bioconductor
# 3.20). Build and push the image, then set params.r_container_image to its tag:
#   docker build -f envs/Dockerfile.r -t <registry>/sc_lfm_r:<tag> envs
#   docker push <registry>/sc_lfm_r:<tag>
FROM bioconductor/bioconductor_docker:RELEASE_3_20

# Bioconductor packages: SingleR and celldex (references), scran and scater (scType clustering),
# zellkonverter with rhdf5 and HDF5Array (readH5AD(reader = "R") reads .h5ad files without
# basilisk or Python).
RUN R -e "BiocManager::install(c( \
      'SingleR','celldex','SingleCellExperiment','SummarizedExperiment', \
      'scran','scater','zellkonverter','rhdf5','HDF5Array','ExperimentHub' \
    ), update=FALSE, ask=FALSE, version='3.20')" \
 && R -e "if (!all(c('SingleR','scran','zellkonverter') %in% rownames(installed.packages()))) quit(status=1)"

# CRAN packages of the predict.R scripts (arguments, YAML, marker workbook, clustering).
RUN R -e "install.packages(c('optparse','yaml','openxlsx','HGNChelper','igraph','dplyr'), \
      repos='https://cloud.r-project.org')" \
 && R -e "if (!all(c('optparse','HGNChelper','openxlsx') %in% rownames(installed.packages()))) quit(status=1)"
