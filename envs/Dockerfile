FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml -n sc_lfm && \
    conda clean --all --yes

# Install specific versions that are compatible
RUN /opt/conda/envs/sc_lfm/bin/python -m pip install setuptools==59.6.0 importlib-metadata

# Explicitly install pip packages to ensure they're visible in build logs
RUN /opt/conda/envs/sc_lfm/bin/pip install \
    cell2sentence==1.2.0 \
    pybiomart==0.2.0 \
    omicverse==1.7.7

# Create writable numba cache directory
RUN mkdir -p /tmp/numba_cache && chmod 777 /tmp/numba_cache

ENV CONDA_DEFAULT_ENV=sc_lfm
ENV PATH=/opt/conda/envs/sc_lfm/bin:$PATH
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

CMD ["/bin/bash"]