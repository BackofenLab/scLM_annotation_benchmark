FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        unzip \
        wget \
        bzip2 \
        ca-certificates \
        git-lfs \
        procps \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml -n sc_lfm && \
    conda clean --all --yes

# Install CUDA-enabled PyTorch + torch-geometric with CUDA extensions
# PyTorch CUDA wheels: https://pytorch.org/get-started/previous-versions/#v271
# PyG CUDA wheels:     https://data.pyg.org/whl/torch-2.7.1+cu118.html
RUN /opt/conda/envs/sc_lfm/bin/pip install --no-cache-dir \
    torch==2.7.1 torchvision==0.22.1 \
    torch-geometric==2.6.1 \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    -f https://data.pyg.org/whl/torch-2.7.1+cu118.html

# Explicitly install pip packages to ensure they're visible in build logs
RUN /opt/conda/envs/sc_lfm/bin/pip install \
    cell2sentence==1.2.0 \
    pybiomart==0.2.0 \
    omicverse==1.7.7 \
    loompy==3.0.8 \
    mygene==3.2.2 \
    psutil==7.2.2 \
    pynvml==13.0.1 \
    gdown==5.2.1 \
    einops==0.8.2 \
    local-attention==1.11.2

# Create writable numba cache directory
RUN mkdir -p /tmp/numba_cache && chmod 777 /tmp/numba_cache

ENV CONDA_DEFAULT_ENV=sc_lfm
ENV PATH=/opt/conda/envs/sc_lfm/bin:$PATH
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

CMD ["/bin/bash"]