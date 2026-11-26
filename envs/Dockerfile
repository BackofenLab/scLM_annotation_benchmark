FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        unzip \
        wget \
        bzip2 \
        ca-certificates \
        git-lfs \
        procps \
        gcc \
        g++ \
        python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml -n sc_lfm && \
    conda clean --all --yes

# PyTorch and torch-geometric with their CUDA 11.8 extensions. Wheels:
# PyTorch https://pytorch.org/get-started/previous-versions/#v271
# PyG     https://data.pyg.org/whl/torch-2.7.1+cu118.html
RUN /opt/conda/envs/sc_lfm/bin/pip install --no-cache-dir \
    torch==2.7.1 torchvision==0.22.1 \
    torch-geometric==2.6.1 \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    -f https://data.pyg.org/whl/torch-2.7.1+cu118.html

# pip packages in a step of their own, so that they appear in the build log
RUN /opt/conda/envs/sc_lfm/bin/pip install --no-cache-dir \
    setuptools==80.1.0 \
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

# mmcontext and adata_hf_datasets without dependencies: these would pull anndata>=0.12, which
# needs scipy>=1.12 and conflicts with omicverse
RUN /opt/conda/envs/sc_lfm/bin/pip install --no-cache-dir --no-deps \
    git+https://github.com/mengerj/mmcontext.git@main \
    git+https://github.com/mengerj/adata_hf_datasets.git

# The other dependencies of mmcontext
RUN /opt/conda/envs/sc_lfm/bin/pip install --no-cache-dir \
    datasets hydra-core python-dotenv scib pysradb tdigest

# CellWhisperer with its Geneformer submodule; CellWhisperer's pyproject.toml installs
# geneformer from modules/Geneformer
RUN git lfs install && \
    git clone --recurse-submodules https://github.com/mengerj/CellWhisperer.git /tmp/CellWhisperer && \
    cd /tmp/CellWhisperer/modules/Geneformer && git lfs pull && \
    /opt/conda/envs/sc_lfm/bin/pip install --no-cache-dir /tmp/CellWhisperer && \
    rm -rf /tmp/CellWhisperer

# Writable numba cache directory
RUN mkdir -p /tmp/numba_cache && chmod 777 /tmp/numba_cache

ENV CONDA_DEFAULT_ENV=sc_lfm
ENV PATH=/opt/conda/envs/sc_lfm/bin:$PATH
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

CMD ["/bin/bash"]