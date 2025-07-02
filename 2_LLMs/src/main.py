
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np

from ollama import chat
from ollama import ChatResponse

from omegaconf.omegaconf import OmegaConf


cfg = OmegaConf.load("../config/llms.yaml")
celltype_key = "cell_ontology_class"

def read_anndata():
    hvg = 3000
    sc_data = cfg.data_path
    adata = ad.read_h5ad(sc_data)
    # preprocess anndata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=hvg
    )
    adata_hvg = adata[:, adata.var.highly_variable].copy()
    print(adata_hvg.X.shape)
    
    print(len(adata_hvg.obs[celltype_key]))
    print(adata_hvg.obs[celltype_key].unique())
    return adata_hvg

def get_responses():
    llm = "{}:{}".format(cfg.model_name, cfg.model_version)
    response: ChatResponse = chat(model=llm, messages=[
      {
        'role': 'user',
        'content': 'Does perturbing CCND3 gene in K562 cells cause the gene expression of \
            SDE2 to upregulate or downregulate? Provide response either as Yes or No',
      },
    ])
    print(response.message.content)


def get_celltype(anndata):
    gene_names = anndata.var_names
    X_dense = anndata.X.toarray() if not isinstance(anndata.X, np.ndarray) else anndata.X
    cell_expression_dicts = [
        dict(zip(gene_names, cell_row)) for cell_row in X_dense
    ]
    all_celltypes = anndata.obs[celltype_key].tolist()
    print(len(cell_expression_dicts), len(all_celltypes))
    llm = "{}:{}".format(cfg.model_name, cfg.model_version)
    for c_idx, c_expr in enumerate(cell_expression_dicts):
        print(c_expr)
        filtered_dict = {gene: expr for gene, expr in c_expr.items() if expr != 0.0}
        print(filtered_dict)
        response: ChatResponse = chat(model=llm, messages=[
          {
              'role': 'user',
              'content': f"Given this gene expression {filtered_dict} of single cell data, infer the cell type using the provided single cell gene expression data?" 
          },
        ])
        print(response.message.content)
        print()
        print(f"Original cell type: {all_celltypes[c_idx]}")
        break


if __name__ == "__main__":
    #get_responses()
    hgv_anndata = read_anndata()
    get_celltype(hgv_anndata)
