#!/usr/bin/env python
"""Give a mouse AnnData the human gene identifiers that human-only models read.

CellWhisperer, LangCell and mmContext match genes by human Ensembl ID and InstructCell by human
gene symbol, so on mouse data they find no genes and return one label for every cell. This maps
each mouse gene that has a one-to-one human ortholog (Ensembl 114,
cache/ENSEMBL/mouse_human_one2one_orthologs_ensembl114.tsv) to that ortholog's Ensembl ID and
symbol. Expression values are not touched. Genes without a one-to-one ortholog keep their mouse
identifiers, as InstructCell's own preprocessing does, so the human models do not find them.

    python bin/orthologize.py --input mouse.h5ad --output mouse_human_orthologs.h5ad
"""
import argparse
import json
from pathlib import Path

import anndata
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORTHOLOGS = ROOT / "cache" / "ENSEMBL" / "mouse_human_one2one_orthologs_ensembl114.tsv"


def load_orthologs(path: Path = ORTHOLOGS) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if not (table.mouse_gene_id.is_unique and table.human_gene_id.is_unique):
        raise ValueError(f"{path} is not one-to-one")
    return table.set_index("mouse_gene_id")


def to_human_orthologs(adata: anndata.AnnData, orthologs: pd.DataFrame,
                       id_col: str = "ensembl_id", name_col: str = "gene_names") -> anndata.AnnData:
    """Return a copy whose var index, `id_col` and `name_col` hold the human orthologs."""
    var = adata.var
    mouse_id = (var[id_col] if id_col in var else pd.Series(var.index, index=var.index))
    mouse_id = mouse_id.astype(str).str.replace(r"\.\d+$", "", regex=True)
    mouse_name = var[name_col].astype(str) if name_col in var else mouse_id
    hit = mouse_id.isin(orthologs.index)
    human_id = mouse_id.map(orthologs.human_gene_id)
    human_name = mouse_id.map(orthologs.human_gene_name)

    out = adata.copy()
    out.var["mouse_ensembl_id"] = mouse_id.values
    out.var["mouse_gene_name"] = mouse_name.values
    out.var[id_col] = human_id.where(hit, mouse_id).values
    name = human_name.where(hit & human_name.ne(""), mouse_name)
    # An unmapped gene must not share a symbol with a human ortholog (e.g. a mouse gene named
    # "PISD" next to the ortholog of Pisd); it falls back to its mouse Ensembl ID.
    clash = ~hit & name.isin(set(name[hit]))
    out.var[name_col] = name.where(~clash, mouse_id).values
    out.var_names = pd.Index(out.var[id_col].astype(str), name=adata.var_names.name)
    if not out.var_names.is_unique:
        raise ValueError("gene IDs are not unique after mapping")
    out.uns["orthologs"] = json.dumps({
        "table": ORTHOLOGS.name, "source": "Ensembl Genes 114, one-to-one orthologs",
        "genes": int(len(hit)), "mapped_to_human": int(hit.sum()),
        "kept_mouse_ids": int((~hit).sum()),
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--orthologs", default=str(ORTHOLOGS))
    ap.add_argument("--id-col", default="ensembl_id")
    ap.add_argument("--name-col", default="gene_names")
    a = ap.parse_args()
    adata = anndata.read_h5ad(a.input)
    out = to_human_orthologs(adata, load_orthologs(Path(a.orthologs)), a.id_col, a.name_col)
    out.write_h5ad(a.output)
    print(out.uns["orthologs"])


if __name__ == "__main__":
    main()
