# Resources

Versioned inputs of the pipeline.

| File | Content |
|---|---|
| `cell_type_ontology_mappings.csv` | answers that are not Cell Ontology terms, with the term each one stands for and whether a person has checked it (see the main README) |
| `all_cell_types.txt` | the names of the cell types of the Cell Ontology, release 2025-10-16, one per line; CellWhisperer, LangCell and mmContext choose from them in the unconstrained setting |
| `ensembl/mouse_human_one2one_orthologs_ensembl114.tsv` | one-to-one mouse-human orthologs from Ensembl release 114, for methods that read human genes only; the `.json` file records how it was queried |
| `ensembl/human_pancreas_ensembl_260423.csv` | Ensembl ids of the gene symbols of the human pancreas dataset, used when Ensembl BioMart cannot be reached |

The Cell Ontology itself is downloaded into `cache/` at the release fixed in
`bin/util/cell_ontology_similarity.py`.
