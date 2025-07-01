from pathlib import Path

# sc.set_figure_params(figsize=(6, 6))
import scanpy as sc
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix


def make_umap_plot(predictions, labels, results, proc_datasets):

    adata_test_raw = proc_datasets.adata_test_raw
    id2type = proc_datasets.id2type
    celltypes = list(proc_datasets.celltypes)
    save_dir = Path(proc_datasets.save_dir)

    adata_test_raw.obs["predictions"] = [id2type[p] for p in predictions]

    # plot
    palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    palette_ = (
        plt.rcParams["axes.prop_cycle"].by_key()["color"]
        + plt.rcParams["axes.prop_cycle"].by_key()["color"]
        + plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    palette_ = {c: palette_[i] for i, c in enumerate(celltypes)}

    with plt.rc_context({"figure.figsize": (6, 4), "figure.dpi": (300)}):
        sc.pl.umap(
            adata_test_raw,
            color=[proc_datasets.celltype_key, "predictions"],
            palette=palette_,
            show=False,
        )
        plt.savefig(save_dir / "results.png", dpi=300)

    save_dict = {
        "predictions": predictions,
        "labels": labels,
        "results": results,
        "id_maps": id2type,
    }
    with open(save_dir / "results.pkl", "wb") as f:
        pickle.dump(save_dict, f)


def make_confusion_plot(predictions, labels, proc_datasets):
    id2type = proc_datasets.id2type
    celltypes = list(proc_datasets.celltypes)
    save_dir = Path(proc_datasets.save_dir)
    for i in set([id2type[p] for p in predictions]):
        if i not in celltypes:
            celltypes.remove(i)
    cm = confusion_matrix(labels, predictions)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm = pd.DataFrame(
        cm, index=celltypes[: cm.shape[0]], columns=celltypes[: cm.shape[1]]
    )
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt=".1f", cmap="Blues")
    plt.savefig(save_dir / "confusion_matrix.png", dpi=300)
