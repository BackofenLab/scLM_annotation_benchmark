# from sklearn.metrics import confusion_matrix

# sys.path.insert(0, "../")

from pathlib import Path

# sc.set_figure_params(figsize=(6, 6))
import scanpy as sc
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix


def make_umap_plot(predictions, labels, results, tr_valid_sets):

    adata_test_raw = tr_valid_sets["adata_test_raw"]
    # adata_test = tr_valid_sets["adata_test"]
    id2type = tr_valid_sets["id2type"]
    celltypes = tr_valid_sets["celltypes"]
    save_dir = Path(tr_valid_sets["save_dir"])
    # predictions, labels, results = test(best_model, adata_test)
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
            color=["celltype", "predictions"],
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


def make_confusion_plot(predictions, labels, tr_valid_sets):
    id2type = tr_valid_sets["id2type"]
    celltypes = list(tr_valid_sets["celltypes"])
    save_dir = Path(tr_valid_sets["save_dir"])
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
