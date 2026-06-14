import matplotlib.pyplot as plt
# Some LLMs emit garbage labels containing '$', which matplotlib's mathtext
# parser tries to interpret as LaTeX math and crashes on. Treat '$' as literal.
plt.rcParams['text.parse_math'] = False
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

from util.pub_style import set_pub_style
from util.color_definition import COLORMAPS

class UMAPCellPlotter:
    def __init__(
        self, 
        colormap_name="CM: Teal–White",
        point_size=10,
        annotate_centroids=True,
        add_umap_arrows=True,
        font_size=8
    ):
        set_pub_style()
        self.colormap = COLORMAPS.get(colormap_name, plt.get_cmap("viridis"))
        self.point_size = point_size
        self.annotate_centroids = annotate_centroids
        self.add_umap_arrows = add_umap_arrows
        self.font_size = font_size

        self.font_settings = {
            "fontsize": self.font_size,
            "fontweight": "normal",
            "family": "Arial"
        }
        self.title_font_settings = {
            **self.font_settings,
            "fontsize": self.font_size + 1
        }
        self.large_title_font_settings = {
            **self.font_settings,
            "fontsize": self.font_size + 2
        }
        

    def plot_cm(
        self,
        true_labels,
        pred_labels,
        output_path,
        title="Confusion Matrix",
        normalize=True
    ):
        """
        Plots a confusion matrix heatmap comparing true and predicted labels.

        Parameters:
        - true_labels (array-like): Ground truth labels.
        - pred_labels (array-like): Predicted labels.
        - output_path (str or Path): File path to save the confusion matrix figure.
        - title (str): Title for the plot.
        - normalize (bool): Whether to normalize the confusion matrix.
        - cmap (str): Matplotlib colormap name.
        """
        
        cmap = self.colormap
        
        true_labels = pd.Series(true_labels).astype(str)
        pred_labels = pd.Series(pred_labels).astype(str)

        # Unified label set for consistent axis order
        labels = sorted(set(true_labels) | set(pred_labels))

        # Compute confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=labels)

        # Normalize if requested
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)  # handle divide-by-zero cases

        # Adjust figure size based on number of labels
        width_per_label = 0.7
        height_per_label = 0.7
        fig_width = max(6, width_per_label * len(labels))
        fig_height = max(6, height_per_label * len(labels))

        plt.figure(figsize=(fig_width, fig_height))
        ax = sns.heatmap(
            cm,
            annot=True,
            square=True,
            fmt=".2f" if normalize else "d",
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Proportion" if normalize else "Count", "shrink": 0.6,"aspect": 20},
            annot_kws={"size": self.font_size},
            vmax=1.0,
            vmin=0,
        )
        
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 0.5, 1])
        cbar.outline.set_edgecolor("black")
        
        plt.xlabel("Predicted Label", fontsize=self.font_size)
        plt.ylabel("True Label", fontsize=self.font_size)
        plt.title(title, **self.font_settings)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()