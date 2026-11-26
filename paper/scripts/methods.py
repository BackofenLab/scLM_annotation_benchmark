"""Definitions shared by the figure and table scripts, and the computations on the result tables
that several of them use.

A method is named by its base name, such as ``gpt-5.5``. A run of a method carries its setting as
a suffix: ``_blind`` for the unconstrained run and ``_unblind`` for the run constrained by the list
of expected cell types. The chance model has no suffix. The figure modules, the table scripts,
extract_statistics.py, rare_type_table.py and paper/experiments/src/lists.py import from this
module, which needs no matplotlib.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression

import paths

CHANCE_MODEL = "random_stratified"

# Families in the order of every figure, with the name the figures print for them.
FAMILIES = [("classical", "baselines"), ("scLLM", "scLMs"), ("LLM", "LLMs")]
# Subcategories of each family: (key, name, marker, marker area in points squared). Marker shapes
# are unique across families, so a marker identifies its subcategory on its own.
CATEGORY_TIERS = {
    "LLM": [
        ("closed_reasoning", "closed-weight reasoning", "*", 150),
        ("open_reasoning", "open-weight reasoning", "^", 52),
        ("open_nonreasoning", "open-weight non-reasoning", "s", 40),
    ],
    "scLLM": [
        ("generative", "generative (free text)", "o", 46),
        ("retrieval", "embedding (closed set)", "v", 52),
    ],
    "classical": [
        ("marker", "marker-set scoring", "D", 38),
        ("reference", "reference-based classifier", "P", 62),
    ],
}
# The methods of every subcategory. LLM names are the OpenRouter model names without the vendor.
CATEGORY_TIER_MEMBERS = {
    "LLM": {
        "closed_reasoning": {"claude-opus-4.7", "claude-sonnet-4.6", "gemini-3.1-pro-preview",
                             "gpt-5.5"},
        "open_reasoning": {"deepseek-v3.2", "gpt-oss-120b", "gpt-oss-20b", "kimi-k2.6"},
        "open_nonreasoning": {"llama-3.1-70b-instruct", "llama-3.3-70b-instruct",
                              "llama-4-maverick"},
    },
    "scLLM": {
        "generative": {"c2s_scale_2B", "c2s_scale_27B", "instructcell"},
        "retrieval": {"cellwhisperer", "langcell", "mmcontext"},
    },
    "classical": {
        "marker": {"scsa", "sctype"},
        "reference": {"celltypist", "singler"},
    },
}
_CLASSES = {base: (family, tier) for family, tiers in CATEGORY_TIER_MEMBERS.items()
            for tier, members in tiers.items() for base in members}

# A run's setting, from its suffix. Figures and tables tag a run with the single letter and spell
# it out in a key.
REGIME = {"blind": "unconstrained", "unblind": "constrained"}
REGIME_SHORT = {"blind": "U", "unblind": "C"}
GROUP_FAMILY = {"classical": "baselines", "scLLM": "scLLM", "LLM": "LLM"}
# The method groups as rows of a figure, every family in both settings: (family, setting, label).
MAPPING_ROWS = [(family, mode, f"{name}, {REGIME_SHORT[mode]}")
                for family, name in FAMILIES for mode in ("blind", "unblind")]

DISPLAY_NAMES = {
    "celltypist": "CellTypist", "scsa": "SCSA", "sctype": "scType", "singler": "SingleR",
    "c2s_scale_2B": "C2S-Scale 2B", "c2s_scale_27B": "C2S-Scale 27B",
    "cellwhisperer": "CellWhisperer", "langcell": "LangCell", "mmcontext": "mmContext",
    "instructcell": "InstructCell", "gpt-5.5": "GPT-5.5", "gpt-oss-20b": "GPT-OSS-20B",
    "gpt-oss-120b": "GPT-OSS-120B", "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "claude-opus-4.7": "Claude Opus 4.7", "claude-sonnet-4.6": "Claude Sonnet 4.6",
    "deepseek-v3.2": "DeepSeek-V3.2", "kimi-k2.6": "Kimi-K2.6",
    "llama-3.1-70b-instruct": "Llama-3.1-70B", "llama-3.3-70b-instruct": "Llama-3.3-70B",
    "llama-4-maverick": "Llama-4-Maverick", CHANCE_MODEL: "Random Stratified",
}
DATASET_ORDER = ["HFRA", "HIHA", "Zeisel", "human_pancreas"]
# Dataset names for a reader; the other keys are already the names.
DATASET_DISPLAY = {"human_pancreas": "human pancreas"}

# The metrics of the metric comparison and the result tables, in their groups.
METRIC_GROUPS = [
    ("exact match", [("balanced_accuracy", "balanced accuracy"), ("accuracy", "accuracy"),
                     ("macro_f1", "macro F1"), ("weighted_f1", "weighted F1")]),
    ("ontology-aware", [("agreement_score", "agreement score"), ("rada_avg_similarity", "Rada"),
                        ("lin_avg_similarity", "Lin")]),
    ("text similarity", [("text_emb_norm_balanced_mapped", "text, mapped"),
                         ("text_emb_norm_balanced_raw", "text, raw")]),
]
# Where an answer sits in the Cell Ontology relative to the true type (the ``lineage`` column of
# the confusion pairs), in stacking order.
ANSWER_TYPES = ["exact", "coarser", "broad", "finer", "other", "none"]

# The conditions of the prompt-context ablation in the order of the figures, from the least to
# the most the model is given, and their labels.
MODE_ORDER = ["M6", "M1", "M3", "M2", "M4"]
MODE_LABELS = {"M6": "genes,\nshuffled\norder", "M1": "genes\nonly", "M3": "genes\n+ tissue",
               "M2": "genes\n+ list", "M4": "genes\n+ tissue\n+ list"}
# The arms of the ablation and of the list study as the benchmark's methods.
STUDY_ARM_MODEL = {"A1_llm_percell": "gpt-oss-120b", "A4_llm_percluster": "gpt-oss-120b",
                   "A8_gemini_percluster": "gemini-3.1-pro-preview",
                   "A3_scllm_percell": "c2s_scale_2B", "S1_mmcontext": "mmcontext",
                   "S2_langcell": "langcell", "B1_singler": "singler",
                   "B2_celltypist": "celltypist", "B3_sctype": "sctype", "B4_scsa": "scsa"}


def derive_mode(model):
    """Setting of a run from its suffix: "blind", "unblind", or "baseline" for the chance model."""
    m = str(model).lower()
    if m.endswith("_unblind"):
        return "unblind"
    if m.endswith("_blind"):
        return "blind"
    return "baseline"


def model_base(model):
    """Method of a run: the run's name without its setting."""
    m = str(model)
    for suffix in ("_blind", "_unblind"):
        if m.endswith(suffix):
            return m[: -len(suffix)]
    return m


def categorize_model(model):
    """Family of a run: "classical", "scLLM", "LLM", or "chance" for the chance model."""
    base = model_base(model)
    return "chance" if base == CHANCE_MODEL else _CLASSES[base][0]


def model_tier(model):
    """Subcategory of a run within its family; None for the chance model."""
    base = model_base(model)
    return None if base == CHANCE_MODEL else _CLASSES[base][1]


def run_label(model, name=None):
    """``name`` (by default the method) with the run's setting as U or C."""
    name = model_base(model) if name is None else name
    short = REGIME_SHORT.get(derive_mode(model))
    return f"{name} ({short})" if short else name


def group_name(model):
    """Family and setting of a run, e.g. "constrained baselines" or "unconstrained LLM"."""
    return f"{REGIME[derive_mode(model)]} {GROUP_FAMILY[categorize_model(model)]}"


def _display(base):
    """Name of a method as the paper writes it."""
    return DISPLAY_NAMES[base]


def dataset_label(name):
    """Name of a dataset as the paper writes it."""
    return DATASET_DISPLAY.get(str(name), str(name))


def _list_study_arms(runs):
    """The list study's arms in the order of the figures: family, subcategory, then method."""
    order = {(family, tier): (i, j) for i, (family, _) in enumerate(FAMILIES)
             for j, (tier, *_rest) in enumerate(CATEGORY_TIERS[family])}
    model = {arm: STUDY_ARM_MODEL[arm] for arm in runs["arm"].unique()}
    return sorted(model, key=lambda a: (*order[(categorize_model(model[a]),
                                                model_tier(model[a]))], model[a]))


def load_aggregate(path, subset_size=1000):
    """The benchmark metrics, one row per run and dataset, at ``subset_size`` cells or, with
    None, at every size."""
    df = pd.read_csv(path)
    if subset_size is not None:
        df = df[df["subset_size"] == subset_size].copy()
    return df


def load_confusion_pairs(subset_size=1000):
    """Cells per run, dataset, true type and answer at ``subset_size`` cells."""
    d = pd.read_csv(paths.CONFUSION_PAIRS)
    return d[d["subset_size"] == subset_size].drop(columns="subset_size").reset_index(drop=True)


def _benchmark_pairs(df, pairs=None, chance=False):
    """The confusion pairs of the runs in ``df``, those of the chance model only with ``chance``."""
    pairs = load_confusion_pairs() if pairs is None else pairs
    runs = df["model"] if chance else df.loc[df["model"].map(categorize_model) != "chance", "model"]
    return pairs[pairs["model"].isin(set(runs))]


def _class_balanced(pairs, counts):
    """Per run and dataset, the mean over true cell types of each count's share of the type's
    cells, the way balanced accuracy averages recall. ``counts`` maps a name to a count per pair."""
    keys = [pairs["model"], pairs["dataset"], pairs["true_label"]]
    per_type = pd.DataFrame(counts).groupby(keys).sum()
    return per_type.div(pairs["n_cells"].groupby(keys).sum(), axis=0) \
                   .groupby(["model", "dataset"]).mean()


def _group_means(per_model, rows):
    """Mean of ``per_model`` (one row per run) over the runs of each ``(family, setting, label)``
    of ``rows``, or of each ``(family, setting, label, subcategory)``."""
    means = []
    for family, mode, _label, *tier in rows:
        keep = [categorize_model(m) == family and derive_mode(m) == mode
                and (not tier or model_tier(m) == tier[0]) for m in per_model.index]
        means.append(per_model[keep].mean())
    return pd.DataFrame(means, index=[row[2] for row in rows])


def _mapping_scores(pairs):
    """Balanced accuracy per run and dataset with the curated mapping, and counting only the
    answers that needed none: Cell Ontology terms or the dataset's own labels, as written."""
    hit = pairs["true_label"] == pairs["predicted_label"]
    return _class_balanced(pairs, {"with": pairs["n_cells"].where(hit, 0),
                                   "without": pairs["n_direct"].where(hit, 0)})


def _needs_mapping(pairs):
    """Share of cells per run and dataset whose answer needed the curated mapping, over all cells
    rather than per cell type."""
    cells = pairs.groupby(["model", "dataset"])[["n_cells", "n_direct"]].sum()
    return 1.0 - cells["n_direct"] / cells["n_cells"]


def _answer_shares(pairs):
    """Class-balanced share of each of ANSWER_TYPES per run and dataset."""
    return _class_balanced(pairs, {k: pairs["n_cells"].where(pairs["lineage"] == k, 0)
                                   for k in ANSWER_TYPES})


def _unlisted(pairs):
    """Class-balanced share of cells per run and dataset whose answer names a cell type that is not
    on the list, the run's own true types. Both columns are Cell Ontology names, so another name
    for a listed type counts as listed. Meaningful for the constrained runs, which got the list."""
    named = pd.MultiIndex.from_frame(pairs[["model", "dataset", "predicted_label"]])
    listed = pd.MultiIndex.from_frame(pairs[["model", "dataset", "true_label"]])
    unlisted = (pairs["lineage"] != "none") & ~named.isin(listed)
    return _class_balanced(pairs, {"unlisted": pairs["n_cells"].where(unlisted, 0)})["unlisted"]


def spread(targets, gap):
    """Positions as close as possible to ``targets`` (least squares) that keep their order and lie
    at least ``gap`` apart, for labels that would otherwise overlap: isotonic regression of the
    sorted targets minus their rank offsets."""
    targets = np.asarray(targets, float)
    order = np.argsort(targets, kind="stable")
    steps = np.arange(len(targets)) * gap
    placed = np.empty(len(targets))
    placed[order] = isotonic_regression(targets[order] - steps).x + steps
    return placed
