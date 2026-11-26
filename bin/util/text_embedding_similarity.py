"""
Biomedical text-encoder similarity for cell-type annotation evaluation.

This module provides a *mapping-free* complementary metric to the Cell-Ontology graph
metrics in ``cell_ontology_similarity.py``. It encodes predicted and true cell-type labels
with a biomedical text encoder (default: SapBERT) and scores them by cosine similarity.

Why it complements the ontology metrics
---------------------------------------
The ontology metrics (Lin, Rada, agreement) only credit labels that resolve to a Cell-Ontology
(CL) term, which in this pipeline requires the curated ``cell_type_ontology_mappings.csv`` +
manual ``reviewed=True`` step. A text encoder scores *any* string, so it also covers free-text
output the ontology path can only count as wrong.

Crucially, the encoder is the only metric that can put the **raw model output** and the
**ontology-mapped label** on the *same* scale (graph metrics cannot score non-CL raw labels;
exact-match on raw labels sits near the floor). We therefore report both, plus their
difference (``mapping_lift``) as a diagnostic of how much the curated mapping inflates
apparent semantic performance.

Caveats (intentional, surfaced in the metric)
---------------------------------------------
Cosine similarity from biomedical encoders is uncalibrated and anisotropic: even unrelated
terms score well above zero. Scores are therefore rescaled as in BERTScore (Zhang et al., ICLR
2020), (s - b) / (1 - b), with ONE baseline b per encoder: the mean cosine between two different
Cell Ontology cell-type names. Every model and dataset shares it, so rescaling changes how the
numbers read but never how models rank.

Depends only on ``transformers`` + ``torch`` (already in ``envs/evaluation.yml``); SapBERT is
a plain BERT, so we use [CLS]-token pooling per its documented usage. Sentence-transformer
models (e.g. ``all-MiniLM-L6-v2``) use mean pooling instead; pooling is auto-selected from
the model name and can be overridden.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Default biomedical entity encoder (UMLS self-aligned). Good for short cell-type strings.
DEFAULT_TEXT_ENCODER = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# Module-level cache so the (expensive) model load happens once per process even if
# load_text_encoder is called multiple times with the same args.
_ENCODER_CACHE: Dict[Tuple[str, str], "TextEncoder"] = {}


def _select_pooling(model_name: str) -> str:
    """Pick a sensible pooling for the encoder. SapBERT -> CLS; sentence-transformers -> mean."""
    name = model_name.lower()
    if "sapbert" in name:
        return "cls"
    if "sentence-transformers" in name or "minilm" in name or "mpnet" in name:
        return "mean"
    # Default for unknown plain BERT-style encoders: CLS (matches SapBERT-family usage).
    return "cls"


class TextEncoder:
    """Thin wrapper around a HuggingFace encoder that returns L2-normalized label embeddings."""

    def __init__(self, model_name: str, cache_dir: Optional[Path] = None):
        import torch  # local import so the module is importable without torch installed
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.pooling = _select_pooling(model_name)
        cache_dir_str = str(cache_dir) if cache_dir is not None else None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir_str)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir_str)
        self.model.eval()

        # cuda if available, else cpu. (mps is skipped to avoid op-coverage surprises on
        # short-text BERT; label counts are tiny so cpu is fine.)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self._torch = torch

    def encode(self, labels: List[str], batch_size: int = 64, max_length: int = 32) -> np.ndarray:
        """Encode a list of strings to an (N, H) array of L2-normalized embeddings."""
        torch = self._torch
        vecs = []
        with torch.no_grad():
            for i in range(0, len(labels), batch_size):
                batch = labels[i:i + batch_size]
                toks = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
                ).to(self.device)
                out = self.model(**toks)
                last = out.last_hidden_state  # (B, T, H)
                if self.pooling == "cls":
                    vec = last[:, 0]  # [CLS] token
                else:  # mean pooling over real tokens
                    mask = toks["attention_mask"].unsqueeze(-1).to(last.dtype)
                    vec = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                vec = torch.nn.functional.normalize(vec, p=2, dim=1)
                vecs.append(vec.cpu().numpy())
        if not vecs:
            hidden = self.model.config.hidden_size
            return np.empty((0, hidden), dtype=np.float32)
        return np.concatenate(vecs, axis=0).astype(np.float32)


def load_text_encoder(
    model_name: str = DEFAULT_TEXT_ENCODER,
    cache_dir: Optional[Path] = None,
) -> TextEncoder:
    """
    Load (and process-cache) a text encoder, downloading weights into ``cache_dir`` using the
    standard HuggingFace cache layout (``cache_dir/models--<org>--<name>/``), matching the
    existing ``cache/models--sentence-transformers--all-MiniLM-L6-v2/``.

    Raises on failure (missing package, download error) — callers should wrap in try/except
    to keep evaluation running.
    """
    key = (model_name, str(cache_dir) if cache_dir is not None else "")
    if key not in _ENCODER_CACHE:
        # Keep HF's hub cache co-located with the repo cache so containers/HPC reuse it.
        if cache_dir is not None:
            os.environ.setdefault("HF_HOME", str(cache_dir))
        _ENCODER_CACHE[key] = TextEncoder(model_name, cache_dir=cache_dir)
    return _ENCODER_CACHE[key]


def _clean_labels(series: pd.Series) -> pd.Series:
    """Coerce to string, treating NaN/blank as missing (np.nan) so they are skipped in means."""
    s = series.where(series.notna(), other=np.nan)
    s = s.astype("object")
    return s.map(lambda x: np.nan if (x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == "")
                 else str(x))


def _per_row_cosine(
    true_labels: pd.Series,
    pred_labels: pd.Series,
    vec_of: Dict[str, np.ndarray],
) -> np.ndarray:
    """Row-wise cosine for already-L2-normalized vectors (== dot product). NaN where missing."""
    out = np.full(len(true_labels), np.nan, dtype=np.float64)
    t_vals = true_labels.to_numpy()
    p_vals = pred_labels.to_numpy()
    for i in range(len(out)):
        t, p = t_vals[i], p_vals[i]
        if isinstance(t, float) and np.isnan(t):
            continue
        if isinstance(p, float) and np.isnan(p):
            continue
        tv = vec_of.get(t)
        pv = vec_of.get(p)
        if tv is None or pv is None:
            continue
        out[i] = float(np.dot(tv, pv))
    return out


def _balanced_mean(sims: np.ndarray, true_labels: pd.Series) -> float:
    """Mean of per-true-class means (equal weight per class), ignoring NaN, like balanced accuracy."""
    tmp = pd.DataFrame({"true": true_labels.to_numpy(), "sim": sims})
    tmp = tmp.dropna(subset=["sim"])
    if tmp.empty:
        return float("nan")
    per_class = tmp.groupby("true")["sim"].mean()
    return float(per_class.mean()) if len(per_class) else float("nan")


def reference_baseline(encoder: TextEncoder, names: Iterable[str],
                       cache_dir: Optional[Path] = None) -> float:
    """Mean cosine between two different reference names: the rescaling baseline.

    The BERTScore baseline is the mean score of random text pairs; with the full vocabulary at
    hand, the mean over all pairs is exact. For L2-normalized embeddings e_i it needs no n x n
    matrix: sum_{i != j} e_i.e_j = |sum_i e_i|^2 - sum_i |e_i|^2. The value is cached in
    ``cache_dir``, keyed by encoder and vocabulary, so a run encodes the vocabulary only once.
    """
    names = sorted(set(names))
    key = hashlib.sha1("\n".join([encoder.model_name, *names]).encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"text_emb_baseline_{key}.json" if cache_dir is not None else None
    if path is not None and path.exists():
        return float(json.loads(path.read_text())["baseline"])
    emb = encoder.encode(names).astype(np.float64)
    total = emb.sum(axis=0)
    n = len(names)
    baseline = float((total @ total - (emb * emb).sum()) / (n * (n - 1)))
    if path is not None:
        tmp = path.with_suffix(f".{os.getpid()}.tmp")  # atomic: parallel runs may race here
        tmp.write_text(json.dumps({"encoder": encoder.model_name, "n_names": n,
                                   "baseline": baseline}))
        tmp.replace(path)
    return baseline


def normalize_similarity(avg: float, baseline: float) -> Tuple[float, bool]:
    """Rescale a raw cosine so the baseline sits at 0 and an exact match at 1.

    Returns ``(value, clipped)``. ``clipped`` is True when the unclipped value fell outside
    [0, 1], i.e. the output is no closer to the truth than two random cell-type names. That case
    is real (InstructCell, the chance baselines) and stays visible, because a clipped score
    understates any difference computed from it, such as the mapping lift.
    """
    if np.isnan(avg) or np.isnan(baseline) or (1.0 - baseline) <= 1e-9:
        return float("nan"), False
    raw = (avg - baseline) / (1.0 - baseline)
    return float(np.clip(raw, 0.0, 1.0)), bool(raw < 0.0 or raw > 1.0)


# Result keys, in output order: the contract with aggregate.py.
TEXT_EMB_KEYS = (
    "text_emb_balanced_similarity_raw",
    "text_emb_balanced_similarity_mapped",
    "text_emb_baseline",
    "text_emb_norm_balanced_raw",
    "text_emb_norm_balanced_mapped",
    "text_emb_norm_mapping_lift",
    "text_emb_norm_clipped",
)


def compute_text_embedding_similarities(
    df: pd.DataFrame,
    true_col: str,
    raw_pred_col: str,
    mapped_pred_col: str,
    encoder: TextEncoder,
    baseline: float,
    no_answer: Iterable[str] = (),
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Cosine similarity between the true labels and both the raw and the mapped predictions.

    ``raw_pred_col`` is the model's own output (normalized, before ontology mapping), including
    free text; ``mapped_pred_col`` is the ontology-mapped label. Means are class-balanced, like
    balanced accuracy, and rescaled with ``baseline`` (see ``reference_baseline``). The lift is
    the difference of the two rescaled scores, so it is on their scale. ``no_answer`` holds the
    placeholders for a missing or unusable answer. They are not encoded as if they were labels but
    score the baseline, i.e. like a random cell-type name (0 after rescaling): in both variants
    when the model gave no answer, in the mapped one only when just the mapping is missing.

    Returns one row per unique (true, raw, mapped) triple with its cosines and cell count, and
    the scalar metrics named in ``TEXT_EMB_KEYS``.
    """
    true_s = _clean_labels(df[true_col])
    raw_s = _clean_labels(df[raw_pred_col])
    mapped_s = _clean_labels(df[mapped_pred_col])

    no_answer = set(no_answer)
    vocab = sorted({x for col in (true_s, raw_s, mapped_s) for x in pd.unique(col)
                    if isinstance(x, str) and x not in no_answer})
    embs = encoder.encode(vocab)
    vec_of: Dict[str, np.ndarray] = {lab: embs[i] for i, lab in enumerate(vocab)}

    none_given = raw_s.isin(no_answer).to_numpy()
    sims_raw = _per_row_cosine(true_s, raw_s, vec_of)
    sims_mapped = _per_row_cosine(true_s, mapped_s, vec_of)
    sims_raw[none_given] = baseline
    sims_mapped[none_given | mapped_s.isin(no_answer).to_numpy()] = baseline
    balanced_raw = _balanced_mean(sims_raw, true_s)
    balanced_mapped = _balanced_mean(sims_mapped, true_s)
    norm_raw, clip_raw = normalize_similarity(balanced_raw, baseline)
    norm_mapped, clip_mapped = normalize_similarity(balanced_mapped, baseline)
    metrics = dict(zip(TEXT_EMB_KEYS, (
        balanced_raw, balanced_mapped, baseline, norm_raw, norm_mapped,
        norm_mapped - norm_raw, float(clip_raw or clip_mapped))))

    combos = pd.DataFrame({
        true_col: true_s.to_numpy(),
        raw_pred_col: raw_s.to_numpy(),
        mapped_pred_col: mapped_s.to_numpy(),
        "text_emb_similarity_raw": sims_raw,
        "text_emb_similarity_mapped": sims_mapped,
    })
    combos_df = (
        combos.groupby([true_col, raw_pred_col, mapped_pred_col], dropna=False)
        .agg(text_emb_similarity_raw=("text_emb_similarity_raw", "first"),
             text_emb_similarity_mapped=("text_emb_similarity_mapped", "first"),
             cell_count=("text_emb_similarity_raw", "size"))
        .reset_index()
    )
    return combos_df, metrics
