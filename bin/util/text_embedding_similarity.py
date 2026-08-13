"""
Biomedical text-encoder similarity for cell-type annotation evaluation.

This module provides a *mapping-free* complementary metric to the Cell-Ontology graph
metrics in ``cell_ontology_similarity.py``. It encodes predicted and true cell-type labels
with a biomedical text encoder (default: SapBERT) and scores them by cosine similarity.

Why it complements the ontology metrics
---------------------------------------
The ontology metrics (``rada``/``lin``/``id_accuracy``) only score labels that resolve to a
Cell-Ontology (CL) term, which in this pipeline requires the curated
``cell_type_ontology_mappings.csv`` + manual ``reviewed=True`` step. Free-text model output
that cannot be mapped is dropped. A text encoder scores *any* string, so it covers exactly
the cases the ontology path cannot.

Crucially, the encoder is the only metric that can put the **raw model output** and the
**ontology-mapped label** on the *same* scale (graph metrics cannot score non-CL raw labels;
exact-match on raw labels sits near the floor). We therefore report both, plus their
difference (``mapping_lift``) as a diagnostic of how much the curated mapping inflates
apparent semantic performance.

Caveats (intentional, surfaced in the metric)
---------------------------------------------
Cosine similarity from biomedical encoders is uncalibrated and anisotropic: even unrelated
terms score well above zero. We therefore also report a baseline-normalized score relative
to the mean cosine of random (true x pred) label pairs.

Depends only on ``transformers`` + ``torch`` (already in ``envs/evaluation.yml``); SapBERT is
a plain BERT, so we use [CLS]-token pooling per its documented usage. Sentence-transformer
models (e.g. ``all-MiniLM-L6-v2``) use mean pooling instead; pooling is auto-selected from
the model name and can be overridden.
"""
from __future__ import annotations

import os
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

    def __init__(self, model_name: str, cache_dir: Optional[Path] = None, pooling: Optional[str] = None):
        import torch  # local import so the module is importable without torch installed
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.pooling = pooling or _select_pooling(model_name)
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
    pooling: Optional[str] = None,
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
        _ENCODER_CACHE[key] = TextEncoder(model_name, cache_dir=cache_dir, pooling=pooling)
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
    """Mean of per-true-class means (equal weight per class), ignoring NaN — mirrors balanced_id_accuracy."""
    tmp = pd.DataFrame({"true": true_labels.to_numpy(), "sim": sims})
    tmp = tmp.dropna(subset=["sim"])
    if tmp.empty:
        return float("nan")
    per_class = tmp.groupby("true")["sim"].mean()
    return float(per_class.mean()) if len(per_class) else float("nan")


def normalize_similarity(avg: float, baseline: float) -> Tuple[float, bool]:
    """Rescale a raw cosine so the random-pair floor sits at 0 and an exact match at 1.

    Returns ``(value, clipped)``. ``clipped`` is True when the unclipped value fell outside
    [0, 1] — i.e. the model's output is no closer to the truth than randomly paired label
    names. That case is real (it happens for InstructCell and the chance baselines) and must
    stay visible, because a clipped score silently *understates* any difference computed from
    it, such as the mapping lift.
    """
    if np.isnan(avg) or np.isnan(baseline) or (1.0 - baseline) <= 1e-9:
        return float("nan"), False
    raw = (avg - baseline) / (1.0 - baseline)
    return float(np.clip(raw, 0.0, 1.0)), bool(raw < 0.0 or raw > 1.0)


# Keys produced by derive_text_emb_metrics(). aggregate.py backfills these for result JSONs
# written before they existed, so the list is the contract between the two modules.
TEXT_EMB_DERIVED_KEYS = (
    "text_emb_norm_balanced_raw",
    "text_emb_norm_balanced_mapped",
    "text_emb_norm_mapping_lift",
    "text_emb_norm_clipped",
    "text_emb_baseline_used",
)


def derive_text_emb_metrics(avgs: Dict[str, float],
                            baseline_override: Optional[float] = None) -> Dict[str, float]:
    """Derive the baseline-normalized, class-balanced scores and the normalized mapping lift.

    Pure arithmetic over values already in ``avgs`` — no encoder, no per-cell data — so
    ``aggregate.py`` can backfill result JSONs written before these keys existed instead of
    re-running the (GPU/torch-bound) scoring stage.

    Two defects in the original metric set are what this fixes:

    1. ``text_emb_mapping_lift`` was ``avg_mapped - avg_raw`` on the RAW cosine scale, while
       ``text_emb_norm_avg_*`` were baseline-normalized. The two differ by a factor of
       1/(1 - baseline), which is per-run (1.5x-3.7x across this benchmark's runs), so the
       stored lift could not be quoted beside the normalized scores. The lift returned here is
       a difference of normalized scores and is therefore on one scale with them.
    2. ``_norm`` was applied to the plain per-cell means, not the class-balanced ones, so the
       normalized scores were dominated by abundant cell types while every other headline
       metric in this benchmark is class-balanced. These use the balanced means.

    ``baseline_override`` replaces the per-run baseline (computed from that run's own predicted
    vocabulary, so every model gets a different denominator) with a fixed, model-independent
    one. Levels are only comparable across models under a fixed baseline; the lift is a
    within-model difference and is unaffected either way.
    """
    baseline = baseline_override if baseline_override is not None else avgs.get("text_emb_baseline")
    baseline = float("nan") if baseline is None else float(baseline)

    def _get(key):
        v = avgs.get(key)
        return float("nan") if v is None else float(v)

    norm_raw, clip_raw = normalize_similarity(_get("text_emb_balanced_similarity_raw"), baseline)
    norm_mapped, clip_mapped = normalize_similarity(_get("text_emb_balanced_similarity_mapped"), baseline)
    lift = (norm_mapped - norm_raw) if not (np.isnan(norm_raw) or np.isnan(norm_mapped)) else float("nan")
    return {
        "text_emb_norm_balanced_raw": norm_raw,
        "text_emb_norm_balanced_mapped": norm_mapped,
        "text_emb_norm_mapping_lift": lift,
        "text_emb_norm_clipped": float(clip_raw or clip_mapped),
        "text_emb_baseline_used": baseline,
    }


def compute_text_embedding_similarities(
    df: pd.DataFrame,
    true_col: str,
    raw_pred_col: str,
    mapped_pred_col: str,
    encoder: TextEncoder,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute text-encoder cosine similarity between true labels and both the raw and the
    ontology-mapped predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Per-cell dataframe with the columns named below (already label-normalized upstream).
    true_col : str
        Ground-truth label column.
    raw_pred_col : str
        Raw (post-normalization, pre-ontology-mapping) prediction column = actual model
        output, including unmappable free text.
    mapped_pred_col : str
        Ontology-mapped prediction column. If identical to raw_pred_col (mapping disabled),
        the mapped == raw and ``text_emb_mapping_lift`` is 0.
    encoder : TextEncoder
        A loaded encoder (load once per process via :func:`load_text_encoder`).

    Returns
    -------
    (combos_df, avg_dict)
        combos_df : one row per unique (true, raw_pred, mapped_pred) triple with per-pair
                    cosines and cell_count (for inspection / a side-car scored CSV).
        avg_dict  : scalar metrics (raw + mapped variants, baseline, mapping_lift).
    """
    true_s = _clean_labels(df[true_col])
    raw_s = _clean_labels(df[raw_pred_col])
    mapped_s = _clean_labels(df[mapped_pred_col])

    # Unique non-null vocabulary across all three columns; encode each unique string once.
    true_vocab = [x for x in pd.unique(true_s) if isinstance(x, str)]
    raw_vocab = [x for x in pd.unique(raw_s) if isinstance(x, str)]
    mapped_vocab = [x for x in pd.unique(mapped_s) if isinstance(x, str)]
    vocab = sorted(set(true_vocab) | set(raw_vocab) | set(mapped_vocab))

    if not vocab or not true_vocab:
        nan = float("nan")
        empty = pd.DataFrame(
            columns=[true_col, raw_pred_col, mapped_pred_col,
                     "text_emb_similarity_raw", "text_emb_similarity_mapped", "cell_count"]
        )
        return empty, {
            "text_emb_avg_similarity_raw": nan, "text_emb_balanced_similarity_raw": nan,
            "text_emb_norm_avg_raw": nan,
            "text_emb_avg_similarity_mapped": nan, "text_emb_balanced_similarity_mapped": nan,
            "text_emb_norm_avg_mapped": nan,
            "text_emb_baseline": nan, "text_emb_mapping_lift": nan,
            **{k: nan for k in TEXT_EMB_DERIVED_KEYS},
        }

    embs = encoder.encode(vocab)
    vec_of: Dict[str, np.ndarray] = {lab: embs[i] for i, lab in enumerate(vocab)}

    # Baseline = mean cosine over all (unique true) x (unique pred) cross-pairs ("random pair"
    # floor). Pred side = union of raw+mapped unique labels. Same baseline for both variants.
    pred_vocab = sorted(set(raw_vocab) | set(mapped_vocab))
    if true_vocab and pred_vocab:
        t_mat = np.stack([vec_of[x] for x in true_vocab])      # (nt, H), L2-normalized
        p_mat = np.stack([vec_of[x] for x in pred_vocab])      # (np, H)
        baseline = float((t_mat @ p_mat.T).mean())
    else:
        baseline = float("nan")

    def _norm(avg: float) -> float:
        return normalize_similarity(avg, baseline)[0]

    sims_raw = _per_row_cosine(true_s, raw_s, vec_of)
    sims_mapped = _per_row_cosine(true_s, mapped_s, vec_of)

    avg_raw = float(np.nanmean(sims_raw)) if np.isfinite(sims_raw).any() else float("nan")
    avg_mapped = float(np.nanmean(sims_mapped)) if np.isfinite(sims_mapped).any() else float("nan")

    avg_dict = {
        "text_emb_avg_similarity_raw": avg_raw,
        "text_emb_balanced_similarity_raw": _balanced_mean(sims_raw, true_s),
        "text_emb_norm_avg_raw": _norm(avg_raw),
        "text_emb_avg_similarity_mapped": avg_mapped,
        "text_emb_balanced_similarity_mapped": _balanced_mean(sims_mapped, true_s),
        "text_emb_norm_avg_mapped": _norm(avg_mapped),
        "text_emb_baseline": baseline,
        # Kept on the RAW cosine scale for backward compatibility with existing result JSONs.
        # Do NOT quote it beside the normalized scores -- use text_emb_norm_mapping_lift, which
        # is a difference of normalized, class-balanced scores (see derive_text_emb_metrics).
        "text_emb_mapping_lift": (avg_mapped - avg_raw)
        if not (np.isnan(avg_mapped) or np.isnan(avg_raw)) else float("nan"),
    }
    avg_dict.update(derive_text_emb_metrics(avg_dict))

    # Per-unique-combo table for inspection / side-car CSV.
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
    for col in ["text_emb_similarity_raw", "text_emb_similarity_mapped"]:
        combos_df[col] = combos_df[col].round(4)

    return combos_df, avg_dict
