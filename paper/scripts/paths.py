"""Locations of the paper's inputs, results and outputs inside the repository."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PIPELINE_BIN = REPO / "bin"
PAPER = REPO / "paper"
DATA = PAPER / "data"
RESULTS = PAPER / "results"
EXPERIMENTS = PAPER / "experiments"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"

BENCHMARK_METRICS = RESULTS / "benchmark_metrics.csv"
PER_CLASS_SCORES = RESULTS / "per_class_scores.csv"
CONFUSION_PAIRS = RESULTS / "confusion_pairs.csv"
RARE_TYPE_OUTCOMES = RESULTS / "rare_type_outcomes.csv"
STATISTICS = RESULTS / "statistics"

DATASET_FACTS = DATA / "dataset_facts.csv"
MODEL_FACTS = DATA / "model_facts.csv"
OPENROUTER_ACTIVITY = DATA / "openrouter_activity.csv"
CELL_TYPE_DISTRIBUTIONS = DATA / "cell_type_distributions"

ABLATION_RESULTS = EXPERIMENTS / "results" / "context_ablation.csv"
LIST_RUNS = EXPERIMENTS / "results_lists" / "runs.csv"
