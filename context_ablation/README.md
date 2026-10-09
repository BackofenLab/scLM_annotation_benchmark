# Context ablation: what does prompt context buy, and is per-cell prompting worth it?

Standalone study, independent of `main.nf`. It consumes the benchmark's cached artifacts and reuses
its scoring path, so every number lands on the same scale as the main table.

`config.yml` is the design record — it carries the reasoning for each choice, not just the values.

## Why this exists

Three things the paper currently **asserts** become measurable:

1. **The missing base-model control.** `MANUSCRIPT_SKELETON.md` D3 flags it as the objection a
   reviewer who knows these models will raise: we never ran the base LLM that C2S-Scale was
   fine-tuned from, so "domain adaptation gave no advantage" cannot be separated from "these base
   models are small".
2. **Context is not equal across models today.** Verified in code: C2S-Scale's template already
   injects `{organism}` *and* the candidate list, while the OpenRouter template exposes only
   `{num_genes}` / `{cell_sentence}` / `{unique_cell_types}`. The benchmark's comparison already
   differs in context and nobody has measured what that is worth.
3. **Per-cell vs per-cluster**, which the Introduction defends and D5 prices, is argued not shown.

## Design

| Arm | Stack | Model | Granularity | Template |
|---|---|---|---|---|
| A1 | OpenRouter | `openai/gpt-oss-120b` | cell | instruction |
| A2 | local | `google/gemma-2-2b` | cell | instruction (same as A1) |
| A3 | local | `vandijklab/C2S-Scale-Gemma-2-2B` | cell | C2S's own |
| A4 | OpenRouter | `openai/gpt-oss-120b` | **cluster** | marker genes |
| A5 | OpenRouter | `openai/gpt-oss-20b` | cell | instruction |

**A2 vs A3 is the comparison that carries the claim**, and both run locally on one stack so the
pairing cannot be contaminated by provider differences. **A1 vs A4 is the granularity contrast**,
same model on both sides so granularity is the only variable.

Modes, applied per arm:

| | candidate types | tissue + species | gene input |
|---|---|---|---|
| M1 | – | – | yes |
| M2 | yes | – | yes |
| M3 | – | yes | yes |
| M4 | yes | yes | yes |
| M5 | yes | yes | **no** |

M5 is the control that makes the rest readable: without it, a gain from naming the tissue cannot be
separated from the model shifting toward types common in that tissue.

## Decisions worth knowing before changing anything

**M1 and M2 reproduce the benchmark's prompts byte for byte**, and a test asserts it. That is what
licenses comparing any number here to the main table, and it is why the templates in `prompts.py`
are copied rather than paraphrased. Two details that took a correction to get right:

* The candidate list is **shuffled per cell** (`predict.py:477` permutes it from a seeded RNG).
  LLMs have position bias, so a fixed order would let a model score by favouring whatever sits
  first. `candidate_list()` is a separate call so it cannot be silently forgotten. The C2S
  template is the exception: the C2S-Scale wrapper sorts the list once for every cell, and
  `ordered_candidates()` does the same. Its organism is the subset's `organism` column, as there.
* The instruction template renders the list as a **Python repr** — `['neutrophil', 'b cell']`,
  brackets and quotes included — because the benchmark passes the list object into `str.format`.
  C2S's path uses `", ".join(...)` and renders bare. `render_candidates()` reproduces each.

**Context is inserted inline**, e.g. *"...this gene profile mouse nervous system represents..."*,
and tissue strings are truncated at the first parenthesis or colon. The YAML tissues differ wildly
in length (8 chars for `pancreas`, 159 for Zeisel's CNS/PNS enumeration); inserted verbatim, the
intervention would be a whole clause for Zeisel and one word for pancreas, and Zeisel would likely
show the largest gain for reasons unrelated to the model. Override per dataset with
`context_phrase` when the head noun is not the natural term.

**HVG selection uses `flavor="seurat"` on the context h5ads**, not `seurat_v3_paper`. That flavor
needs raw counts and the contexts carry only a log1p `X` — no layers, no raw. The full preprocessed
data would have counts but `cache/datasets/preprocessed/` is empty under the Zenodo fast-path, and
the `_random` subsets are not on the Zenodo record. 100k cells at near-natural composition beat 10k
stratified cells with the better flavor.

**The pipeline's `mt` / `ribo` / `hb` columns are not usable** and this study ignores them: HFRA and
HIHA have all three at zero, HIHA additionally flags 24,011 of 25,234 genes as `mito`, and
human_pancreas reports zero mitochondrial genes. Exclusion is recomputed from resolved gene
symbols, and `hvg.py` refuses to write a list containing any of those classes.

**Pool size is 5,000, and it truncates.** At that pool the share of cells whose sentence falls short
of the 200-gene cap is HFRA 49%, HIHA 27%, Zeisel 22%, human_pancreas 2%. Gene choice and sentence
length are therefore partly confounded; `sentences.py` records realized length per cell so this is
measurable rather than invisible, and the rate must be reported next to the result.

**Provider routing must be pinned before the full run.** The benchmark leaves `sort: price` with
`allow_fallbacks: true`, so requests go to whichever provider is cheapest at that moment and
different providers serve different quantizations. Tolerable for the benchmark's large effects; not
for an ablation measuring what a tissue string buys. Set `provider.only` to one provider.

**Seeds must differ per repeat.** `predict.py` injects the seed into the request payload, so
repeats at a fixed seed can return identical completions and would measure nothing.

**The local arms decode greedily** (`decoding: greedy` in config.yml), as the C2S-Scale authors
benchmark and as the benchmark's C2S wrapper does. Their seeds therefore matter only where the seed
changes the prompt: the shuffled genes of M6, the shuffled lists of the instruction template, and
the drawn lists of the imperfect-list study.

## Status

| Component | State |
|---|---|
| `config.yml` | design record, complete |
| `env.yml` | considered starting spec, **not yet installed or locked** |
| `src/hvg.py` | done; artifacts written to `cache/hvg/`, verified free of excluded classes |
| `src/prompts.py` | done; self-test covers the benchmark anchor and 160 whitespace combinations |
| `src/sentences.py` | written; **equivalence test not yet run** (needs `cell2sentence`) |
| `src/cluster.py` | written; not yet run (needs `leidenalg`) |
| `src/runner_cache.py` | done; self-test passes |
| `src/run_openrouter.py` | written; `--dry-run` works, no live call yet |
| `src/run_local.py` | written; not yet run (needs the torch env + weights) |
| `src/score.py` | written; wraps `bin/map_predictions.py` + `bin/evaluate.py`, both stages cached |
| `src/aggregate.py` | written; emits the tidy table and checks the 0.554 anchor automatically |

Everything parses and the two self-tests pass, but only `hvg.py` and `prompts.py` have been
exercised against real data. Nothing has made a live model call.

Two decisions still open, neither blocking:

* **Which Gemma-2-2B variant** C2S-Scale started from (base vs `-it`). No longer a manual check:
  `run_local.verify_base_model()` reads C2S-Scale's config, compares `_name_or_path` to the control
  arm, and **stops the run on a mismatch** — silently producing the wrong control is worse than not
  running. A missing field warns and continues, since absence is not evidence of mismatch.
* **Which provider to pin.** `run_openrouter.py` warns on every invocation while `provider.only` is
  empty.

## Caching

Both runners cache per PROMPT, keyed on `sha256(model | prompt | seed)`, in append-only JSONL.

File-level caching (what `main.nf` does) is useless at this scale: one (arm, mode, variant, seed)
file is 1,000 calls, so losing the last one would discard the other 999 — across ~150,000
generations and a metered API. Content-addressing also means **editing a template invalidates
exactly the prompts it changed**, with no manual cache-busting and no risk of serving answers to a
question we no longer ask, which is the failure mode that would quietly corrupt an ablation.

* The seed is in the key because it is sent with the request, also where it leaves the prompt
  unchanged (M1, M3). Greedy local arms are the exception: there the seed changes nothing, so
  their answers sit in a cache of their own (`<arm>__<dataset>__greedy.jsonl`), keyed without it.
* Failed calls are cached with their error, so a rerun does not re-pay for a prompt the provider
  will reject again.
* The provider that actually served each call is recorded, not the one requested.
* Writes are `fsync`ed per entry and result CSVs are written atomically, because a truncated
  predictions file scores silently on fewer cells whereas a missing one is obvious.
* `--dry-run` on either runner reports hits and misses before anything is spent.

## Running it: the work is split across two machines

| | arms | why |
|---|---|---|
| **laptop** (M2 Max, 96 GB, arm64) | A1, A4, A5 + prompt building, clustering, scoring | OpenRouter needs no GPU, no weights, no torch |
| **cluster** | A2, A3 | ~5 GB of checkpoint each plus a torch install |

The laptop has ample memory and a native arm64 stack, so a 2.6B model would run fine there in
principle — but only ~18 GB of disk is free, against ~17–19 GB for a torch env plus both
checkpoints. Repo cruft reclaims about 1 GB, which is not enough headroom to be worth the risk.

**Prompts are built once, on the laptop, and shipped.** Both runners consume the same files. This
is the point of the split rather than an inconvenience of it: `cell2sentence` defines the sentence
construction, so if the two machines resolved different versions the arms would silently read
different inputs and stop being comparable. Building prompts in one place makes that impossible.
It also means the cluster job needs neither `cell2sentence`, nor `scanpy`, nor the 1 GB contexts.

```bash
# --- laptop: core stack, no torch (~2 GB) --------------------------------
conda env create -f env.yml
conda activate sc_lfm_context_ablation
conda env export --no-builds > env.lock.yml      # commit this, it is the actual lock

python src/hvg.py                                 # one-off; writes ../cache/hvg/*.csv  [DONE]
python src/sentences.py --check-equivalence       # MUST pass before trusting any number
python src/sentences.py --materialize             # -> results/prompts/
python src/cluster.py                             # A4 clusters + marker prompts
python src/run_openrouter.py --host laptop        # A1, A4, A5

# --- cluster: adds the torch stack ---------------------------------------
conda env create -f env-local.yml
conda activate sc_lfm_context_ablation_local
python src/run_local.py --host cluster            # A2, A3, consuming results/prompts/

# --- back on the laptop ---------------------------------------------------
python src/score.py && python src/aggregate.py
```

`--check-equivalence` is not optional: it verifies that sentences built here are byte-identical to
the benchmark's for the same cells with HVG restriction off. If it fails, nothing from this study
may be compared to the main table.

## Deferred: the 27B pair

`google/gemma-2-27b-it` is on OpenRouter and `c2s_scale_27B` is already benchmarked, so a second
base-vs-adapted pair at 27B is available in principle. It is out of scope because C2S-Scale-27B does
not fit the available hardware, which would leave its half sourced from the existing benchmark row
(M1/M2 only, single run, C2S's own template) — a provisional pairing, not a matched experiment. See
the extension block in `config.yml` for the two other caveats (`-it` vs base, and cost).
