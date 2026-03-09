# Optimize openrouter_LLM predict.py — Design

## Problem

predict.py is ~837 lines with massive parameter-passing boilerplate (30+ params threaded through 3 layers). Models run sequentially. Fixed `delay_between_requests` wastes time when no rate limit is hit.

## Context

- OpenRouter imposes no platform-level rate limits for paid models. Limits come from upstream providers, handled via automatic fallback routing.
- Target models span 6 providers: Meta/Llama, Anthropic, Google, DeepSeek, OpenAI, Mistral — their rate limit pools are independent.
- Nextflow already parallelizes across datasets via `.combine()`.

## Design

### 1. predict.py — single model, config dicts

- Accept `--model` (singular) instead of `--models`. One invocation = one model = one CSV.
- Replace 30+ individual parameters with three config dicts built in `main()`:
  - `provider_cfg`: order, only, ignore, allow_fallbacks, sort, quantizations, max_price, etc.
  - `generation_cfg`: max_tokens, temperature, top_p, top_k, seed, penalties, stop, stream, response_format.
  - `reasoning_cfg`: effort, max_tokens, exclude.
- `query_openrouter_model` and `run_OpenRouter_model_prediction` receive these dicts and unpack internally.
- Remove `model_to_slug` from predict.py — run.sh handles filenames.
- Estimated reduction: ~837 lines to ~400-500 lines.

### 2. Smart rate limiting

- Remove `delay_between_requests` entirely.
- Increase default `max_workers` from 3 to 10.
- Increase `max_retries` from 3 to 5.
- Keep existing adaptive mechanisms: `Retry-After` header respect, exponential backoff on 429/5xx.
- On 429, temporarily pause new request submissions for the `Retry-After` duration.
- Guarantee: a request needs 5 consecutive failures across multiple providers to be lost — effectively impossible for paid models.

### 3. run.sh — parallel model execution

- Read `model_names` list from model_cfg.yml (already implemented).
- Loop over models and invoke predict.py once per model in parallel (`&` + `wait`).
- Construct per-model output filenames with slug (e.g. `predictions_celltype_meta-llama--llama-3.1-8b-instruct.csv`).
- Each predict.py process has independent adaptive backoff — no cross-model interference.

### 4. model_cfg.yml

- Remove `delay_between_requests`.
- Change `max_workers` default to 10.
- `model_names` list unchanged.

### 5. nextflow.config

- Add `openrouter_LLM` to `params.models` default list.
- Remove `openrouter_LLM` from `params.throttled_models`.
- openrouter_LLM runs through `run_model` with full Nextflow parallelism across datasets.

## Files changed

| File | Change |
|---|---|
| predict.py | Single model, config dicts, adaptive backoff, remove delay |
| run.sh | Parallel model invocation via & + wait |
| model_cfg.yml | Remove delay_between_requests, max_workers to 10 |
| nextflow.config | Move openrouter_LLM from throttled to normal models |

## Files NOT changed

- main.nf
- Evaluation pipeline
- C2S dataset building logic
- CL mapping logic
