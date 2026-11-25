# Final OpenRouter activity CSV — build audit

Rows written: **95,928** to `openrouter_activity_merged.csv` (24 model×mode groups, 12 models).
Exact-duplicate rows removed before trimming: 1,999 (merged.csv's gpt-5.5 blind block vs. the 05-23 export).
Total spend: **$1,193.91** (benchmarked models only: $1,193.77).

Sources: `openrouter_activity_merged.csv` (HEAD, unblind campaign + opus blind), `openrouter_activity_2026-06-11.csv` (blind campaign), `openrouter_activity_2026-05-23_gpt55_xhigh.csv` (gpt-5.5 blind, separate API key).

Trim rule: each (model, mode) capped at 3,997 calls (4 datasets × ~1000 cells). Rows are split into run attempts by a 45-min idle gap; attempts covering a whole number of dataset runs are kept (most recent first), aborted partials are dropped first, then failed (non-`stop`) calls, then the oldest rows. The log carries no cell id, so the exact mapping from calls to the cached predictions is unrecoverable — this is a documented convention, not a reconstruction.

`run attempts` shows the sub-block sizes found in the raw log (e.g. `34+999+657+1999+1001` = five attempts).

| model | mode | rows before | run attempts | rows kept | dropped | of which failed | dropped $ | kept span | cost | in benchmark |
|---|---|---:|---|---:|---:|---:|---:|---|---:|:--:|
| `anthropic/claude-4.6-sonnet-20260217` | blind | 3,997 | 3997 | 3,997 |  |  |  | 2026-05-15 13:40 → 2026-05-15 15:11 | $53.35 | yes |
| `anthropic/claude-4.6-sonnet-20260217` | unblind | 3,997 | 1998+1000+999 | 3,997 |  |  |  | 2026-05-09 12:22 → 2026-05-09 15:59 | $42.02 | yes |
| `anthropic/claude-4.7-opus-20260416` | blind | 3,997 | 3997 | 3,997 |  |  |  | 2026-05-10 17:06 → 2026-05-10 17:16 | $27.36 | yes |
| `anthropic/claude-4.7-opus-20260416` | unblind | 3,997 | 999+1000+999+999 | 3,997 |  |  |  | 2026-05-07 16:56 → 2026-05-08 11:21 | $32.62 | yes |
| `deepseek/deepseek-v3.2-20251201` | blind | 3,998 | 3998 | 3,997 | 1 | 1 | $0.00 | 2026-05-15 15:17 → 2026-05-15 21:55 | $3.37 | yes |
| `deepseek/deepseek-v3.2-20251201` | unblind | 3,997 | 3997 | 3,997 |  |  |  | 2026-05-06 11:25 → 2026-05-06 22:24 | $4.94 | yes |
| `google/gemini-3.1-pro-preview-20260219` | blind | 3,997 | 3997 | 3,997 |  |  |  | 2026-05-15 13:40 → 2026-05-15 15:04 | $38.23 | yes |
| `google/gemini-3.1-pro-preview-20260219` | unblind | 3,997 | 2789+1208 | 3,997 |  |  |  | 2026-05-07 05:07 → 2026-05-07 07:25 | $34.24 | yes |
| `meta-llama/llama-3.1-70b-instruct` | blind | 3,997 | 3997 | 3,997 |  |  |  | 2026-05-15 13:40 → 2026-05-15 14:55 | $1.31 | yes |
| `meta-llama/llama-3.1-70b-instruct` | unblind | 3,997 | 3997 | 3,997 |  |  |  | 2026-05-06 10:55 → 2026-05-06 11:06 | $1.57 | yes |
| `meta-llama/llama-3.1-8b-instruct` | blind | 3,997 | 3997 | 3,997 |  |  |  | 2026-05-15 13:40 → 2026-05-15 14:54 | $0.07 | **no** |
| `meta-llama/llama-3.1-8b-instruct` | unblind | 3,997 | 3997 | 3,997 |  |  |  | 2026-05-06 10:55 → 2026-05-06 11:06 | $0.08 | **no** |
| `meta-llama/llama-3.3-70b-instruct` | blind | 3,997 | 3997 | 3,997 |  |  |  | 2026-05-15 13:40 → 2026-05-15 14:55 | $0.33 | yes |
| `meta-llama/llama-3.3-70b-instruct` | unblind | 3,997 | 3997 | 3,997 |  |  |  | 2026-05-06 10:55 → 2026-05-06 11:06 | $0.40 | yes |
| `meta-llama/llama-4-maverick-17b-128e-instruct` | blind | 3,997 | 3997 | 3,997 |  |  |  | 2026-05-15 13:40 → 2026-05-15 14:54 | $0.57 | yes |
| `meta-llama/llama-4-maverick-17b-128e-instruct` | unblind | 3,997 | 3997 | 3,997 |  |  |  | 2026-05-06 10:55 → 2026-05-06 11:05 | $1.03 | yes |
| `moonshotai/kimi-k2.6-20260420` | blind | 4,678 | 2671+2007 | 3,997 | 681 | 460 | $35.07 | 2026-05-16 19:16 → 2026-05-19 17:14 | $176.46 | yes |
| `moonshotai/kimi-k2.6-20260420` | unblind | 3,997 | 3997 | 3,997 |  |  |  | 2026-05-07 16:56 → 2026-05-09 00:48 | $96.68 | yes |
| `openai/gpt-5.5-20260423` | blind | 4,690 | 34+999+657+1999+1001 | 3,997 | 693 | 3 | $116.89 | 2026-05-22 17:53 → 2026-05-26 12:17 | $523.23 | yes |
| `openai/gpt-5.5-20260423` | unblind | 3,997 | 3997 | 3,997 |  |  |  | 2026-05-07 09:15 → 2026-05-07 12:43 | $154.41 | yes |
| `openai/gpt-oss-120b` | blind | 3,997 | 999+1000+999+999 | 3,997 |  |  |  | 2026-05-15 15:17 → 2026-05-15 20:49 | $0.13 | yes |
| `openai/gpt-oss-120b` | unblind | 3,997 | 3997 | 3,997 |  |  |  | 2026-05-09 18:33 → 2026-05-09 19:55 | $0.16 | yes |
| `openai/gpt-oss-20b` | blind | 3,997 | 999+1000+999+999 | 3,997 |  |  |  | 2026-05-15 15:17 → 2026-05-15 21:15 | $0.72 | yes |
| `openai/gpt-oss-20b` | unblind | 3,997 | 3997 | 3,997 |  |  |  | 2026-05-09 12:22 → 2026-05-09 16:59 | $0.65 | yes |
