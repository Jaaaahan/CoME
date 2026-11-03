<p align="center">
  <img src="assets/graphmemix_title.png" alt="GraphMemix" width="420">
</p>

<p align="center">
  <em>Query-time evidence organization for long-term multimodal memory.</em>
</p>

GraphMemix builds a bounded, query-relevant graph over multimodal memories. It
combines multi-view retrieval, direct-evidence verification,
anchor-conditioned relation verification, and evidence-forest optimization to
produce a compact context for the final reader.

<p align="center" width="100%">
  <img src="assets/graphmemix_method.png" alt="GraphMemix method overview" style="width: 96%; min-width: 280px; display: block; margin: auto;">
</p>

The repository provides the GraphMemix implementation, a unified benchmark
harness, reproducible evaluation contracts, baseline adapters, and an
ATM-Bench-Hard evidence-graph extension.

The default runner uses **GraphMemix with offline CoME observations**. CoME
extracts reusable details from source images; GraphMemix embeds each observation
as a source-image crop plus its description. The original offline mutual-kNN
graph now contains both memory and observation units. Queries use the original
one-hop expansion over stored edges, starting from retrieved units. Selected units
remain independent through verification, forest selection, pruning, and reading;
parent memory IDs are retained for provenance and source-level evaluation.
See [the CoME guide](docs/come.md) for interfaces, budgets, and evaluation.
The results below describe the original GraphMemix release; its configuration
is preserved in `configs/release/graphmemix_legacy.json`.

## 📰 Updates

- **2026-08-27** — Paper released on [arXiv:2608.26983](https://arxiv.org/abs/2608.26983).

## 🎯 Overview

- **Query-aware candidate graph.** Multi-view retrieval finds seed memories;
  schema and semantic relations expand a bounded candidate set.
- **Separated evidence verification.** A node verifier estimates direct
  support, while the Evidence-Chain Verifier (ECV) estimates the incremental
  value of a candidate relative to an anchor.
- **Evidence-forest optimization.** GraphMemix jointly selects memories and
  reliable relations under a maximum reader budget.
- **Native multimodal context.** The reader receives selected image, video,
  email, and text evidence rather than only generated summaries.

```text
multimodal archive + query
          │
          ▼
multi-view retrieval ──► bounded candidate graph
                                │
                      ┌─────────┴─────────┐
                      ▼                   ▼
               node verifier             ECV
                      └─────────┬─────────┘
                                ▼
                     evidence-forest optimizer
                                ▼
                       compact reader context
```

## 📊 Main Results

All values are GPT-5-mini Judge Accuracy (%). Each backbone configuration uses
the same model for node verification, ECV, and final answering.

### Qwen3-VL-8B-Instruct

| Method | ATM-Bench | Mem-Gallery | MemEye | H2HMem | Average |
| :-- | --: | --: | --: | --: | --: |
| A-MEM | 41.86 | 52.89 | 37.09 | 39.94 | 42.94 |
| UniversalRAG | 43.10 | 63.76 | 47.98 | 44.36 | 49.80 |
| MemGuide | 48.47 | 48.10 | 41.40 | 39.94 | 44.48 |
| LightMem | 22.70 | 38.28 | 36.23 | 31.04 | 32.06 |
| VimRAG | 33.81 | 48.33 | 28.25 | 17.89 | 32.07 |
| **GraphMemix** | **55.27** | **76.33** | **53.64** | **60.96** | **61.55** |

### Gemma 4 12B Unified

| Method | ATM-Bench | Mem-Gallery | MemEye | H2HMem | Average |
| :-- | --: | --: | --: | --: | --: |
| A-MEM | 42.43 | 50.03 | 24.42 | 44.20 | 40.27 |
| UniversalRAG | 49.04 | 64.82 | 58.17 | 48.34 | 55.09 |
| MemGuide | 54.98 | 48.33 | 30.40 | 47.98 | 45.43 |
| LightMem | 24.33 | 33.72 | 20.70 | 32.80 | 27.89 |
| VimRAG | 22.51 | 46.99 | 31.81 | 39.00 | 35.08 |
| **GraphMemix** | **58.05** | **81.47** | **66.68** | **63.47** | **67.42** |

The frozen result manifest is
[`release/benchmark_results.json`](release/benchmark_results.json).

| Dataset key | Benchmark track | Questions |
| :-- | :-- | --: |
| `atm` | ATM-Bench, default + hard | 1,044 |
| `mem_gallery` | Mem-Gallery | 1,711 |
| `memeye` | MemEye | 1,855 |
| `h2hmem` | H2HMem evidence-supported track | 1,982 |

## 🛠️ Installation

```bash
git clone https://github.com/ligeng0197/graphmemix.git
cd graphmemix
python -m pip install -e '.[methods]'
```

## 📦 Data and Artifacts

The full evaluation bundles (unified snapshots plus raw media) for the four
benchmarks are published on Hugging Face:
[graphmemix-benchmarks](https://huggingface.co/datasets/oking0197/graphmemix-benchmarks).
Download them into the repository's `data/` directory:

```bash
hf download oking0197/graphmemix-benchmarks --repo-type dataset --local-dir data
```

Large reusable artifacts use the following default layout:

```text
artifacts/graphmemix/
├── captions/h2hmem.jsonl
├── checkpoints/{atm,mem_gallery,memeye,h2hmem}/
├── relation_store/{atm,mem_gallery,memeye,h2hmem}.jsonl
├── source_priors/{atm,mem_gallery,memeye,h2hmem}.jsonl
└── vendor/memix-core/
```

Raw benchmark assets retain their upstream licenses. Consult each benchmark's
license before downloading or redistributing its data.

## 🚀 Run the Benchmark Suite

Validate the current configuration:

```bash
python scripts/run_graphmemix_release.py validate-config
```

Inspect an end-to-end run without invoking a model:

```bash
python scripts/run_graphmemix_release.py benchmark \
  --dataset atm \
  --backbone qwen3vl8b \
  --base-url http://127.0.0.1:8000/v1 \
  --judge-base-url https://YOUR-JUDGE-ENDPOINT/v1 \
  --dry-run
```

Run one dataset/backbone cell end-to-end (all stages from `checkpoint`
through `judge`):

```bash
export JUDGE_API_KEY=your_api_key

python scripts/run_graphmemix_release.py benchmark \
  --dataset atm \
  --backbone qwen3vl8b \
  --base-url http://127.0.0.1:8000/v1 \
  --judge-base-url https://YOUR-JUDGE-ENDPOINT/v1 \
  --judge-api-key-env JUDGE_API_KEY
```

Run or resume selected stages:

```bash
export JUDGE_API_KEY=your_api_key

python scripts/run_graphmemix_release.py benchmark \
  --dataset memeye \
  --backbone gemma4_12b \
  --base-url http://127.0.0.1:8000/v1 \
  --judge-base-url https://YOUR-JUDGE-ENDPOINT/v1 \
  --judge-api-key-env JUDGE_API_KEY \
  --from-stage reader \
  --to-stage judge
```

For a no-ECV ablation, add `--disable-ecv`. This skips ECV inference and uses
fixed offline explicit edges for both selection and pruning, retaining node
verification and the candidate pool. Outputs are isolated under `no-ecv/` within
the dataset run directory. To reuse existing node scores, start at `select`:

```bash
python scripts/run_graphmemix_release.py benchmark \
  --dataset memeye --backbone qwen3vl8b --disable-ecv \
  --from-stage select --to-stage reader --base-url http://127.0.0.1:8000/v1
```

### Adaptive unit reading

`run_graphmemix_adaptive_reader.py` implements query-guided reading directly from
frozen observation checkpoints, source priors, and relation stores. It takes 24
retrieved seed units, expands one hop to at most 48 candidates, and recomputes
cosine similarity against every candidate's checkpoint vector. Original-question
text and generated queries use the same GME text encoding. Query images remain
visible to the reader; the existing seed retrieval retains its multimodal policy.
Memory vectors use normalized means of their indexed views; observation vectors
use their stored crop-description representation.

The reader consumes **two new units per reading round, at most 10 rounds and 20 units per question**,
retaining all visited multimodal evidence. The candidate pool still contains up to
48 units. Each query caches its top 20 currently unread candidates from that pool
(or fewer if fewer remain), plus scores and a record of consumed units. The
original question is the first active query. Global deduplication prevents any
unit from being read twice, including when switching between cached queries.

Exact normalized query hits skip encoding. Cosine similarity >= `--query-threshold`
(default 0.95) reuses a fixed representative's ranking and scores; new paraphrases
are compared to representatives, avoiding transitive similarity drift. Repeating
the active query continues through its unread list, including the fifth and later
units. Query repetition alone does not trigger stopping.

A proposed different query is accepted only when **the similarity between the
proposed query and its next two unread units, averaged, strictly exceeds the mean
similarity between the last accepted query and its own next two unread units**.
Both lists skip globally visited units. If fewer than two remain in a list, use
the mean over its available units without zero padding. There is no extra improvement margin. Equal or lower
scores reject the proposed direction and keep reading from the previous active
query, rather than reverting to the original question. Rejected queries remain
cached to avoid recomputing their scores; they do not become active. Later cache
hits are compared against the then-active query. Acceptance is a query-level gate,
not a per-unit fallback, and is a retrieval heuristic rather than proof of accuracy.

Retrieval decisions and final answering are separate stages. ENM decisions report
`supported` requirements with references to **already-read unit aliases**, `missing`
requirements, and explicit `coverage` (complete, reason, evidence). `read_units`
maps aliases to canonical Evidence numbers, which can differ after query switching.
A `sufficient_evidence` stop requires nonempty supported evidence, no missing
requirements, and complete coverage backed by read references. For list/all-photo
questions, the decision prompt requires an explanation of completeness; one match
alone does not establish that all requested photos have been found. Validation
checks references and consistency, not whether the model's visual claims are true.
Invalid decisions receive a repair request within the existing retry/call limits.
Accepted assessments and raw decisions, plus invalid responses and validation
errors, are retained in the trace's controller fields.

Stops distinguish `sufficient_evidence`, `cannot_find_more`, and `budget_exhausted`.
A model claiming it cannot find more must explain why; unresolved requirements
remain recorded. Candidate/query exhaustion is also `cannot_find_more`. Round,
unit, API-call and decision-retry limits are `budget_exhausted`, with the specific
cause in `stop_detail`. Neither of these reasons claims evidence sufficiency.

The final call reuses **the canonical CoME answer request unchanged**, including
its original question/instruction, all visited images/descriptions, and tool
parameters. No extra answer system prompt or JSON schema is imposed. Retrieval
assessments, relations, intermediate queries and tentative answers are excluded
from this final request. The model's plain-text answer is retained.

The default limits are 10 reading rounds and **10 actual API calls including
format retries**. One call is always reserved for the final answer. Without retries,
a full run uses nine retrieval decisions followed by one final call after reading
units 19 and 20. Early stopping also goes through the canonical final-answer stage.
Retries can shorten retrieval. Invalid retrieval decisions fall through to final
answering when retries are exhausted. Empty/non-text final responses retry the same
canonical request within the remaining budget. If no valid answer is produced,
an explicitly unanswered row is recorded with raw response/error details; the next
question proceeds. The old three-batch and stagnation blocking rules remain removed.

No node verification, ECV, forest selection, or pruning is performed. Query matching,
query table size, switching and stopping settings are part of the resume contract;
use a new output path when changing them or switching from an earlier protocol.

```bash
PYTHONPATH=src:scripts python scripts/run_graphmemix_adaptive_reader.py \
  --bundle data/unified/memeye \
  --observations artifacts/graphmemix/observations-v1/observations/memeye.jsonl \
  --checkpoint-root artifacts/graphmemix/observations-v1/checkpoints/memeye \
  --source-priors artifacts/graphmemix/observations-v1/source_priors/memeye.jsonl \
  --relation-store artifacts/graphmemix/observations-v1/relation_store/memeye.jsonl \
  --embedding-model /path/to/gme-Qwen2-VL-2B-Instruct \
  --backend standalone --initial-limit 48 --batch-size 2 \
  --query-table-size 20 --max-rounds 10 --max-llm-calls 10 \
  --max-model-len 131072 --image-max-edge 1280 \
  --base-url http://127.0.0.1:8000/v1 \
  --output runs/adaptive/memeye/predictions.jsonl --resume
```

Use the backend and embedding model that produced the checkpoint. The embedding
model runs in this process; the reader runs at `--base-url`. The entry point
processes questions sequentially. Use `--question-ids` for a smoke-test subset.
Output rows contain final answers, visited unit IDs, original scores, each query
acceptance/rejection with both lists of comparison units, their scores and means, LLM call counts, and elapsed time. Resume checks
bind results to data, checkpoint, embedding, rendering, code, and run parameters.
Relations disclose parent memory identity, candidate-local semantic neighbors,
and same-session IDs without exposing unread neighbors' contents. Full visited
multimodal evidence is retained each round. Context overflow fails visibly;
it does not silently discard earlier evidence. Invalid decisions are retried up
to `--decision-retries` (default 3), including failure to stop at exhaustion.

The stages are `observations` (optional preprocessing), `checkpoint`, `priors`,
`relations`, `node`, `ecv`, `select`, `prune`, `reader`, and `judge`. Queries read
the frozen observations without extracting new ones or changing the index.
Cached outputs record their input contracts so incompatible rows cannot be
silently reused. See [the CoME guide](docs/come.md) for the current workflow and
[the release guide](docs/graphmemix_release.md) for historical reproduction.

### MemEye and MemoryGallery ENM parameter sweeps

Both canonical datasets are supported, including every MemEye question variant.
For the logged-in SwanLab account on the cluster, replace the generation command
below with `scripts/run_enm_swanlab.py generate` using the same arguments. It
clears all six HTTP/HTTPS/ALL proxy variables and records parameters and question
generation progress in SwanLab. The `sync` subcommand accepts completed scalar
metrics as JSON on stdin and resumes the same experiment after API judging.
The judge only publishes the accuracy after all questions have been scored.

For cluster nodes without W&B/OpenAI access, use the local search driver
`scripts/run_enm_local_search.py`. It samples 24 distinct configurations with a
fixed seed plus the default baseline, without importing W&B. The search space
is in `config/enm_mem_gallery_local_search.json`.

```bash
python scripts/run_enm_local_search.py plan \
  --settings /absolute/path/to/settings.json --output search-plan.json
# Run one index per Slurm array element, with --gres=gpu:2 --array=0-24%1:
python scripts/run_enm_local_search.py generate \
  --plan search-plan.json --index 0 --base-url http://127.0.0.1:24481/v1
```

Generation writes a compact `judge-input.jsonl` and a `generation-complete.json`
marker only after all answers are present. On a machine with OpenAI access, the
`watch` command fetches these files through the storage SSH host, verifies hashes
and question coverage, evaluates with GPT-5.4 mini and updates `leaderboard.json`:

```bash
python scripts/run_enm_local_search.py watch \
  --plan search-plan.json --bundle /local/copy/of/the/same/bundle \
  --remote-root /remote/experiment/root --output runs/enm-local-scores
```

The scoring bundle needs `manifest.json` and the exact `questions.jsonl` from the
generation bundle. Successful judgments are reused after interruption. This
workflow uses two GPUs per dataset at a time; API judging uses no local GPU.
The remaining instructions describe the optional W&B workflow.

`scripts/run_enm_sweep.py` uses **all 1,711 MemoryGallery questions** in every
trial, with no development/test split or sampling. The default judge is
**OpenAI / gpt-5.4-mini** through Chat Completions and the
existing binary LLM-judge rubric. The W&B objective is `tune/judge_accuracy`
(maximize), reported as **tuning-set accuracy**. Reader failures count as
incorrect; judge/API failures are retried and must be resolved before the
objective is logged. This is a GPT-5.4 mini judge metric, not the benchmark's official
judge score.

No split preparation is required. The runner takes every question from the
bundle, verifies complete prediction coverage and records `evaluation_scope:
full_dataset` in the run metadata. Old `dev_ids.txt` / `test_ids.txt` files are
unused; remove `split_dir` from older settings files. The new run protocol
prevents reusing the earlier 200-question experiment outputs.

Copy `config/enm_mem_gallery_settings.example.json` to a local settings file.
Set paths to the frozen observations, checkpoints, source priors, relation store
and embedding weights, plus the reader endpoint/context capacity. **Relative paths
resolve against the settings file's directory.** Configure W&B's project/entity
there, and provide credentials through the environment:

```bash
export ENM_SWEEP_SETTINGS=/absolute/path/to/settings.json
# OPENAI_BASE_URL defaults to https://api.openai.com/v1.
# Set OPENAI_API_KEY and WANDB_API_KEY through your usual secret configuration.

python scripts/run_enm_sweep.py trial --settings "$ENM_SWEEP_SETTINGS" --dry-run
```

Preflight checks all questions, stored semantic-neighbor capacity and source-hit
coverage. It makes no API calls and requires no API key or W&B installation.
The default judge model ID is `gpt-5.4-mini`, which supports Chat Completions
and structured outputs ([OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-5.4-mini)).
The existing judge client omits `temperature` for GPT-5 models. At the official
OpenAI endpoint, it also reads `OPENAI_API_KEY` from the repository's ignored
`.env` file when the environment variable is unset; that file must have owner-only
permissions (`chmod 600 .env`). W&B also accepts credentials saved by `wandb login`
in the user's `.netrc`. Credentials never go in the JSON settings file.

Install the optional W&B dependency in the reader's environment, run the default
baseline, then launch the supplied 24-run random sweep from the repository root:

```bash
python -m pip install -e '.[sweeps]'
python scripts/run_enm_sweep.py trial --settings "$ENM_SWEEP_SETTINGS" --run-id baseline
wandb sweep --project enm-mem-gallery config/enm_mem_gallery_sweep.yaml
wandb agent ENTITY/PROJECT/SWEEP_ID
```

The sweep varies `query_threshold`, `source_top_l`, `candidate_limit`, `semantic_k`
and `query_table_size`. The default `semantic_k` choices are 2/4/8, compatible
with the available MemoryGallery graph stored at k=8. `initial_limit` follows
`candidate_limit`. Reading stays
at two units per round, at most ten rounds, twenty units and ten LLM calls;
decision attempts stay at three. Graph neighbors beyond the stored
`max_semantic_k` and seed counts exceeding stored unit hits are rejected before
generation. The reader/GME environment is the same as for the standalone
adaptive-reader command; the `sweeps` extra only installs W&B.

Each trial has its own `contract.json`, `reader.log`, `predictions.jsonl`,
`judgments.jsonl` and `metrics.json` under `output_root/RUN_ID`. Logs record the
run ID. Resume an interrupted trial with the same settings, sampled parameters
and `--run-id RUN_ID`; successful judgments are reused. Changed inputs, code,
judge endpoint or parameters require a new ID. API keys are not written into
the contract or W&B config. Only aggregate metrics and experiment metadata are
sent to W&B. Use `--wandb-mode offline` to retain W&B logs locally or `disabled`
to run without W&B. No Hyperband early termination is configured, and the
objective is logged exactly once after complete evaluation of each trial.

## 🧩 ATM-Bench-Hard Evidence-Graph Extension

The repository also contains a specialized 31-question ATM-Bench-Hard
pipeline that reaches **57.95 QS for $3.61** — within a point of the official
leaderboard's highest-scoring configurations at roughly one-third of their
cost. Pi drives DeepSeek V4 Flash to iteratively mine the memory archive and
assemble a query-conditioned evidence graph, then a deterministic GraphMemix
forest optimizer commits to the evidence under a strict reader budget, weighing
graph confidence, structural coverage, semantic source priors, edge incidence,
and connectivity.

<p align="center" width="100%">
  <img src="assets/atm_hard_price_performance.png" alt="ATM-Bench-Hard price-performance comparison with GraphMemix" style="width: 94%; min-width: 280px; display: block; margin: auto;">
</p>

The highlighted point is the released selector: **$3.61 / 57.95 QS** (list
recall 74.70, number 66.67, open end 38.46). The cost uses the same frozen
list-price snapshot as the chart and includes DeepSeek graph construction plus
GPT-5.6 Sol reading; GPT-5-mini evaluation and local graph compute are
excluded.

The frozen configuration and graph-construction prompt are:

- [`configs/release/atm_hard_graph_proof.json`](configs/release/atm_hard_graph_proof.json)
- [`configs/release/prompts/atm_hard_evidence_graph_v3_compact.txt`](configs/release/prompts/atm_hard_evidence_graph_v3_compact.txt)

Inspect the pipeline without API calls:

```bash
python scripts/run_atm_hard_graph_proof_release.py \
  --atm-agent-root vendor/ATM-Bench \
  --checkpoint-root artifacts/graphmemix/checkpoints/atm \
  --source-priors artifacts/graphmemix/source_priors/atm.jsonl \
  --memix-repo artifacts/graphmemix/vendor/memix-core \
  --dry-run
```

## 📁 Repository Layout

```text
graph-memix/
├── assets/                   # README method and benchmark figures
├── config/                   # general benchmark and evaluation defaults
├── configs/release/          # frozen GraphMemix and ATM-Hard contracts
├── docs/                     # data, evaluation, and execution documentation
├── release/                  # result and repository-integrity manifests
├── schema/                   # unified-memory JSON schema
├── scripts/                  # pipeline stages, runners, and audits
├── src/mm_memory_bench/      # benchmark harness, converters, and methods
├── tests/                    # unit and integration tests
├── CITATION.cff
├── pyproject.toml
└── README.md
```

## 📄 Citation

If you use GraphMemix, please cite:

```bibtex
@misc{li2026graphmemix,
  title         = {GraphMemix: Query-Aware Evidence Forests for Long-Term Multimodal Agent Memory},
  author        = {Li, Geng and Wang, Yuhao and Li, Dong and Hao, Jianye and Peng, Yuxin},
  year          = {2026},
  eprint        = {2608.26983},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2608.26983}
}
```

## 🙏 Acknowledgements

GraphMemix is evaluated on ATM-Bench, Mem-Gallery, MemEye, and H2HMem and uses
Qwen3-VL and Gemma 4 as reasoning backbones. We thank the benchmark and model
authors for making reproducible multimodal-memory research possible.
