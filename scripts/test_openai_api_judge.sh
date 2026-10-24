#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${COME_PYTHON:-"$HOME/.conda/envs/graphmemix/bin/python"}
bundle="$project_root/data/unified/memeye"
predictions="$project_root/result_h100/come-results-20261024/come/experiments/memeye-h2hmem-20261022/memeye/runs/observations-v1/qwen3vl8b/memeye/reader/predictions.jsonl"
output="$project_root/runs/openai-api/memeye-gpt-5-mini.judgments.jsonl"
concurrency=${COME_JUDGE_CONCURRENCY:-4}

if [[ ! -x "$python_bin" ]]; then
  echo "Python environment not found: $python_bin" >&2
  exit 2
fi
if [[ ! -f "$project_root/.env" ]]; then
  echo "Project API key file not found: $project_root/.env" >&2
  exit 2
fi
if [[ ! -f "$bundle/manifest.json" || ! -f "$predictions" ]]; then
  echo "MemEye bundle or predictions are missing" >&2
  exit 2
fi

mkdir -p -- "$(dirname -- "$output")"
env -u OPENAI_API_KEY PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_bin" -m mm_memory_bench.cli judge \
  "$bundle" "$predictions" \
  --output "$output" \
  --model gpt-5-mini \
  --base-url https://api.openai.com/v1 \
  --concurrency "$concurrency"

echo "MemEye judgment output: $output"
