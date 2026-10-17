#!/bin/bash
#SBATCH --job-name=h2h-graph-dense-eval
#SBATCH --partition=gpu3node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-23:00:00
#SBATCH --output=/share/home/jyh/graphmemix/cluster/cuhk-suite/logs/%x-%j.out
#SBATCH --error=/share/home/jyh/graphmemix/cluster/cuhk-suite/logs/%x-%j.err
source /share/home/jyh/graphmemix/cluster/cuhk-suite/h2h_common.sh
source "$SUITE_ROOT/h2h_server.sh"
# Retain the frozen released pipeline and its strict resume validation.
run_logged "$EXPERIMENT_ROOT/logs/graph-$SLURM_JOB_ID.log" \
  "$PYTHON" -u scripts/run_graphmemix_release.py --config "$SUITE_ROOT/h2h-config.json" benchmark \
  --dataset h2hmem --backbone qwen3vl8b --backend standalone \
  --data-root "$DATA_ROOT/data/unified" --artifact-root "$EXPERIMENT_ROOT/artifacts" \
  --run-root "$EXPERIMENT_ROOT/runs" --base-url "$BASE_URL" --concurrency 2 \
  --from-stage node --to-stage reader
run_logged "$EXPERIMENT_ROOT/logs/dense-$SLURM_JOB_ID.log" \
  "$PYTHON" -u scripts/run_memix_answers_from_retrieval.py \
  --bundle "$DATA_ROOT/data/unified/h2hmem_main" \
  --retrieval "$EXPERIMENT_ROOT/runs/qwen3vl8b/h2hmem/atomic_retrieval.jsonl" \
  --checkpoint-root "$EXPERIMENT_ROOT/artifacts/checkpoints/h2hmem" --backend standalone \
  --output "$EXPERIMENT_ROOT/runs/dense/h2hmem/predictions.jsonl" \
  --base-url "$BASE_URL" --model Qwen/Qwen3-VL-8B-Instruct --concurrency 2 \
  --max-model-len 32768 --max-output-tokens 1000 --reasoning-effort none \
  --overflow-image-max-edge 2500 --reader-ocr-budget-chars 320 \
  --allow-variable-retrieval --resume --retry-errors --record-errors --progress-interval 100
"$PYTHON" -u "$SUITE_ROOT/evaluate_h2h.py"
run_logged "$EXPERIMENT_ROOT/logs/judge-$SLURM_JOB_ID.log" \
  "$PYTHON" -u "$SUITE_ROOT/qwen_judge_h2h.py" --base-url http://127.0.0.1:18310/v1 --concurrency 2
echo H2H_EXPERIMENT_COMPLETE
