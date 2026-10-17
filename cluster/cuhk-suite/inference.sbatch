#!/bin/bash
#SBATCH --job-name=graphmemix-suite-inference
#SBATCH --partition=gpu2node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-23:00:00
#SBATCH --output=/share/home/jyh/graphmemix/cluster/cuhk-suite/logs/%x-%j.out
#SBATCH --error=/share/home/jyh/graphmemix/cluster/cuhk-suite/logs/%x-%j.err
source /share/home/jyh/graphmemix/cluster/cuhk-suite/common.sh
# The cluster's per-user process/thread soft limit is 256. Each service and
# tensor-parallel worker otherwise creates its own BLAS/OpenMP thread pools.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
"$PYTHON" - <<'PY'
import torch,os,socket
assert torch.cuda.device_count()==2
names=[torch.cuda.get_device_name(i) for i in range(2)]
assert all('4090' in n for n in names),names
print({'job':os.environ['SLURM_JOB_ID'],'host':socket.gethostname(),'GPUs':names},flush=True)
PY
PORT=$((18000 + SLURM_JOB_ID % 1000))
PROXY_PORT=$((PORT+1000))
BASE_URL=http://127.0.0.1:$PROXY_PORT/v1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
"$SERVING_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$DATA_ROOT/models/Qwen3-VL-8B-Instruct" \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --max-num-seqs 2 --max-num-batched-tokens 4096 \
  --limit-mm-per-prompt '{"image":80,"video":0}' --skip-mm-profiling --enforce-eager \
  > "$EXPERIMENT_ROOT/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
SERVER_PID=$!
"$SERVING_PYTHON" -u "$SUITE_ROOT/token_proxy.py" --model-path "$DATA_ROOT/models/Qwen3-VL-8B-Instruct" --upstream "http://127.0.0.1:$PORT" --port "$PROXY_PORT" > "$EXPERIMENT_ROOT/logs/token-proxy-$SLURM_JOB_ID.log" 2>&1 &
PROXY_PID=$!
"$PYTHON" -u "$SUITE_ROOT/watch_server.py" --parent-pid "$$" --server-pid "$SERVER_PID" --proxy-pid "$PROXY_PID" --log "$EXPERIMENT_ROOT/logs/vllm-$SLURM_JOB_ID.log" > "$EXPERIMENT_ROOT/logs/watch-server-$SLURM_JOB_ID.log" 2>&1 &
WATCH_PID=$!
trap 'kill -TERM "$WATCH_PID" "$PROXY_PID" "$SERVER_PID" 2>/dev/null || true; wait "$WATCH_PID" "$PROXY_PID" "$SERVER_PID" 2>/dev/null || true' EXIT
trap 'exit 1' TERM INT
"$PYTHON" -u "$SUITE_ROOT/check_server.py" --base-url "$BASE_URL" --pid "$SERVER_PID" --output "$EXPERIMENT_ROOT/validation/server-$SLURM_JOB_ID.json"
for DATASET in atm memeye mem_gallery; do
  "$PYTHON" -u scripts/run_graphmemix_release.py --config "$SUITE_ROOT/suite-config.json" benchmark \
    --dataset "$DATASET" --backbone qwen3vl8b --backend standalone \
    --data-root "$DATA_ROOT/data/unified" --artifact-root "$EXPERIMENT_ROOT/artifacts" \
    --run-root "$EXPERIMENT_ROOT/runs" --base-url "$BASE_URL" --concurrency 2 \
    --from-stage node --to-stage reader 2>&1 | tee "$EXPERIMENT_ROOT/logs/inference-$DATASET-$SLURM_JOB_ID.log"
  BUNDLE=$DATASET
  if [[ "$DATASET" == atm ]]; then BUNDLE=atm_bench; fi
  "$PYTHON" -u scripts/run_memix_answers_from_retrieval.py \
    --bundle "$DATA_ROOT/data/unified/$BUNDLE" \
    --retrieval "$EXPERIMENT_ROOT/runs/qwen3vl8b/$DATASET/atomic_retrieval.jsonl" \
    --checkpoint-root "$EXPERIMENT_ROOT/artifacts/checkpoints/$DATASET" --backend standalone \
    --output "$EXPERIMENT_ROOT/runs/dense/$DATASET/predictions.jsonl" \
    --base-url "$BASE_URL" --model Qwen/Qwen3-VL-8B-Instruct --concurrency 2 \
    --max-model-len 32768 --max-output-tokens 1000 --reasoning-effort none \
    --overflow-image-max-edge 2500 --reader-ocr-budget-chars 320 \
    --allow-variable-retrieval --resume --retry-errors --record-errors --progress-interval 100 \
    2>&1 | tee "$EXPERIMENT_ROOT/logs/dense-$DATASET-$SLURM_JOB_ID.log"
  "$PYTHON" -u "$SUITE_ROOT/evaluate_suite.py" --dataset "$DATASET"
done
"$PYTHON" -u "$SUITE_ROOT/evaluate_suite.py"
echo INFERENCE_SUITE_COMPLETE
