#!/bin/bash
#SBATCH --job-name=smmbench-vl8b
#SBATCH --partition=gpu2node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm}"
: "${SMM_RUN_ROOT:?Frozen experiment directory required}"
MODE=${1:?observations or index}
WORKER=${SLURM_ARRAY_TASK_ID:-0}
DATA_ROOT=/share/workspace3/yhjia/graphmemix
PYTHON=/share/home/jyh/.conda/envs/graphmemix/bin/python
SERVING_PYTHON=/share/home/jyh/.conda/envs/atmbench/bin/python
PROJECT_ROOT=$SMM_RUN_ROOT/source
SUITE_ROOT=$PROJECT_ROOT/cluster/cuhk-suite
BUNDLE=$SMM_RUN_ROOT/data/smmbench
OBS=$SMM_RUN_ROOT/artifacts/observations/smmbench.jsonl
CKPT=$SMM_RUN_ROOT/artifacts/checkpoints/smmbench
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/scripts"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1
# Multiple replicas share the UID's per-node thread count. Match the existing
# H2HMem launcher, raising only the soft limit within the inherited hard limit.
smm_nproc_target=4096
smm_nproc_hard=$(ulimit -H -u)
if [[ $smm_nproc_hard != unlimited ]] && ((smm_nproc_hard < smm_nproc_target)); then
  smm_nproc_target=$smm_nproc_hard
fi
if [[ $(ulimit -S -u) != unlimited ]] && (($(ulimit -S -u) < smm_nproc_target)); then
  ulimit -S -u "$smm_nproc_target"
fi
echo "SMM_RUNTIME nproc=$(ulimit -S -u) rayon=$RAYON_NUM_THREADS"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$DATA_ROOT/cache/huggingface
export NLTK_DATA=$DATA_ROOT/nltk_data
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
cd "$PROJECT_ROOT"
mkdir -p "$SMM_RUN_ROOT/artifacts/observations"
exec 9>"$SMM_RUN_ROOT/$MODE-$WORKER.lock"
flock -n 9 || exit 2
echo "job=$SLURM_JOB_ID node=$(hostname) mode=$MODE visible_gpus=${CUDA_VISIBLE_DEVICES:?}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
source "$SUITE_ROOT/process_lifecycle.sh"
# Preserve failure status even when the caller uses an if/retry block.
run_logged() {
  local log=$1 status
  shift
  setsid bash -c 'set -o pipefail; log=$1; shift; "$@" 2>&1 | tee "$log"' _ "$log" "$@" &
  STAGE_PID=$!
  if wait "$STAGE_PID"; then status=0; else status=$?; fi
  STAGE_PID=
  return "$status"
}
if [[ $MODE == observations ]]; then
  if [[ ! -f $BUNDLE/manifest.json ]]; then
    run_logged "$SMM_RUN_ROOT/logs/prepare-$SLURM_JOB_ID.log" "$PYTHON" scripts/prepare_smmbench.py \
      --raw "$DATA_ROOT/data/raw/SMMBench" --output "$BUNDLE"
  fi
  OBS=$SMM_RUN_ROOT/artifacts/observations/smmbench-worker-$WORKER.jsonl
  PORT=$((32586 + WORKER))
  PROXY_PORT=$((33586 + WORKER))
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  setsid "$SERVING_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$DATA_ROOT/models/Qwen3-VL-8B-Instruct" \
    --served-model-name Qwen/Qwen3-VL-8B-Instruct --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 32768 \
    --gpu-memory-utilization 0.85 --max-num-seqs 4 --max-num-batched-tokens 4096 \
    --mm-processor-cache-gb 0 \
    --limit-mm-per-prompt '{"image":80,"video":0}' --skip-mm-profiling --enforce-eager \
    > "$SMM_RUN_ROOT/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
  SERVER_PID=$!
  CUDA_VISIBLE_DEVICES= setsid "$SERVING_PYTHON" -u "$SUITE_ROOT/token_proxy.py" \
    --model-path "$DATA_ROOT/models/Qwen3-VL-8B-Instruct" --upstream "http://127.0.0.1:$PORT" \
    --port "$PROXY_PORT" > "$SMM_RUN_ROOT/logs/proxy-$SLURM_JOB_ID.log" 2>&1 &
  PROXY_PID=$!
  setsid "$PYTHON" -u "$SUITE_ROOT/watch_server.py" --parent-pid "$$" \
    --server-pid "$SERVER_PID" --proxy-pid "$PROXY_PID" --log "$SMM_RUN_ROOT/logs/vllm-$SLURM_JOB_ID.log" \
    > "$SMM_RUN_ROOT/logs/watch-$SLURM_JOB_ID.log" 2>&1 &
  WATCH_PID=$!
  run_logged "$SMM_RUN_ROOT/logs/startup-$SLURM_JOB_ID.log" "$PYTHON" "$SUITE_ROOT/check_server.py" \
    --base-url "http://127.0.0.1:$PROXY_PORT/v1" --pid "$SERVER_PID" --startup-timeout-seconds 1200 \
    --output "$SMM_RUN_ROOT/validation/server-$SLURM_JOB_ID.json"
  # Retry only incomplete extraction; successful image observations are resumable.
  complete=0
  for attempt in 1 2 3; do
    if run_logged "$SMM_RUN_ROOT/logs/observations-$SLURM_JOB_ID-$attempt.log" \
      "$PYTHON" -m mm_memory_bench.come.runner --config "$SMM_RUN_ROOT/config.json" \
      --bundle "$BUNDLE" --artifact-root "$SMM_RUN_ROOT/artifacts/extraction" --output "$OBS" \
      --model Qwen/Qwen3-VL-8B-Instruct --base-url "http://127.0.0.1:$PROXY_PORT/v1" \
      --reasoning-effort none --workers 4 --worker-id "$WORKER" --context-workers 4; then complete=1; break; fi
  done
  [[ $complete == 1 ]]
  echo "SMMBENCH_OBSERVATIONS_COMPLETE job=$SLURM_JOB_ID"
elif [[ $MODE == index ]]; then
  run_logged "$SMM_RUN_ROOT/logs/merge-$SLURM_JOB_ID.log" \
    "$PYTHON" scripts/merge_smmbench_observations.py --root "$SMM_RUN_ROOT"
  test -s "${OBS%.jsonl}.manifest.json"
  run_logged "$SMM_RUN_ROOT/logs/checkpoints-$SLURM_JOB_ID.log" \
    "$PYTHON" scripts/smmbench_parallel_index.py --root "$SMM_RUN_ROOT" \
    --embedding-model "$DATA_ROOT/models/gme-Qwen2-VL-2B-Instruct"
  run_logged "$SMM_RUN_ROOT/logs/relations-$SLURM_JOB_ID.log" \
    "$PYTHON" scripts/build_graphmemix_relation_store.py --bundle "$BUNDLE" --observations "$OBS" \
    --checkpoint-root "$CKPT" --output "$SMM_RUN_ROOT/artifacts/relations/smmbench.jsonl" --max-semantic-k 16
  run_logged "$SMM_RUN_ROOT/logs/priors-$SLURM_JOB_ID.log" \
    "$PYTHON" scripts/run_memix_main_track_retrieval.py --bundle "$BUNDLE" --observations "$OBS" \
    --checkpoint-root "$CKPT" --embedding-model "$DATA_ROOT/models/gme-Qwen2-VL-2B-Instruct" \
    --backend standalone --output "$SMM_RUN_ROOT/artifacts/atomic_retrieval.jsonl" \
    --source-priors "$SMM_RUN_ROOT/artifacts/source_priors/smmbench.jsonl" --source-top-k 200 --processes 1
  "$PYTHON" scripts/validate_smmbench_build.py --root "$SMM_RUN_ROOT"
  echo "SMMBENCH_BUILD_COMPLETE job=$SLURM_JOB_ID"
else
  exit 2
fi
