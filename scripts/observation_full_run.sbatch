#!/bin/bash
#SBATCH --job-name=come-observations-full
#SBATCH --partition=gpu3node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00

set -eo pipefail
: "${COME_FULL_ROOT:?frozen run root required}"
: "${COME_DATASET:?atm or mem_gallery required}"
DATA_ROOT=${COME_MODEL_ROOT:-/share/workspace3/yhjia/graphmemix}
PYTHON=${COME_PYTHON:-/share/home/jyh/.conda/envs/graphmemix/bin/python}
SERVING_PYTHON=${COME_SERVING_PYTHON:-/share/home/jyh/.conda/envs/atmbench/bin/python}
PROJECT_ROOT=$COME_FULL_ROOT/source
SUITE_ROOT=$PROJECT_ROOT/cluster/cuhk-suite
RUN_ROOT=$COME_FULL_ROOT/$COME_DATASET
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/scripts:$SUITE_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$RUN_ROOT/cache/huggingface
export NLTK_DATA=$DATA_ROOT/nltk_data
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
cd "$PROJECT_ROOT"
source "$SUITE_ROOT/process_lifecycle.sh"
# Refuse concurrent writers to the same experiment; separate datasets have separate locks.
exec 9>"$RUN_ROOT/run.lock"
flock -n 9 || { echo 'Another writer holds this run'; exit 2; }
case "$COME_DATASET" in
  atm) BUNDLE=atm_bench; PORT=32110 ;;
  mem_gallery) BUNDLE=mem_gallery; PORT=32120 ;;
  *) exit 2 ;;
esac
PROXY_PORT=$((PORT + 1000))
BASE_URL=http://127.0.0.1:$PROXY_PORT/v1
SERVING_GPUS=${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation missing}
IFS=',' read -ra ASSIGNED_GPUS <<< "$SERVING_GPUS"
[[ ${#ASSIGNED_GPUS[@]} == 4 ]]
"$PYTHON" - <<'PY'
import hashlib, json, os, socket, torch
from pathlib import Path
root = Path(os.environ['COME_FULL_ROOT'])
for manifest, folder in [('source-manifest.json', 'source'), ('data-manifest.json', 'data')]:
    for relative, expected in json.loads((root / manifest).read_text()).items():
        assert hashlib.sha256((root / folder / relative).read_bytes()).hexdigest() == expected, relative
report = dict(job_id=os.environ['SLURM_JOB_ID'], host=socket.gethostname(), dataset=os.environ['COME_DATASET'],
              gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
              serving_tensor_parallel_size=4, extraction_workers=4, query_concurrency=4,
              embedding_shares_live_server=False)
(root / os.environ['COME_DATASET'] / 'validation' / ('allocation-' + os.environ['SLURM_JOB_ID'] + '.json')).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
PY

phase() {
  COME_PHASE=$1 "$PYTHON" - <<'PY'
import json, os, time
from pathlib import Path
from mm_memory_bench.come.store import atomic_json
atomic_json(Path(os.environ['COME_FULL_ROOT']) / os.environ['COME_DATASET'] / 'phase.json',
            dict(phase=os.environ['COME_PHASE'], job_id=os.environ['SLURM_JOB_ID'], time=time.time()))
PY
}

start_services() {
  local stage=$1
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  CUDA_VISIBLE_DEVICES=$SERVING_GPUS setsid "$SERVING_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$DATA_ROOT/models/Qwen3-VL-8B-Instruct" \
    --served-model-name Qwen/Qwen3-VL-8B-Instruct --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 32768 \
    --gpu-memory-utilization 0.80 --max-num-seqs 4 --max-num-batched-tokens 4096 \
    --limit-mm-per-prompt '{"image":80,"video":0}' --skip-mm-profiling --enforce-eager \
    > "$RUN_ROOT/logs/vllm-$stage-$SLURM_JOB_ID.log" 2>&1 &
  SERVER_PID=$!
  CUDA_VISIBLE_DEVICES= setsid "$SERVING_PYTHON" -u "$SUITE_ROOT/token_proxy.py" \
    --model-path "$DATA_ROOT/models/Qwen3-VL-8B-Instruct" --upstream "http://127.0.0.1:$PORT" \
    --port "$PROXY_PORT" > "$RUN_ROOT/logs/proxy-$stage-$SLURM_JOB_ID.log" 2>&1 &
  PROXY_PID=$!
  setsid "$PYTHON" -u "$SUITE_ROOT/watch_server.py" --parent-pid "$$" \
    --server-pid "$SERVER_PID" --proxy-pid "$PROXY_PID" --log "$RUN_ROOT/logs/vllm-$stage-$SLURM_JOB_ID.log" \
    > "$RUN_ROOT/logs/watch-$stage-$SLURM_JOB_ID.log" 2>&1 &
  WATCH_PID=$!
  run_logged "$RUN_ROOT/logs/startup-$stage-$SLURM_JOB_ID.log" "$PYTHON" -u "$SUITE_ROOT/check_server.py" \
    --base-url "$BASE_URL" --pid "$SERVER_PID" --startup-timeout-seconds 1200 \
    --output "$RUN_ROOT/validation/server-$stage-$SLURM_JOB_ID.json"
}

stop_services() {
  if [[ -n "$WATCH_PID" ]]; then
    kill -TERM -- "-$WATCH_PID" 2>/dev/null || true
    wait "$WATCH_PID" 2>/dev/null || true
    WATCH_PID=
  fi
  local groups=() pid attempt alive
  for pid in "$PROXY_PID" "$SERVER_PID"; do [[ -z "$pid" ]] || groups+=("-$pid"); done
  if ((${#groups[@]})); then
    kill -TERM -- "${groups[@]}" 2>/dev/null || true
    for ((attempt=0; attempt<30; attempt++)); do
      alive=0
      for pid in "${groups[@]}"; do kill -0 -- "$pid" 2>/dev/null && alive=1; done
      ((alive)) || break
      sleep 1
    done
    kill -KILL -- "${groups[@]}" 2>/dev/null || true
    wait "$SERVER_PID" "$PROXY_PID" 2>/dev/null || true
  fi
  SERVER_PID= PROXY_PID=
}

ARTIFACTS=$RUN_ROOT/artifacts/observations-v1
phase observations
if [[ ! -f "$ARTIFACTS/observations/$COME_DATASET.manifest.json" ]]; then start_services extraction; fi
run_logged "$RUN_ROOT/logs/observations-$SLURM_JOB_ID.log" \
  "$PYTHON" -u -m mm_memory_bench.come.runner --config "$COME_FULL_ROOT/config.json" \
  --bundle "$COME_FULL_ROOT/data/$BUNDLE" --artifact-root "$ARTIFACTS/extraction/$COME_DATASET" \
  --output "$ARTIFACTS/observations/$COME_DATASET.jsonl" --model Qwen/Qwen3-VL-8B-Instruct \
  --base-url "$BASE_URL" --reasoning-effort none --workers 4
stop_services

common=("$PYTHON" -u scripts/run_graphmemix_release.py --config "$COME_FULL_ROOT/config.json" benchmark
  --dataset "$COME_DATASET" --backbone qwen3vl8b --backend standalone
  --data-root "$COME_FULL_ROOT/data" --artifact-root "$RUN_ROOT/artifacts"
  --run-root "$RUN_ROOT/runs" --base-url "$BASE_URL" --concurrency 4 --local-processes 1)
phase checkpoint_priors_relations
export CUDA_VISIBLE_DEVICES=${ASSIGNED_GPUS[0]}
run_logged "$RUN_ROOT/logs/index-$SLURM_JOB_ID.log" "${common[@]}" --from-stage checkpoint --to-stage relations
"$PYTHON" scripts/evaluate_observation_full_run.py --root "$COME_FULL_ROOT" --dataset "$COME_DATASET" --seal-only

phase node_through_reader
start_services query
run_logged "$RUN_ROOT/logs/query-$SLURM_JOB_ID.log" "${common[@]}" --from-stage node --to-stage reader
phase metrics_and_local_judge
run_logged "$RUN_ROOT/logs/evaluation-$SLURM_JOB_ID.log" "$PYTHON" -u scripts/evaluate_observation_full_run.py \
  --root "$COME_FULL_ROOT" --dataset "$COME_DATASET" --base-url "$BASE_URL" --concurrency 4
phase complete
echo "COME_OBSERVATIONS_FULL_COMPLETE dataset=$COME_DATASET job=$SLURM_JOB_ID"
