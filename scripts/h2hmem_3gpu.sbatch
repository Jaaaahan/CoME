#!/bin/bash
#SBATCH --job-name=come-h2hmem-3gpu
#SBATCH --partition=gpu3node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -eo pipefail
: "${COME_FULL_ROOT:?run prepare_observation_full_run.py --datasets h2hmem --gpus 3 first}"
DATA_ROOT=${COME_MODEL_ROOT:-/share/workspace3/yhjia/graphmemix}
PYTHON=${COME_PYTHON:-/share/home/jyh/.conda/envs/graphmemix/bin/python}
SERVING_PYTHON=${COME_SERVING_PYTHON:-/share/home/jyh/.conda/envs/atmbench/bin/python}
PROJECT_ROOT=$COME_FULL_ROOT/source
SUITE_ROOT=$PROJECT_ROOT/cluster/cuhk-suite
RUN_ROOT=$COME_FULL_ROOT/h2hmem
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/scripts:$SUITE_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$RUN_ROOT/cache/huggingface
export NLTK_DATA=$DATA_ROOT/nltk_data
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
# Threads count toward NPROC; the inherited soft limit can be only 256.
come_nproc_target=4096
come_nproc_hard=$(ulimit -H -u)
come_nproc_soft=$(ulimit -S -u)
if [[ "$come_nproc_hard" != unlimited ]] && ((come_nproc_hard < come_nproc_target)); then
  come_nproc_target=$come_nproc_hard
fi
if [[ "$come_nproc_soft" != unlimited ]] && ((come_nproc_soft < come_nproc_target)); then
  ulimit -S -u "$come_nproc_target"
fi
printf 'COME_RUNTIME nproc=%s compile_threads=%s\n' "$(ulimit -S -u)" "$TORCHINDUCTOR_COMPILE_THREADS"
cd "$PROJECT_ROOT"
source "$SUITE_ROOT/process_lifecycle.sh"
exec 9>"$RUN_ROOT/run.lock"
flock -n 9 || { echo 'Another writer holds this run'; exit 2; }
IFS=',' read -ra ASSIGNED_GPUS <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation missing}"
[[ ${#ASSIGNED_GPUS[@]} == 3 ]]
SERVING_GPUS=${ASSIGNED_GPUS[0]},${ASSIGNED_GPUS[1]}
EMBEDDING_GPU=${ASSIGNED_GPUS[2]}
PORT=$((28000 + SLURM_JOB_ID % 1000))
PROXY_PORT=$((PORT + 1000))
BASE_URL=http://127.0.0.1:$PROXY_PORT/v1
"$PYTHON" - <<'PY'
import json, os, socket, torch
from pathlib import Path
from mm_memory_bench.come.store import file_digest
root = Path(os.environ['COME_FULL_ROOT'])
for manifest, folder in [('source-manifest.json', 'source'), ('data-manifest.json', 'data')]:
    for relative, expected in json.loads((root / manifest).read_text()).items():
        assert file_digest(root / folder / relative) == expected, relative
inputs = json.loads((root / 'inputs.json').read_text())
assert inputs['gpus_per_job'] == 3 and inputs['extraction_workers'] == 2
assert file_digest(root / 'h2hmem/artifacts/captions/h2hmem.jsonl') == inputs['captions']['sha256']
assert torch.cuda.device_count() == 3
report = dict(job_id=os.environ['SLURM_JOB_ID'], host=socket.gethostname(),
              gpus=[torch.cuda.get_device_name(i) for i in range(3)],
              cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
              serving_tensor_parallel_size=2, embedding_gpu_index=2,
              extraction_workers=2, query_concurrency=2)
(root / 'h2hmem/validation' / ('allocation-' + os.environ['SLURM_JOB_ID'] + '.json')).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
PY

phase() {
  COME_PHASE=$1 "$PYTHON" - <<'PY'
import os, time
from pathlib import Path
from mm_memory_bench.come.store import atomic_json
atomic_json(Path(os.environ['COME_FULL_ROOT']) / 'h2hmem/phase.json',
            dict(phase=os.environ['COME_PHASE'], job_id=os.environ['SLURM_JOB_ID'], time=time.time()))
PY
}

phase server_startup
export VLLM_WORKER_MULTIPROC_METHOD=spawn
CUDA_VISIBLE_DEVICES=$SERVING_GPUS setsid "$SERVING_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$DATA_ROOT/models/Qwen3-VL-8B-Instruct" \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --max-num-seqs 2 --max-num-batched-tokens 4096 \
  --limit-mm-per-prompt '{"image":80,"video":0}' --skip-mm-profiling --enforce-eager \
  > "$RUN_ROOT/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
SERVER_PID=$!
CUDA_VISIBLE_DEVICES= setsid "$SERVING_PYTHON" -u "$SUITE_ROOT/token_proxy.py" \
  --model-path "$DATA_ROOT/models/Qwen3-VL-8B-Instruct" --upstream "http://127.0.0.1:$PORT" \
  --port "$PROXY_PORT" > "$RUN_ROOT/logs/proxy-$SLURM_JOB_ID.log" 2>&1 &
PROXY_PID=$!
setsid "$PYTHON" -u "$SUITE_ROOT/watch_server.py" --parent-pid "$$" \
  --server-pid "$SERVER_PID" --proxy-pid "$PROXY_PID" --log "$RUN_ROOT/logs/vllm-$SLURM_JOB_ID.log" \
  > "$RUN_ROOT/logs/watch-$SLURM_JOB_ID.log" 2>&1 &
WATCH_PID=$!
run_logged "$RUN_ROOT/logs/startup-$SLURM_JOB_ID.log" "$PYTHON" -u "$SUITE_ROOT/check_server.py" \
  --base-url "$BASE_URL" --pid "$SERVER_PID" --startup-timeout-seconds 1200 \
  --output "$RUN_ROOT/validation/server-$SLURM_JOB_ID.json"

export CUDA_VISIBLE_DEVICES=$EMBEDDING_GPU
ARTIFACTS=$RUN_ROOT/artifacts/observations-v1
phase observations
run_logged "$RUN_ROOT/logs/observations-$SLURM_JOB_ID.log" \
  "$PYTHON" -u -m mm_memory_bench.come.runner --config "$COME_FULL_ROOT/config.json" \
  --bundle "$COME_FULL_ROOT/data/h2hmem" --artifact-root "$ARTIFACTS/extraction/h2hmem" \
  --output "$ARTIFACTS/observations/h2hmem.jsonl" --model Qwen/Qwen3-VL-8B-Instruct \
  --base-url "$BASE_URL" --reasoning-effort none --workers 2

common=("$PYTHON" -u scripts/run_graphmemix_release.py --config "$COME_FULL_ROOT/config.json" benchmark
  --dataset h2hmem --backbone qwen3vl8b --backend standalone
  --data-root "$COME_FULL_ROOT/data" --artifact-root "$RUN_ROOT/artifacts"
  --run-root "$RUN_ROOT/runs" --base-url "$BASE_URL" --concurrency 2 --local-processes 1)
phase checkpoint_priors_relations
run_logged "$RUN_ROOT/logs/index-$SLURM_JOB_ID.log" "${common[@]}" --from-stage checkpoint --to-stage relations
"$PYTHON" scripts/evaluate_observation_full_run.py --root "$COME_FULL_ROOT" --dataset h2hmem --seal-only
phase node_through_reader
run_logged "$RUN_ROOT/logs/query-$SLURM_JOB_ID.log" "${common[@]}" --from-stage node --to-stage reader
phase metrics_and_local_judge
run_logged "$RUN_ROOT/logs/evaluation-$SLURM_JOB_ID.log" "$PYTHON" -u scripts/evaluate_observation_full_run.py \
  --root "$COME_FULL_ROOT" --dataset h2hmem --base-url "$BASE_URL" --concurrency 2
phase complete
echo "COME_H2HMEM_COMPLETE job=$SLURM_JOB_ID"
