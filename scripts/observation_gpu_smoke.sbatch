#!/bin/bash
#SBATCH --job-name=come-observation-smoke
#SBATCH --partition=gpu3node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00

set -eo pipefail
: "${COME_SMOKE_ROOT:?run prepare_observation_gpu_smoke.py first}"
DATA_ROOT=${COME_MODEL_ROOT:-/share/workspace3/yhjia/graphmemix}
PYTHON=${COME_PYTHON:-/share/home/jyh/.conda/envs/graphmemix/bin/python}
SERVING_PYTHON=${COME_SERVING_PYTHON:-/share/home/jyh/.conda/envs/atmbench/bin/python}
PROJECT_ROOT=$COME_SMOKE_ROOT/source
SUITE_ROOT=$PROJECT_ROOT/cluster/cuhk-suite
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/scripts:$SUITE_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$COME_SMOKE_ROOT/cache/huggingface
export NLTK_DATA=$DATA_ROOT/nltk_data
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export COME_SMOKE_TRACE=$COME_SMOKE_ROOT/requests.jsonl
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
cd "$PROJECT_ROOT"
source "$SUITE_ROOT/process_lifecycle.sh"
IFS=',' read -ra ASSIGNED_GPUS <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation missing}"
[[ ${#ASSIGNED_GPUS[@]} == 3 ]]
SERVING_GPUS=${ASSIGNED_GPUS[0]},${ASSIGNED_GPUS[1]}
EMBEDDING_GPU=${ASSIGNED_GPUS[2]}
PORT=$((28000 + SLURM_JOB_ID % 1000))
PROXY_PORT=$((PORT + 1000))
BASE_URL=http://127.0.0.1:$PROXY_PORT/v1
"$PYTHON" - <<'PY'
import hashlib, json, os, socket, torch
from pathlib import Path
run = Path(os.environ['COME_SMOKE_ROOT'])
for name, expected in json.loads((run / 'source-manifest.json').read_text()).items():
    assert hashlib.sha256((run / 'source' / name).read_bytes()).hexdigest() == expected, name
report = dict(job_id=os.environ['SLURM_JOB_ID'], host=socket.gethostname(),
              gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
(run / 'allocation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
PY
export VLLM_WORKER_MULTIPROC_METHOD=spawn
CUDA_VISIBLE_DEVICES=$SERVING_GPUS setsid "$SERVING_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$DATA_ROOT/models/Qwen3-VL-8B-Instruct" \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.80 --max-num-seqs 1 --max-num-batched-tokens 4096 \
  --limit-mm-per-prompt '{"image":80,"video":0}' --skip-mm-profiling --enforce-eager \
  > "$COME_SMOKE_ROOT/logs/vllm.log" 2>&1 &
SERVER_PID=$!
CUDA_VISIBLE_DEVICES= setsid "$SERVING_PYTHON" -u "$SUITE_ROOT/token_proxy.py" \
  --model-path "$DATA_ROOT/models/Qwen3-VL-8B-Instruct" \
  --upstream "http://127.0.0.1:$PORT" --port "$PROXY_PORT" \
  > "$COME_SMOKE_ROOT/logs/proxy.log" 2>&1 &
PROXY_PID=$!
run_logged "$COME_SMOKE_ROOT/logs/startup.log" "$PYTHON" -u "$SUITE_ROOT/check_server.py" \
  --base-url "$BASE_URL" --pid "$SERVER_PID" --startup-timeout-seconds 1200 \
  --output "$COME_SMOKE_ROOT/server.json"
export CUDA_VISIBLE_DEVICES=$EMBEDDING_GPU
for dataset in mem_gallery memeye; do
  common=("$PYTHON" -u scripts/run_graphmemix_release.py --config "$COME_SMOKE_ROOT/config.json" benchmark
    --dataset "$dataset" --backbone qwen3vl8b --backend standalone
    --data-root "$COME_SMOKE_ROOT/data" --artifact-root "$COME_SMOKE_ROOT/artifacts"
    --run-root "$COME_SMOKE_ROOT/runs" --base-url "$BASE_URL" --concurrency 1 --local-processes 1)
  run_logged "$COME_SMOKE_ROOT/logs/$dataset-offline.log" "${common[@]}" \
    --from-stage observations --to-stage checkpoint
  # Hash all offline artifacts before query stages. A later audit compares bytes.
  COME_SMOKE_DATASET=$dataset "$PYTHON" - <<'PY'
import hashlib, json, os
from pathlib import Path
run = Path(os.environ['COME_SMOKE_ROOT'])
dataset = os.environ['COME_SMOKE_DATASET']
root = run / 'artifacts/observations-v1'
paths = [root / 'observations' / (dataset + '.jsonl'), root / 'observations' / (dataset + '.manifest.json')]
paths += [p for folder in ('checkpoints', 'extraction') for p in (root / folder / dataset).rglob('*') if p.is_file()]
(run / (dataset + '-offline-hashes.json')).write_text(json.dumps({str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}, indent=2) + '\n')
PY
  run_logged "$COME_SMOKE_ROOT/logs/$dataset-query.log" "${common[@]}" \
    --from-stage priors --to-stage reader
done
echo "COME_OBSERVATION_GPU_SMOKE_COMPLETE job=$SLURM_JOB_ID"
