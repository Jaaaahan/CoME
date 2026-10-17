#!/bin/bash
#SBATCH --job-name=graphmemix-suite-server-probe
#SBATCH --partition=gpu2node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/share/home/jyh/graphmemix/cluster/cuhk-suite/logs/%x-%j.out
#SBATCH --error=/share/home/jyh/graphmemix/cluster/cuhk-suite/logs/%x-%j.err
source /share/home/jyh/graphmemix/cluster/cuhk-suite/common.sh
"$SERVING_PYTHON" - <<'PY'
import torch,json,os,socket
assert torch.cuda.device_count()==2
names=[torch.cuda.get_device_name(i) for i in range(2)]
assert all('4090' in n for n in names),names
print(json.dumps({'job':os.environ['SLURM_JOB_ID'],'host':socket.gethostname(),'GPUs':names,'torch':torch.__version__}),flush=True)
PY
PORT=$((18000 + SLURM_JOB_ID % 1000))
PROXY_PORT=$((PORT+1000))
export VLLM_WORKER_MULTIPROC_METHOD=spawn
"$SERVING_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$DATA_ROOT/models/Qwen3-VL-8B-Instruct" \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --max-num-seqs 2 --max-num-batched-tokens 4096 \
  --limit-mm-per-prompt '{"image":80,"video":0}' --skip-mm-profiling --enforce-eager \
  > "$EXPERIMENT_ROOT/logs/vllm-probe-$SLURM_JOB_ID.log" 2>&1 &
SERVER_PID=$!
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$SERVING_PYTHON" -u "$SUITE_ROOT/token_proxy.py" --model-path "$DATA_ROOT/models/Qwen3-VL-8B-Instruct" --upstream "http://127.0.0.1:$PORT" --port "$PROXY_PORT" > "$EXPERIMENT_ROOT/logs/token-proxy-$SLURM_JOB_ID.log" 2>&1 &
PROXY_PID=$!
trap 'kill -TERM "$PROXY_PID" "$SERVER_PID" 2>/dev/null || true; wait "$PROXY_PID" "$SERVER_PID" 2>/dev/null || true' EXIT
"$SERVING_PYTHON" -u "$SUITE_ROOT/check_server.py" --base-url "http://127.0.0.1:$PROXY_PORT/v1" --pid "$SERVER_PID" --output "$EXPERIMENT_ROOT/validation/server-probe-$SLURM_JOB_ID.json"
echo SERVER_PROBE_COMPLETE
