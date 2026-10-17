if [[ "${SLURM_JOB_NAME:-}" == "h2h-qwen-captions" ]] && [[ -s "$EXPERIMENT_ROOT/artifacts/captions/h2hmem.summary.json" ]]; then
  "$PYTHON" -u "$SUITE_ROOT/caption_h2h.py" --base-url "$BASE_URL" --concurrency 2
  echo H2H_CAPTIONS_REUSED_COMPLETE
  exit 0
fi
source "$SUITE_ROOT/process_lifecycle.sh"
"$PYTHON" - <<'PY'
import torch,os,socket
assert torch.cuda.device_count()==2
names=[torch.cuda.get_device_name(i) for i in range(2)]
assert all('4090' in n for n in names),names
print({'job':os.environ['SLURM_JOB_ID'],'host':socket.gethostname(),'GPUs':names},flush=True)
for port in (18310,19310):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',port))
PY
export VLLM_WORKER_MULTIPROC_METHOD=spawn
setsid "$SERVING_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$DATA_ROOT/models/Qwen3-VL-8B-Instruct" \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --max-num-seqs 2 --max-num-batched-tokens 4096 \
  --limit-mm-per-prompt '{"image":80,"video":0}' --skip-mm-profiling --enforce-eager \
  > "$EXPERIMENT_ROOT/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
SERVER_PID=$!
setsid "$SERVING_PYTHON" -u "$SUITE_ROOT/token_proxy.py" --model-path "$DATA_ROOT/models/Qwen3-VL-8B-Instruct" --upstream "http://127.0.0.1:$PORT" --port "$PROXY_PORT" > "$EXPERIMENT_ROOT/logs/token-proxy-$SLURM_JOB_ID.log" 2>&1 &
PROXY_PID=$!
setsid "$PYTHON" -u "$SUITE_ROOT/watch_server.py" --parent-pid "$$" --server-pid "$SERVER_PID" --proxy-pid "$PROXY_PID" --log "$EXPERIMENT_ROOT/logs/vllm-$SLURM_JOB_ID.log" > "$EXPERIMENT_ROOT/logs/watch-server-$SLURM_JOB_ID.log" 2>&1 &
WATCH_PID=$!
run_logged "$EXPERIMENT_ROOT/logs/startup-$SLURM_JOB_ID.log" "$PYTHON" -u "$SUITE_ROOT/check_server.py" --base-url "$BASE_URL" --pid "$SERVER_PID" --output "$EXPERIMENT_ROOT/validation/server-$SLURM_JOB_ID.json" --startup-timeout-seconds 3600
