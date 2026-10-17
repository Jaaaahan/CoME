#!/bin/bash
#SBATCH --job-name=graphmemix-qwen-judge
#SBATCH --partition=gpu3node
#SBATCH --nodelist=gpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/share/home/jyh/graphmemix/cluster/cuhk-suite/logs/%x-%j.out
#SBATCH --error=/share/home/jyh/graphmemix/cluster/cuhk-suite/logs/%x-%j.err
source /share/home/jyh/graphmemix/cluster/cuhk-suite/common.sh
PROJECT_ROOT=$EXPERIMENT_ROOT/source/graphmemix-2d8045d96918
export PROJECT_ROOT DATA_ROOT EXPERIMENT_ROOT SUITE_ROOT
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/scripts"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
cd "$PROJECT_ROOT"
# If the reuse step finished, finalize its report without starting another model.
if "$PYTHON" -u "$SUITE_ROOT/qwen_judge.py" --check-complete; then
  exec "$PYTHON" -u "$SUITE_ROOT/qwen_judge.py"
else
  check_status=$?
  [[ "$check_status" == 3 ]] || exit "$check_status"
fi
source "$SUITE_ROOT/process_lifecycle.sh"
"$PYTHON" - <<'PY'
import os,socket,torch
assert torch.cuda.device_count()==2
names=[torch.cuda.get_device_name(i) for i in range(2)]
assert all('4090' in name for name in names),names
print({'job':os.environ['SLURM_JOB_ID'],'node':socket.gethostname(),'GPUs':names},flush=True)
with socket.socket() as probe:
    probe.bind(('127.0.0.1',18262))
PY
export VLLM_WORKER_MULTIPROC_METHOD=spawn
setsid "$SERVING_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$DATA_ROOT/models/Qwen3-VL-8B-Instruct" \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct --host 127.0.0.1 --port 18262 \
  --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --max-num-seqs 2 --max-num-batched-tokens 4096 \
  --limit-mm-per-prompt '{"image":80,"video":0}' --skip-mm-profiling --enforce-eager \
  > "$EXPERIMENT_ROOT/logs/vllm-judge-$SLURM_JOB_ID.log" 2>&1 &
SERVER_PID=$!
run_logged "$EXPERIMENT_ROOT/logs/qwen-judge-$SLURM_JOB_ID.log" \
  "$PYTHON" -u "$SUITE_ROOT/qwen_judge.py" --concurrency 2
echo QWEN_JUDGE_JOB_COMPLETE
