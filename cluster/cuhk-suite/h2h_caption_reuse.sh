#!/bin/bash
source /share/home/jyh/graphmemix/cluster/cuhk-suite/h2h_common.sh
set -o pipefail
export BASE_URL=http://127.0.0.1:19262/v1
echo "H2H captions sharing Qwen service: job=$SLURM_JOB_ID step=$SLURM_STEP_ID"
"$PYTHON" -u "$SUITE_ROOT/caption_h2h.py" --base-url "$BASE_URL" --concurrency 1 \
  2>&1 | tee "$EXPERIMENT_ROOT/logs/caption-reuse-$SLURM_JOB_ID-$SLURM_STEP_ID.log"
