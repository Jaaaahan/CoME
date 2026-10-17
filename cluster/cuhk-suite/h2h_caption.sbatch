#!/bin/bash
#SBATCH --job-name=h2h-qwen-captions
#SBATCH --partition=gpu3node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/share/home/jyh/graphmemix/cluster/cuhk-suite/logs/%x-%j.out
#SBATCH --error=/share/home/jyh/graphmemix/cluster/cuhk-suite/logs/%x-%j.err
source /share/home/jyh/graphmemix/cluster/cuhk-suite/h2h_common.sh
test -s "$DATA_ROOT/data/derived/h2hmem/evidence_supported_v1/preparation.json"
source "$SUITE_ROOT/h2h_server.sh"
run_logged "$EXPERIMENT_ROOT/logs/captions-$SLURM_JOB_ID.log" \
  "$PYTHON" -u "$SUITE_ROOT/caption_h2h.py" --base-url "$BASE_URL" --concurrency 2
echo H2H_CAPTION_JOB_COMPLETE
