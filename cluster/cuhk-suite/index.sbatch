#!/bin/bash
#SBATCH --job-name=graphmemix-suite-index
#SBATCH --partition=gpu2node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/share/home/jyh/graphmemix/cluster/cuhk-suite/logs/%x-%j.out
#SBATCH --error=/share/home/jyh/graphmemix/cluster/cuhk-suite/logs/%x-%j.err
source /share/home/jyh/graphmemix/cluster/cuhk-suite/common.sh
"$PYTHON" -m pip install nltk==3.9.2
"$PYTHON" -m pip freeze > "$SUITE_ROOT/requirements-graphmemix.txt"
"$PYTHON" -u "$SUITE_ROOT/verify_inputs.py"
for DATASET in atm memeye mem_gallery; do
  "$PYTHON" -u scripts/run_graphmemix_release.py --config "$SUITE_ROOT/suite-config.json" benchmark \
    --dataset "$DATASET" --backbone qwen3vl8b --backend standalone \
    --data-root "$DATA_ROOT/data/unified" --artifact-root "$EXPERIMENT_ROOT/artifacts" \
    --run-root "$EXPERIMENT_ROOT/runs" --base-url http://127.0.0.1:18786/v1 \
    --to-stage relations 2>&1 | tee "$EXPERIMENT_ROOT/logs/index-$DATASET-$SLURM_JOB_ID.log"
done
echo INDEX_SUITE_COMPLETE
