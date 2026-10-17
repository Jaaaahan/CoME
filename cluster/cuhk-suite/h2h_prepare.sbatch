#!/bin/bash
#SBATCH --job-name=h2h-data-preflight
#SBATCH --partition=gpu3node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/share/home/jyh/graphmemix/cluster/cuhk-suite/logs/%x-%j.out
#SBATCH --error=/share/home/jyh/graphmemix/cluster/cuhk-suite/logs/%x-%j.err
source /share/home/jyh/graphmemix/cluster/cuhk-suite/h2h_common.sh
"$PYTHON" -u "$SUITE_ROOT/prepare_h2h.py" --data-root "$DATA_ROOT" --check-env
echo H2H_PREFLIGHT_COMPLETE
