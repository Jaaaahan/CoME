#!/bin/bash
source /share/home/jyh/graphmemix/cluster/cuhk-suite/common.sh
PROJECT_ROOT=$EXPERIMENT_ROOT/source/graphmemix-2d8045d96918
export PROJECT_ROOT DATA_ROOT EXPERIMENT_ROOT SUITE_ROOT
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/scripts"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$PYTHON" -u "$SUITE_ROOT/qwen_judge.py" --base-url http://127.0.0.1:18262/v1 --concurrency 2
