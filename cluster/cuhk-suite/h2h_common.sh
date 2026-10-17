source /share/home/jyh/graphmemix/cluster/cuhk-suite/common.sh
PROJECT_ROOT=$DATA_ROOT/experiments/standalone-4090-20261015/source/graphmemix-2d8045d96918
EXPERIMENT_ROOT=$DATA_ROOT/experiments/h2hmem-4090-20261017
export PROJECT_ROOT EXPERIMENT_ROOT DATA_ROOT SUITE_ROOT
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/scripts"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
PORT=18310
PROXY_PORT=19310
export BASE_URL=http://127.0.0.1:$PROXY_PORT/v1
mkdir -p "$EXPERIMENT_ROOT/logs" "$EXPERIMENT_ROOT/validation" "$EXPERIMENT_ROOT/metrics"
cd "$SUITE_ROOT"
sha256sum --quiet -c h2h-scripts.sha256
cd "$PROJECT_ROOT"
sha256sum --quiet -c STANDALONE_MANIFEST.sha256

run_h2h_stages() {
  "$PYTHON" -u scripts/run_graphmemix_release.py --config "$SUITE_ROOT/h2h-config.json" benchmark \
    --dataset h2hmem --backbone qwen3vl8b --backend standalone \
    --data-root "$DATA_ROOT/data/unified" --artifact-root "$EXPERIMENT_ROOT/artifacts" \
    --run-root "$EXPERIMENT_ROOT/runs" --base-url "$BASE_URL" --concurrency 2 \
    --local-processes 1 "$@"
}
