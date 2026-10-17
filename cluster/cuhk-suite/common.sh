set -eo pipefail
test -n "${SLURM_JOB_ID:-}"
PROJECT_ROOT=/share/home/jyh/graphmemix
SUITE_ROOT=$PROJECT_ROOT/cluster/cuhk-suite
DATA_ROOT=/share/workspace3/yhjia/graphmemix
EXPERIMENT_ROOT=$DATA_ROOT/experiments/standalone-4090-20261015
PYTHON=/share/home/jyh/.conda/envs/graphmemix/bin/python
SERVING_PYTHON=/share/home/jyh/.conda/envs/atmbench/bin/python
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src:$PROJECT_ROOT/scripts"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=$DATA_ROOT/cache/huggingface
export PIP_CACHE_DIR=$DATA_ROOT/cache/pip
export NLTK_DATA=$DATA_ROOT/nltk_data
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
mkdir -p "$EXPERIMENT_ROOT/logs" "$EXPERIMENT_ROOT/validation" "$EXPERIMENT_ROOT/metrics"
cd "$PROJECT_ROOT"
echo "job=$SLURM_JOB_ID node=$(hostname) cpus=$SLURM_CPUS_PER_TASK visible_gpus=${CUDA_VISIBLE_DEVICES:-scheduler-default}"
