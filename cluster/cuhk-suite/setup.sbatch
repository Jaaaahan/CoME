#!/bin/bash
#SBATCH --job-name=graphmemix-suite-setup
#SBATCH --partition=gpu2node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/share/home/jyh/graphmemix/cluster/cuhk-suite/logs/%x-%j.out
#SBATCH --error=/share/home/jyh/graphmemix/cluster/cuhk-suite/logs/%x-%j.err
set -eo pipefail
test -n "${SLURM_JOB_ID:-}"
echo "job=$SLURM_JOB_ID node=$(hostname)"
source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate graphmemix
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export PIP_CACHE_DIR=/share/workspace3/yhjia/graphmemix/cache/pip
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /share/home/jyh/graphmemix
export PYTHONPATH="$PWD:$PWD/src:$PWD/scripts"
python -m pip install -e '.[methods]' torch==2.8.0 torchvision==0.23.0 transformers==4.51.3 sentence-transformers==4.1.0 'accelerate>=1' pytest
python -m pip check
python -m pytest -q tests --junitxml=cluster/cuhk-suite/logs/tests-${SLURM_JOB_ID}.xml
python -m pip freeze > cluster/cuhk-suite/requirements-graphmemix.txt
python - <<'PY'
import torch,transformers,faiss
print({'torch':torch.__version__,'transformers':transformers.__version__,'faiss':faiss.__version__})
PY
/share/home/jyh/.conda/envs/atmbench/bin/python - <<'PY'
import torch,transformers,vllm
print({'serving_torch':torch.__version__,'serving_transformers':transformers.__version__,'vllm':vllm.__version__})
PY
echo SETUP_COMPLETE
