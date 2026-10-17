#!/bin/bash
#SBATCH --job-name=h2h-dense-index
#SBATCH --partition=gpu3node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/share/home/jyh/graphmemix/cluster/cuhk-suite/logs/%x-%j.out
#SBATCH --error=/share/home/jyh/graphmemix/cluster/cuhk-suite/logs/%x-%j.err
source /share/home/jyh/graphmemix/cluster/cuhk-suite/h2h_common.sh
"$PYTHON" - <<'PY'
import torch,os,json,hashlib
from pathlib import Path
assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0)
p=Path(os.environ['EXPERIMENT_ROOT'])/'artifacts/captions/h2hmem.jsonl'
s=json.loads(p.with_suffix('.summary.json').read_text())
assert s['complete'] and s['assets']==1300
assert hashlib.sha256(p.read_bytes()).hexdigest()==s['sidecar_sha256']
print('FROZEN_CAPTIONS_VERIFIED',flush=True)
PY
run_h2h_stages --from-stage checkpoint --to-stage relations 2>&1 | tee "$EXPERIMENT_ROOT/logs/index-$SLURM_JOB_ID.log"
test "${PIPESTATUS[0]}" -eq 0
echo H2H_INDEX_COMPLETE
