#!/bin/bash -l
#SBATCH -J grace_elastic
#SBATCH -p gpudev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=240G
#SBATCH -t 00:15:00
#SBATCH -o /ptmp/hmai/pwa_elastic/verification/results/slurm_%j.out
#SBATCH -e /ptmp/hmai/pwa_elastic/verification/results/slurm_%j.err
#
# GRACE-2L-SMAX-large elastic constants for Al,Cu,Si,Fe,Ni,W on 4x A100 (gpudev).
# Each material runs as its own process pinned to one GPU (6 procs over 4 GPUs;
# gpu0/gpu1 each host 2, fine with TF_FORCE_GPU_ALLOW_GROWTH for the tiny cells).

set -u
cd /ptmp/hmai/pwa_elastic
source /ptmp/hmai/.mp_api_key                       # exports MP_API_KEY (mode 600, outside repo)
export GRACE_CACHE=/ptmp/hmai/grace_cache
export TF_CPP_MIN_LOG_LEVEL=3 TF_USE_LEGACY_KERAS=1 TF_FORCE_GPU_ALLOW_GROWTH=true
PY=.venv/bin/python

echo "== node $(hostname); date $(date) =="
nvidia-smi -L || { echo "FATAL: no nvidia-smi / no GPU"; exit 1; }

echo "== TensorFlow GPU visibility check =="
$PY - <<'EOF' || { echo "FATAL: TF cannot see a GPU"; exit 2; }
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print("TF physical GPUs:", gpus)
assert gpus, "TensorFlow does not see any GPU"
EOF

echo "== launching 6 materials across 4 GPUs =="
mats=(Al Cu Si Fe Ni W)
gpu=(0 1 2 3 0 1)
for i in "${!mats[@]}"; do
  m=${mats[$i]}
  CUDA_VISIBLE_DEVICES=${gpu[$i]} OMP_NUM_THREADS=8 \
    $PY verification/elastic_grace_vs_mp.py --material "$m" --out-dir verification/results \
    > "verification/results/${m}.stdout" 2> "verification/results/${m}.stderr" &
  echo "  launched $m on GPU ${gpu[$i]} (pid $!)"
done
wait
echo "== ALL_DONE $(date) =="
ls -la verification/results/*.csv
