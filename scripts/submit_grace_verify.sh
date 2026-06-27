#!/bin/bash
#SBATCH -J grace_melt_verify
#SBATCH -o /ptmp/hmai/pwa_melting/_verify_runs/slurm_%j.out
#SBATCH -e /ptmp/hmai/pwa_melting/_verify_runs/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60000
#SBATCH --time=00:25:00

# Self-contained (Raven SBATCH_EXPORT=NONE): set everything inside; log to /ptmp.
set -x
module purge 2>/dev/null || true
module load cuda/12.6 2>/dev/null || module load cuda/12.3-nvhpcsdk 2>/dev/null || true
export OMP_NUM_THREADS=4
export TF_CPP_MIN_LOG_LEVEL=1
nvidia-smi -L || echo "no nvidia-smi"

echo "================ V4: LAMMPS + pair_style grace coexistence iteration ================"
/ptmp/hmai/pwa_melting/.venv/bin/python \
    /ptmp/hmai/pwa_melting/scripts/_verify_lammps_grace_coex.py || echo "V4_FAILED"

echo "================ V3: GRACE-ASE Step-1 estimate ================"
/ptmp/hmai/pwa_melting/.venv-grace/bin/python \
    /ptmp/hmai/pwa_melting/scripts/_verify_grace_ase_step1.py || echo "V3_FAILED"

echo "ALL_DONE"
