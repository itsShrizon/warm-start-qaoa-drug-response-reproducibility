#!/bin/bash --login
#SBATCH --account=pawsey1287-q
#SBATCH --partition=quantum
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --job-name=frontiers-panel-check
#SBATCH --output=/scratch/pawsey1287/shrizon/frontiers_revision_20260905/panel_%j.out
set -euo pipefail
module load cray-python/3.11.7
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /scratch/pawsey1287/shrizon/frontiers_revision_20260905
python3 sensitivity.py --smoke --output analysis/sensitivity_smoke_pawsey.json
python3 sensitivity.py --output analysis/sensitivity_results.json
