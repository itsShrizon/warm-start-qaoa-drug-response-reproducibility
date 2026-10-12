#!/bin/bash --login
#SBATCH --account=pawsey1287-q
#SBATCH --partition=quantum
#SBATCH --nodes=1
#SBATCH --ntasks=72
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --array=0-3%1
#SBATCH --job-name=frontiers-larger-noiseless
#SBATCH --output=/scratch/pawsey1287/shrizon/frontiers_larger_20260906/run_%A_%a.out
set -euo pipefail
module load cray-python/3.11.7
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /scratch/pawsey1287/shrizon/frontiers_larger_20260906
python3 -m unittest test_larger_noiseless -v
widths=(18 20 21 25)
python3 larger_noiseless.py --qubits "${widths[$SLURM_ARRAY_TASK_ID]}" --output "analysis/larger_noiseless/${widths[$SLURM_ARRAY_TASK_ID]}q.json"
