#!/bin/bash --login
#SBATCH --account=pawsey1287-q
#SBATCH --partition=quantum
#SBATCH --nodes=1
#SBATCH --ntasks=288
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --job-name=frontiers-revision
#SBATCH --output=/scratch/pawsey1287/shrizon/frontiers_revision_20260910/run_%j.out
set -euo pipefail
module load cray-python/3.11.7
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=72
export PIP_CACHE_DIR=/scratch/pawsey1287/shrizon/pipcache
cd /scratch/pawsey1287/shrizon/frontiers_revision_20260910
python3 -m venv venv --system-site-packages
source venv/bin/activate
python -m pip install --no-cache-dir numba==0.60.0
python -c 'import numpy, numba; print(numpy.__version__, numba.__version__)'
python -m unittest test_revision -v
python revision_experiments.py --mode classical --workers 12 &
classical_pid=$!
python revision_experiments.py --mode panels --workers 12 &
panels_pid=$!
python revision_experiments.py --mode grover
python revision_noiseless.py --qubits 25 --validate
python revision_noiseless.py --qubits 30
python revision_noiseless.py --qubits 35
wait "$classical_pid"
wait "$panels_pid"
python revision_experiments.py --mode rescore
