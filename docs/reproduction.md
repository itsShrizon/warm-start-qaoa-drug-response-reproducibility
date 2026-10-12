# Reproduction

Run these commands from a downloaded or cloned repository. Use Python 3.12.

## 1. Set up the environment

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on macOS/Linux.

```sh
python -m pip install -r reproduction/requirements.txt
python verify_manifest.py
cd reproduction
```

Check the manifest before rerunning analyses, which can overwrite result files.

## 2. Rebuild tables and figures

The repository includes the completed results. Run:

```sh
python build_improvement_outputs.py
python build_recovered_outputs.py
python build_larger_outputs.py
python build_figures.py
python build_revision_outputs.py
```

These commands write table fragments and figures to `overleaf/`.

## 3. Compile the documents

Upload the complete [reproduction/overleaf/](../reproduction/overleaf/) folder to Overleaf. Select `main.tex` for the manuscript or `supplement.tex` for the supplementary document. Use pdfLaTeX.

For a local build, use pdfLaTeX, BibTeX, then pdfLaTeX twice for the manuscript; use pdfLaTeX twice for the supplement.

## Optional recalculations

The commands below run from `reproduction/` after setup.

<details>
<summary>Reanalyze the archived hardware outputs</summary>

```sh
python reanalysis.py
python quality_analysis.py
python analyze_recovered_hardware.py
```

Then repeat step 2. Hardware analysis uses bundled records and runs offline.

</details>

<details>
<summary>Rerun panel simulations and classical searches</summary>

```sh
python sensitivity.py
python revision_experiments.py --mode classical --workers 6
python revision_experiments.py --mode panels --workers 6
python revision_experiments.py --mode grover
python revision_experiments.py --mode rescore
```

Training uses seeds 0–2, 60 Adam updates and learning rate 0.1. Annealing uses seeds 0–99 and at most 200 unique score evaluations per run. Frozen designs are included. Reconstructing them from original measurements requires the files listed in [Source data access](data-access.md). Small floating-point differences can affect training across platforms.

</details>

<details>
<summary>Recalculate larger statevectors</summary>

```sh
python larger_noiseless.py --qubits 18 --output analysis/local_18q.json
python revision_noiseless.py --qubits 25 --validate
python revision_noiseless.py --qubits 30
python revision_noiseless.py --qubits 35
```

For `larger_noiseless.py`, substitute 20, 21 or 25 as needed. The 30- and 35-qubit statevectors require 8 GiB and 256 GiB, respectively, plus working memory. Execution records include the original software versions. The archived Slurm scripts use the original cluster paths and allocations.

</details>

<details>
<summary>Run the existing scientific checks</summary>

```sh
python -m unittest discover -p 'test_*.py' -v
```

</details>
