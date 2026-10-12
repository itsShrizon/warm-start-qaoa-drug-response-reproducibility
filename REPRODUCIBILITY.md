# Reproducibility

Install the pinned packages in `requirements.txt` with Python 3.12. The README commands reproduce the analyses and regenerate the tables and figures from bundled records. The saved arrays establish the exact feasible optima. All hardware analysis is offline.

## Rerun the simulations

The frozen panel designs are included, so preparation is unnecessary for these commands:

```sh
python sensitivity.py
python revision_experiments.py --mode classical --workers 6
python revision_experiments.py --mode panels --workers 6
python revision_experiments.py --mode grover
python revision_experiments.py --mode rescore
```

Training uses seeds 0, 1 and 2, 60 Adam updates and learning rate 0.1. Simulated annealing uses seeds 0–99 and at most 200 unique score evaluations per run. Floating-point results can differ slightly across platforms.

## Larger statevectors

```sh
python larger_noiseless.py --qubits 18 --output analysis/local_18q.json
python revision_noiseless.py --qubits 25 --validate
python revision_noiseless.py --qubits 30
python revision_noiseless.py --qubits 35
```

The 30- and 35-qubit statevectors require 8 GiB and 256 GiB respectively, plus working memory. Saved results allow table regeneration without repeating these calculations. Original execution records and software versions are preserved beside the results.

## Integrity and checks

Run `python verify_manifest.py` immediately after cloning to compare files with `SHA256SUMS.json`. Analysis reruns overwrite result files, so their hashes may subsequently change. The existing scientific checks can be run with `python -m unittest discover -p 'test_*.py' -v`.

## Compile the documents

From `overleaf/`, use pdfLaTeX, BibTeX, then pdfLaTeX twice for `main.tex`; use pdfLaTeX twice for `supplement.tex`. Alternatively, upload the complete folder to Overleaf. The repository includes the revised sources and existing figure PDFs.
