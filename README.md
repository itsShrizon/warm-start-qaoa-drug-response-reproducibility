# Warm-start QAOA drug-response reproducibility

Code and archived results for **Feasibility and optimum recovery in warm-start quantum optimization for a drug-response model on a trapped-ion processor**. The study covers 25 IonQ Forte-1 executions, 5,300 shots, 18 circuits and 14 optimization instances.

## Reproduce the analysis

Use Python 3.12 and an isolated environment:

```sh
python -m venv .venv
# Activate .venv using your operating system's command.
python -m pip install -r requirements.txt
python reanalysis.py
python quality_analysis.py
python analyze_recovered_hardware.py
python build_improvement_outputs.py
python build_recovered_outputs.py
python build_larger_outputs.py
python build_figures.py
python build_revision_outputs.py
```

These commands use saved hardware outputs and simulations. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) to rerun simulations and check file hashes.

## Supplementary material

Compile `overleaf/supplement.tex` for **Tables S1–S11 and Figures S1–S3**. Compile `overleaf/main.tex` for the manuscript. Upload the contents of `overleaf/` together when using Overleaf. A [short supplementary guide](SUPPLEMENTARY.md) explains every table and figure.

## Repository contents

- `evidence/`: frozen models, designs, archived circuits and hardware outcomes.
- `analysis/`: completed simulations and numerical results.
- `overleaf/`: manuscript, supplementary tables and figures.
- `SHA256SUMS.json`: file integrity manifest.

Original DREAM source tables must be obtained under their provider's access conditions; see [DATA_ACCESS.md](DATA_ACCESS.md). Use release **v1.0.0** to identify this snapshot.
