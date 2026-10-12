"""Freeze four panels and source provenance before running their optimization."""
from pathlib import Path
import hashlib
import itertools
import json
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    destination = ROOT / 'evidence/model_source'
    destination.mkdir(exist_ok=True)
    for name in ('CAMA1c7_drugs.txt', 'CAMA1c7_D3_meta.json', 'CAMA1c7_D3_h.csv', 'CAMA1c7_D3_tox.csv', 'CAMA1c7_D3_Jij.csv'):
        if not (destination / name).is_file():
            raise FileNotFoundError(f'Missing bundled coefficient file: {name}')
    drugs = (destination / 'CAMA1c7_drugs.txt').read_text().splitlines()
    meta = json.loads((destination / 'CAMA1c7_D3_meta.json').read_text())
    h = np.loadtxt(destination / 'CAMA1c7_D3_h.csv', delimiter=',')
    tox = np.loadtxt(destination / 'CAMA1c7_D3_tox.csv', delimiter=',')
    matrix = np.loadtxt(destination / 'CAMA1c7_D3_Jij.csv', delimiter=',')
    choices = list(itertools.combinations(range(len(drugs)), 4))
    selection = np.random.default_rng(20260905).choice(len(choices), 4, replace=False)
    panels = []
    for index, choice in enumerate(selection):
        selected = choices[choice]
        wires = [i * 3 + d for i in selected for d in range(3)]
        j = matrix[np.ix_(wires, wires)]
        panels.append({'name': f'panel_{index+1}', 'source_indices': list(selected),
            'n': 4, 'D': 3, 'nv': 12, 'min_size': 2, 'k': 3, 'm_classes': 2,
            'lam': 0.6, 'drugs': [drugs[i] for i in selected],
            'moa': [meta['moa_class'][drugs[i]] for i in selected],
            'h': h[list(selected)].tolist(), 'tox': tox[list(selected)].tolist(),
            'Jscore': [[a, b, float(j[a, b])] for a in range(12) for b in range(a+1, 12) if j[a,b] != 0]})
    output = ROOT / 'evidence/sensitivity_design.json'
    payload = {'panel_selection_seed': 20260905, 'training_seeds': [0,1,2],
               'steps': 60, 'learning_rate': 0.1, 'panels': panels}
    if output.exists():
        assert json.loads(output.read_text()) == payload, 'Refusing to alter the frozen design'
    else:
        output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'panels': [p['drugs'] for p in panels],
                      'design_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
