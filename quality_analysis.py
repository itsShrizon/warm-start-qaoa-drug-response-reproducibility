"""Feasible-score distributions and a budgeted classical search, without quantum jobs."""
from pathlib import Path
import csv
import itertools
import json
import time
import numpy as np
from reanalysis import evaluate

ROOT = Path(__file__).resolve().parent


def assignment_valid(a, r):
    selected = [i for i, d in enumerate(a) if d >= 0]
    return (r['min_size'] <= len(selected) <= r['k'] and
            len({r['moa'][i] for i in selected}) >= r['m_classes'])


def assignment_score(a, r):
    wires = {i*r['D']+d for i,d in enumerate(a) if d >= 0}
    linear = np.asarray(r['h']).ravel() - r['lam']*np.asarray(r['tox']).ravel()
    return float(sum(linear[v] for v in wires) +
                 sum(j for x,y,j in r['Jscore'] if x in wires and y in wires))


def greedy_local(r, budget=200):
    """Greedy addition, then best-improving reassignment/add/drop/swap; no optimum input."""
    start = time.perf_counter()
    cache = {}
    best_assignment, best_value = None, -float('inf')

    def score(a):
        nonlocal best_assignment, best_value
        if a not in cache:
            if len(cache) == budget:
                return None
            cache[a] = assignment_score(a, r)
        value = cache[a]
        if assignment_valid(a, r) and value > best_value + 1e-12:
            best_assignment, best_value = a, value
        return value

    current = (-1,) * r['n']
    score(current)
    # Construct the required two-drug seed. Score calls on partial assignments count.
    for _ in range(r['min_size']):
        candidates = []
        for i in range(r['n']):
            if current[i] != -1:
                continue
            for d in range(r['D']):
                a = current[:i] + (d,) + current[i+1:]
                value = score(a)
                if value is not None:
                    candidates.append((value, a))
        if not candidates:
            break
        current = max(candidates, key=lambda item: (item[0], tuple(-d for d in item[1])))[1]
    while len(cache) < budget and assignment_valid(current, r):
        neighbors = set()
        for i in range(r['n']):
            for d in range(-1, r['D']):
                a = current[:i] + (d,) + current[i+1:]
                if a != current and assignment_valid(a, r):
                    neighbors.add(a)
        for i in range(r['n']):
            if current[i] < 0:
                continue
            for j in range(r['n']):
                if current[j] >= 0:
                    continue
                for d in range(r['D']):
                    a = list(current)
                    a[i], a[j] = -1, d
                    if assignment_valid(a, r):
                        neighbors.add(tuple(a))
        winner, winner_value = current, cache[current]
        for candidate in sorted(neighbors):
            value = score(candidate)
            if value is not None and value > winner_value + 1e-12:
                winner, winner_value = candidate, value
        if winner == current:
            break
        current = winner
    return {'assignment': best_assignment, 'score': best_value,
            'objective_evaluations': len(cache), 'budget': budget,
            'runtime_seconds': time.perf_counter()-start}


def distribution(r, histogram):
    nv = r['nv']
    indices = np.arange(1 << nv)
    bits = ((indices[:,None] >> np.arange(nv-1, -1, -1)) & 1)
    valid, scores = evaluate(bits, r)
    low, high = float(scores[valid].min()), float(scores[valid].max())
    counts = np.zeros(1 << nv, dtype=int)
    for bitstring, count in histogram.items():
        counts[int(bitstring,2)] = count
    samples = np.repeat(scores[valid], counts[valid])
    normalized = (samples-low)/(high-low)
    search = greedy_local(r)
    # The optimum is consulted only after the search has finished.
    search['normalized_gap'] = (high-search['score'])/(high-low)
    search['found_optimum'] = bool(abs(search['score']-high) < 1e-10)
    values, multiplicities = np.unique(samples, return_counts=True)
    return {'name': r['name'], 'qubits': nv, 'valid_count': len(samples),
            'total_shots': int(counts.sum()), 'optimum': high, 'worst_feasible': low,
            'hardware_mean_normalized': float(normalized.mean()),
            'hardware_quantiles_normalized': np.quantile(normalized,[.25,.5,.75]).tolist(),
            'hardware_best_score': float(samples.max()),
            'hardware_best_normalized_gap': float(1-normalized.max()),
            'feasible_score_histogram': [{'score':float(v),'count':int(c)} for v,c in zip(values,multiplicities)],
            'greedy_local': search}


def main():
    ladder = json.loads((ROOT/'evidence/frozen_ladder.json').read_text())
    verified = json.loads((ROOT/'analysis/verified_results.json').read_text())
    rows = [distribution(r, record['histogram']) for r,record in zip(ladder,verified['rows'])]
    output = {'normalization': '(score - worst feasible)/(optimum - worst feasible)',
              'hardware_conditioning': 'Valid shots only; validity reported separately; descriptive within one batch',
              'classical_cost': 'Unique objective evaluations including partial construction; no quantum cost equivalence',
              'rows': rows}
    (ROOT/'analysis/quality_results.json').write_text(json.dumps(output,indent=2)+'\n')
    with (ROOT/'analysis/quality_summary.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['qubits','valid_shots','hardware_mean_quality','hardware_best_gap','greedy_gap','greedy_evaluations'])
        for r in rows:
            writer.writerow([r['qubits'],r['valid_count'],r['hardware_mean_normalized'],
                r['hardware_best_normalized_gap'],r['greedy_local']['normalized_gap'],r['greedy_local']['objective_evaluations']])
    for r in rows:
        print(r['qubits'], 'hardware best gap',round(r['hardware_best_normalized_gap'],4),
              'greedy gap',round(r['greedy_local']['normalized_gap'],4),
              'evaluations',r['greedy_local']['objective_evaluations'])


if __name__ == '__main__':
    main()
