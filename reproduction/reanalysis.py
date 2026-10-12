"""Offline, independently checked reanalysis of archived IonQ circuits.

No SDK credentials, remote execution, or paid tasks. Extracts only literal
instance definitions from the old runner; does not import or execute it.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import platform
from statistics import NormalDist

import numpy as np

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_ladder(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "LADDER" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("No literal LADDER found")


def evaluate(bits, r):
    """Independent exhaustive decoding; first axis is q[0] (MSB)."""
    blocks = bits.reshape(-1, r["n"], r["D"])
    occupancies = blocks.sum(axis=2)
    selected = occupancies > 0
    class_count = np.array([len({r['moa'][i] for i in np.flatnonzero(s)}) for s in selected])
    valid = ((occupancies <= 1).all(axis=1) & (selected.sum(axis=1) >= r['min_size'])
             & (selected.sum(axis=1) <= r['k']) & (class_count >= r['m_classes']))
    values = bits @ (np.array(r['h']).ravel() - r['lam'] * np.array(r['tox']).ravel())
    for a, b, coefficient in r['Jscore']:
        values += coefficient * bits[:, a] * bits[:, b]
    return valid, values


def ry(angle):
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(angle):
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])


def single(state, matrix, wire, nv):
    tensor = np.moveaxis(state.reshape([2] * nv), wire, 0)
    tensor = np.tensordot(matrix, tensor, axes=(1, 0))
    return np.moveaxis(tensor, 0, wire).reshape(-1)


def parse_gates(source):
    gates = []
    for line in source.splitlines():
        line = line.strip()
        one = re.fullmatch(r'(ry|rz)\(([-+0-9.eE]+)\) q\[(\d+)\];', line)
        two = re.fullmatch(r'cnot q\[(\d+)\], q\[(\d+)\];', line)
        if one:
            gates.append((one[1], float(one[2]), int(one[3])))
        elif two:
            gates.append(('cnot', int(two[1]), int(two[2])))
        elif line and not (line.startswith(('OPENQASM', 'bit[', 'qubit[')) or ' = measure ' in line):
            raise ValueError(f"Unsupported QASM statement: {line}")
    return gates


def qasm_probs(gates, nv):
    state = np.zeros(1 << nv, complex)
    state[0] = 1
    indices = np.arange(1 << nv)
    for name, a, b in gates:
        if name == 'cnot':
            source = indices ^ (((indices >> (nv - 1 - a)) & 1) << (nv - 1 - b))
            state = state[source]
        else:
            state = single(state, ry(a) if name == 'ry' else rz(a), b, nv)
    assert abs(np.linalg.norm(state) - 1) < 1e-10
    return abs(state) ** 2


def formula_probs(r, bits):
    state = np.array([1.0 + 0j])
    for theta in r['theta']:
        state = np.kron(state, np.array([math.cos(theta / 2), math.sin(theta / 2)]))
    z = 1.0 - 2.0 * bits
    energy = z @ np.asarray(r['Hising'])
    for a, b, coefficient in r['Jising']:
        energy += coefficient * z[:, a] * z[:, b]
    state *= np.exp(-1j * r['gamma'] * energy)
    for wire, theta in enumerate(r['theta']):
        mixer = ry(theta) @ rz(2 * r['beta']) @ ry(-theta)
        state = single(state, mixer, wire, r['nv'])
    return abs(state) ** 2


def wilson(k, n, alpha=0.05):
    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = k / n
    center = (p + z*z / (2*n)) / (1 + z*z / n)
    half = z * math.sqrt(p * (1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [center - half, center + half]


def expected_best_normalized(p, quality, valid, shots):
    # Failure (no feasible output) receives zero; valid quality is rescaled [0,1].
    y = np.where(valid, quality, 0)
    values = np.unique(y)
    mass = np.array([p[y == v].sum() for v in values])
    cdf = np.cumsum(mass).clip(0, 1)
    power = cdf ** shots
    return float(np.dot(values, power - np.r_[0, power[:-1]]))


def analyze(raw, r):
    nv = r['nv']
    meta = raw['taskMetadata']
    assert meta['status'] == 'COMPLETED'
    assert raw['measuredQubits'] == list(range(nv))
    assert meta['deviceParameters']['paradigmParameters']['qubitCount'] == nv
    shots = meta['shots']
    counts = np.zeros(1 << nv, dtype=int)
    for bitstring, probability in raw['measurementProbabilities'].items():
        assert len(bitstring) == nv and set(bitstring) <= {'0', '1'}
        count = round(probability * shots)
        assert abs(count - probability * shots) < 1e-7
        counts[int(bitstring, 2)] = count
    assert counts.sum() == shots
    indices = np.arange(1 << nv)
    bits = ((indices[:, None] >> np.arange(nv - 1, -1, -1)) & 1).astype(int)
    valid, value = evaluate(bits, r)
    # Separate construction in assignment space validates the feasible set.
    assignment_ids = []
    for assignment in itertools.product(range(-1, r['D']), repeat=r['n']):
        chosen = [i for i,d in enumerate(assignment) if d >= 0]
        if r['min_size'] <= len(chosen) <= r['k'] and len({r['moa'][i] for i in chosen}) >= r['m_classes']:
            assignment_ids.append(sum(1 << (nv - 1 - (i*r['D'] + assignment[i])) for i in chosen))
    assert sorted(assignment_ids) == list(np.flatnonzero(valid))
    assert valid.sum() == r['n_feasible']
    best, worst = float(value[valid].max()), float(value[valid].min())
    optimal = valid & np.isclose(value, best, rtol=0, atol=1e-10)
    assert abs(best - r['ideal_best'][2]) < 1e-5
    assert optimal[int(r['opt_bitstring'], 2)]
    gates = parse_gates(raw['additionalMetadata']['action']['source'])
    # Verify gate-by-gate identity against the recorded instance, not just width.
    expected = [('ry', t, i) for i,t in enumerate(r['theta'])]
    expected += [('rz', 2*r['gamma']*h, i) for i,h in enumerate(r['Hising']) if abs(h)>1e-12]
    for a,b,j in r['Jising']:
        expected += [('cnot', a,b), ('rz', 2*r['gamma']*j,b), ('cnot', a,b)]
    for i,t in enumerate(r['theta']):
        expected += [('ry', -t,i), ('rz', 2*r['beta'],i), ('ry', t,i)]
    assert len(gates) == len(expected)
    for observed, reference in zip(gates, expected):
        assert observed[0] == reference[0] and observed[2] == reference[2]
        assert abs(observed[1] - reference[1]) < 1e-10
    ideal = qasm_probs(gates, nv)
    analytic = formula_probs(r, bits)
    simulator_error = float(np.max(abs(ideal - analytic)))
    assert simulator_error < 1e-10
    one_prob = np.sin(np.asarray(r['theta'])/2)**2
    warm = np.prod(np.where(bits == 1, one_prob, 1-one_prob), axis=1)
    uniform = np.full(1 << nv, 1/(1 << nv))
    feasible_uniform = valid.astype(float) / valid.sum()
    hardware = counts / shots
    quality = (value - worst)/(best - worst)
    methods = {}
    for name,p in [('uniform_bits',uniform),('warm_start_only',warm),('ideal_qaoa',ideal),('hardware',hardware),('uniform_feasible',feasible_uniform)]:
        mass = float(p[valid].sum())
        opt_p = float(p[optimal].sum())
        methods[name] = {
            'feasibility':mass, 'optimum_probability':opt_p,
            'conditional_mean_score':float(np.dot(p[valid],value[valid])/mass),
            'conditional_mean_normalized_score':float(np.dot(p[valid],quality[valid])/mass),
        }
        if name != 'hardware':
            methods[name]['probability_find_optimum_200'] = -math.expm1(shots * math.log1p(-opt_p)) if opt_p < 1 else 1.0
            methods[name]['expected_best_normalized_score_200'] = expected_best_normalized(p,quality,valid,shots)
    k, opt_k = int(counts[valid].sum()), int(counts[optimal].sum())
    methods['hardware'].update({'valid_count':k, 'valid_ci95':wilson(k,shots),
       'optimum_count':opt_k,'optimum_ci95':wilson(opt_k,shots),
       'observed_best_score':float(value[valid & (counts>0)].max())})
    return {
        'name':r['name'],'qubits':nv,'drugs':r['n'],'doses':r['D'],
        'shots':shots,'task_id':meta['id'],'created_utc':meta['createdAt'],
        'device':meta['deviceId'],'cnot_count':sum(g[0]=='cnot' for g in gates),
        'gate_count':len(gates),'feasible_states':int(valid.sum()),
        'optimum_score':best,'minimum_feasible_score':worst,
        'optimum_bitstrings':[format(i,f'0{nv}b') for i in np.flatnonzero(optimal)],
        'simulator_max_abs_error':simulator_error,
        'qasm':raw['additionalMetadata']['action']['source'],
        'methods':methods,
        'histogram':{format(i,f'0{nv}b'):int(counts[i]) for i in np.flatnonzero(counts)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=None,
                        help='Original Pawsey workspace; omitted to use bundled local evidence')
    args = parser.parse_args()
    bundled = HERE/'evidence'
    if args.source:
        runner = args.source/'projects/drug-search/notebooks/braket_noise_ladder.py'
        ladder = extract_ladder(runner)
        runner_hash = digest(runner)
    else:
        ladder = json.loads((bundled/'frozen_ladder.json').read_text())
        provenance = json.loads((bundled/'provenance.json').read_text())
        runner_hash = provenance['runner_sha256']
        expected_hashes = {item['file'].replace('\\', '/').rsplit('/', 1)[-1]: item['sha256']
                           for item in provenance['sources']}
    rows, sources = [], []
    for r in ladder:
        input_dir = args.source/'Results' if args.source else bundled/'tasks'
        matches = list(input_dir.glob(f"ionq_forte1_ladder_{r['nv']:02d}q_*.json"))
        assert len(matches) == 1
        path = matches[0]
        if not args.source and digest(path) != expected_hashes.get(path.name):
            raise ValueError(f'Archived evidence hash mismatch: {path.name}')
        row = analyze(json.loads(path.read_text(encoding='utf-8-sig')),r)
        row['source_sha256'] = digest(path)
        rows.append(row)
        sources.append({'file':str(path),'sha256':digest(path)})
        if args.source:
            (bundled/'tasks').mkdir(parents=True,exist_ok=True)
            # Exact local evidence copy; no upload is performed.
            (bundled/'tasks'/path.name).write_bytes(path.read_bytes())
        print(r['name'], {m:round(v['feasibility'],5) for m,v in row['methods'].items()}, flush=True)
    out = HERE/'analysis'
    out.mkdir(exist_ok=True)
    result = {'analysis':'retrospective reanalysis; no new hardware tasks',
        'runner_sha256':runner_hash,'sources':sources,'rows':rows,
        'software':{'numpy':np.__version__,'python':platform.python_version()}}
    (out/'verified_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    # Local author-review evidence; redistribution permission must be confirmed.
    (out/'frozen_ladder.json').write_text(json.dumps(ladder,indent=2)+'\n',encoding='utf-8')
    if args.source:
        (bundled/'frozen_ladder.json').write_text(json.dumps(ladder,indent=2)+'\n',encoding='utf-8')
        (bundled/'provenance.json').write_text(json.dumps({'runner_sha256':runner_hash,'sources':sources},indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
