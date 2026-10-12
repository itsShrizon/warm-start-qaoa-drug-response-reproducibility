"""Exact archived p=1 circuit calculation with O(2**q + chunk) memory.

No parameter optimization, hardware submission, or changes to source evidence.
Qubit zero is the most significant bit. Only feasible assignments use a bit matrix.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import platform
import socket
import time
import numpy as np
from reanalysis import ry, rz, parse_gates, qasm_probs
from analyze_recovered_hardware import expected_gates, gate_match, feasible_assignments

ROOT = Path(__file__).resolve().parent
CHUNK = 1 << 18


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def circuit_hash(gates):
    return hashlib.sha256(json.dumps(gates, separators=(',', ':')).encode()).hexdigest()


def apply_single_inplace(state, matrix, wire, nv, chunk=CHUNK):
    """Update disjoint amplitude pairs, with bounded temporary arrays."""
    stride = 1 << (nv - 1 - wire)
    # Pair indices enumerate the q-1 other bits and insert a zero at this wire.
    for start in range(0, len(state)//2, chunk):
        pairs = np.arange(start, min(start+chunk, len(state)//2), dtype=np.int64)
        zero = (pairs // stride) * (2*stride) + pairs % stride
        one = zero + stride
        a, b = state[zero], state[one]
        state[zero] = matrix[0, 0]*a + matrix[0, 1]*b
        state[one] = matrix[1, 0]*a + matrix[1, 1]*b


def chunked_state(r, chunk=CHUNK):
    nv = r['nv']
    if not 1 <= nv <= 25:
        raise ValueError('This bounded calculation supports only 1--25 qubits')
    state = np.empty(1 << nv, dtype=np.complex128)
    # Form initial amplitudes and the diagonal cost phase in integer-index chunks.
    for start in range(0, len(state), chunk):
        ids = np.arange(start, min(start+chunk, len(state)), dtype=np.int64)
        amplitude = np.ones(len(ids))
        energy = np.zeros(len(ids))
        for wire, (theta, h) in enumerate(zip(r['theta'], r['Hising'])):
            bit = (ids >> (nv-1-wire)) & 1
            amplitude *= np.where(bit, np.sin(theta/2), np.cos(theta/2))
            energy += h * (1 - 2*bit)
        for a, b, coefficient in r['Jising']:
            parity = ((ids >> (nv-1-a)) ^ (ids >> (nv-1-b))) & 1
            energy += coefficient * (1 - 2*parity)
        state[start:start+len(ids)] = amplitude * np.exp(-1j*r['gamma']*energy)
    for wire, theta in enumerate(r['theta']):
        mixer = ry(theta) @ rz(2*r['beta']) @ ry(-theta)
        apply_single_inplace(state, mixer, wire, nv, chunk)
    return state


def metrics(mass, values, optimal):
    feasibility = float(mass.sum())
    quality = (values-values.min())/(values.max()-values.min())
    return {'feasibility': feasibility,
            'optimum_probability': float(mass[optimal].sum()),
            'conditional_mean_score': float(mass @ values / feasibility) if feasibility else None,
            'conditional_mean_normalized_score': float(mass @ quality / feasibility) if feasibility else None}


def load_circuit(width):
    evidence = ROOT/'evidence/recovered_hardware'
    manifest = json.loads((evidence/'manifest.json').read_text())
    assert sha(evidence/'frozen_instances.json') == manifest['frozen_instances_sha256']
    for entry in manifest['files']+manifest['runners']:
        assert sha(evidence/entry['archived_path']) == entry['sha256']
    frozen = json.loads((evidence/'frozen_instances.json').read_text())
    tasks = []
    for path in sorted((evidence/'tasks').glob('*.json')):
        raw = json.loads(path.read_text())
        if len(raw['measuredQubits']) != width:
            continue
        assert raw['measuredQubits'] == list(range(width))
        gates = parse_gates(raw['additionalMetadata']['action']['source'])
        matches = [f['instance'] for f in frozen if gate_match(gates, expected_gates(f['instance']))]
        assert matches
        tasks.append((path, raw, gates, matches[0]))
    assert tasks and len({circuit_hash(t[2]) for t in tasks}) == 1
    return tasks


def calculate(width):
    start = time.perf_counter()
    tasks = load_circuit(width)
    r = tasks[0][3]
    state = chunked_state(r)
    norm = sum(float(np.vdot(state[i:i+CHUNK], state[i:i+CHUNK]).real)
               for i in range(0, len(state), CHUNK))
    assert abs(norm-1) < 1e-10
    independent_error = None
    if width == 18:
        independent = qasm_probs(tasks[0][2], width)
        independent_error = float(np.max(np.abs(abs(state)**2-independent)))
        assert independent_error < 1e-10
    bits, values = feasible_assignments(r)
    ids = bits @ (1 << np.arange(width-1, -1, -1, dtype=np.int64))
    assert len(np.unique(ids)) == r['n_feasible']
    optimal = np.isclose(values, values.max(), rtol=0, atol=1e-10)
    assert int(r['opt_bitstring'], 2) in ids[optimal]
    prior = np.sin(np.asarray(r['theta'])/2)**2
    methods = {
        'ideal_qaoa': metrics(abs(state[ids])**2, values, optimal),
        'initial_product': metrics(np.prod(np.where(bits, prior, 1-prior), axis=1), values, optimal),
        'uniform_feasible': metrics(np.full(len(ids), 1/len(ids)), values, optimal)}
    hardware = []
    for path, raw, gates, _ in tasks:
        shots = raw['taskMetadata']['shots']
        probabilities = raw['measurementProbabilities']
        counts = {s: round(p*shots) for s, p in probabilities.items()}
        assert all(abs(counts[s]-p*shots) < 1e-7 for s,p in probabilities.items())
        assert sum(counts.values()) == shots
        mass = np.array([counts.get(format(i, f'0{width}b'), 0)/shots for i in ids])
        hardware.append({'task_uuid': path.stem, 'source_sha256': sha(path),
                         'qasm_sha256': hashlib.sha256(raw['additionalMetadata']['action']['source'].encode()).hexdigest(),
                         'shots': shots, 'created_utc': raw['taskMetadata']['createdAt'],
                         **metrics(mass, values, optimal)})
    try:
        import resource
        peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        peak_kib = None
    return {'qubits': width, 'circuit_sha256': circuit_hash(tasks[0][2]),
            'feasible_states': len(ids), 'optimal_states': int(optimal.sum()),
            'norm': norm, 'independent_gate_max_probability_error': independent_error,
            'methods': methods, 'hardware': hardware,
            'execution': {'utc_completed': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          'elapsed_seconds': time.perf_counter()-start, 'peak_rss_kib_linux': peak_kib,
                          'hostname': socket.gethostname(), 'platform': platform.platform(),
                          'python': platform.python_version(), 'numpy': np.__version__,
                          'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                          'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
                          'state_dtype': 'complex128', 'chunk_size': CHUNK,
                          'code_sha256': {p: sha(ROOT/p) for p in
                              ('larger_noiseless.py', 'reanalysis.py', 'analyze_recovered_hardware.py', 'quality_analysis.py')},
                          'frozen_instances_sha256': sha(ROOT/'evidence/recovered_hardware/frozen_instances.json')}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qubits', type=int, choices=[18,20,21,25], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = calculate(args.qubits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
