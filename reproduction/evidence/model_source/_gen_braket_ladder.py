#!/usr/bin/env python3
"""Generate a self-contained Braket HYBRID-JOB script that runs a LADDER of
increasing dose-instance sizes on a real QPU and measures quality-decay-vs-noise.

This is the honest real-hardware answer to "run the big problem on a QPU": the
FULL 20-30 drug x D=5 problem needs 100-180 qubits (one-hot (D+1)/drug), which no
gate-model QPU on Braket can run (IonQ Forte 36, IQM Emerald 54, Rigetti 108 --
and even where the qubit count fits, a fully-connected cost Hamiltonian's depth
drowns in NISQ noise). So instead we push the SAME method up the size ladder as
far as real hardware holds, and plot where it breaks -- which is exactly the
"NISQ noise-scaling section" Prof Rajib's plan (DATASET_LOG.md §7) reserves the
hardware budget for: "run the instance at 4->12 qubits on real QPUs and plot
quality decay vs the noise-free simulation."

Each rung: pick the top-n sub-clique, pre-optimise QAOA params on the simulator,
bake the circuit + an inline exact feasibility/scoring test. The emitted job
runs every rung on the noise-free LocalSimulator (free, in-job) AND on the
device the job was submitted against, and reports per rung:
  feasible-shot fraction (hardware vs ideal vs uniform-random baseline),
  best feasible combo recovered / true optimum value,
  probability mass on the true optimum.

Writes: notebooks/braket_noise_ladder.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dose_instance import DoseScorer
from dose_qaoa import build_dose_qubo
from qaoa import _qubo_to_ising
from _azdream_hardware_coda import (LAM, build_circuit_pieces, make_circuit,
                                    optimise_on_sim)
from _azdream_quantum_tier import pick_subset

# (n_drugs, D) rungs ordered by qubit count nv = n*D (penalty encoding).
RUNGS = [(3, 2), (4, 2), (3, 3), (5, 2), (4, 3), (5, 3)]   # 6, 8, 9, 10, 12, 15 qubits


def _rand_feasible_fraction(s, rng, trials=20000):
    n, D, nv = s.n, s.D, s.nv
    good = 0
    for _ in range(trials):
        bits = rng.integers(0, 2, nv)
        a = s.onehot_to_assign(bits)
        if a is not None and s.feasible(a):
            good += 1
    return good / trials


def build_rung(n, D, rng):
    s = DoseScorer(D=D).subset(pick_subset(DoseScorer(D=D), n))
    nv, H_cost, theta = build_circuit_pieces(s, LAM)
    ansatz = make_circuit(nv, H_cost, theta, 1)
    params = optimise_on_sim(nv, H_cost, ansatz, 1)
    gamma, beta = float(params[0, 0]), float(params[0, 1])

    Q = build_dose_qubo(s, LAM)
    hh, JJ, _ = _qubo_to_ising(Q)

    Hising = [round(float(x), 8) for x in hh]
    Jising = [[i, j, round(float(JJ[i, j]), 8)] for i in range(nv)
              for j in range(i + 1, nv) if abs(JJ[i, j]) > 1e-12]
    Jscore = [[a, b, round(float(s.J[a, b]), 8)] for a in range(nv)
              for b in range(a + 1, nv) if abs(s.J[a, b]) > 1e-12]

    best = max(s.all_feasible(), key=lambda r: r[1] - LAM * r[2])
    opt_bits = "".join(str(int(x)) for x in s.assign_to_onehot(best[0]))
    n_feasible = sum(1 for _ in s.all_feasible())
    return {
        "name": "%dq_%ddrug_D%d" % (nv, n, D), "nv": nv, "n": n, "D": D,
        "drugs": list(s.drugs), "min_size": s.min_size, "k": s.k,
        "m_classes": s.m_classes, "moa": [s.moa[d] for d in s.drugs],
        "h": [[round(float(x), 8) for x in row] for row in s.h],
        "tox": [[round(float(x), 8) for x in row] for row in s.tox],
        "Jscore": Jscore, "theta": [round(float(x), 8) for x in theta],
        "gamma": gamma, "beta": beta, "Hising": Hising, "Jising": Jising,
        "lam": LAM, "n_twoqubit_gates": 2 * len(Jising), "n_feasible": n_feasible,
        "opt_bitstring": opt_bits, "opt_label": s.label(best[0]),
        "ideal_best": [round(float(best[1]), 6), round(float(best[2]), 6),
                       round(float(best[1] - LAM * best[2]), 6)],
        "random_feasible_frac": round(_rand_feasible_fraction(s, rng), 4),
    }


def main():
    rng = np.random.default_rng(0)
    ladder = []
    for n, D in RUNGS:
        r = build_rung(n, D, rng)
        ladder.append(r)
        print("  built %-14s nv=%2d  2q-gates=%3d  feasible=%4d  ideal_best_value=%.3f  rand_frac=%.3f"
              % (r["name"], r["nv"], r["n_twoqubit_gates"], r["n_feasible"],
                 r["ideal_best"][2], r["random_feasible_frac"]))

    data = json.dumps(ladder, indent=1)
    script = JOB_TEMPLATE.replace("__LADDER_DATA__", data)
    outdir = os.path.join(os.path.dirname(__file__), "..", "notebooks")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "braket_noise_ladder.py")
    with open(outpath, "w", encoding="utf-8") as fh:
        fh.write(script)
    print("\nwrote", outpath, "(%d rungs)" % len(ladder))


JOB_TEMPLATE = '''# ============================================================================
# Track B -- NISQ noise-scaling ladder on a real QPU (Braket Hybrid Job).
# Runs a ladder of increasing dose-instance sizes (6 -> 15 qubits), each a QAOA
# circuit pre-optimised on the simulator, on whichever device this job was
# submitted against. For every rung it reports how far the correct answer
# survives real hardware noise vs the noise-free simulator and a random baseline.
# Self-contained: only needs the Braket SDK + numpy (both in the PennyLane image).
# Entry point:  braket_noise_ladder:main
# ============================================================================
import os
from braket.circuits import Circuit
from braket.devices import LocalSimulator
from braket.aws import AwsDevice
from braket.jobs import save_job_result

SHOTS = 200   # per rung; real-QPU cost ~= $0.30/task + SHOTS*per-shot per rung

LADDER = __LADDER_DATA__


def build_circuit(r):
    nv, th, g, b = r["nv"], r["theta"], r["gamma"], r["beta"]
    c = Circuit()
    for v in range(nv):
        c.ry(v, th[v])
    for v in range(nv):
        if abs(r["Hising"][v]) > 1e-12:
            c.rz(v, 2.0 * g * r["Hising"][v])
    for a, bb, val in r["Jising"]:
        c.cnot(a, bb); c.rz(bb, 2.0 * g * val); c.cnot(a, bb)
    for v in range(nv):
        c.ry(v, -th[v]); c.rz(v, 2.0 * b); c.ry(v, th[v])
    return c


def decode(bitstr, r):
    n, D = r["n"], r["D"]
    assign = [-1] * n
    for i in range(n):
        ones = [d for d in range(D) if bitstr[i * D + d] == "1"]
        if len(ones) > 1:
            return None
        if ones:
            assign[i] = ones[0]
    return assign


def feasible(assign, r):
    sel = [i for i in range(r["n"]) if assign[i] >= 0]
    if not (r["min_size"] <= len(sel) <= r["k"]):
        return False
    if len({r["moa"][i] for i in sel}) < r["m_classes"]:
        return False
    return True


def score(assign, r):
    D = r["D"]
    Js = {(a, b): v for a, b, v in r["Jscore"]}
    sel = [i for i in range(r["n"]) if assign[i] >= 0]
    E = sum(r["h"][i][assign[i]] for i in sel)
    for x in range(len(sel)):
        for y in range(x + 1, len(sel)):
            i, j = sel[x], sel[y]
            a, bb = i * D + assign[i], j * D + assign[j]
            key = (min(a, bb), max(a, bb))
            E += Js.get(key, 0.0)
    T = sum(r["tox"][i][assign[i]] for i in sel)
    return E, T


def analyse(counts, shots, r, where):
    feas, best = 0, None
    opt_mass = 0
    ie, it, iv = r["ideal_best"]
    opt_bits = r["opt_bitstring"]
    for bitstr, cnt in counts.items():
        bitstr = "".join(map(str, bitstr)) if not isinstance(bitstr, str) else bitstr
        if bitstr == opt_bits:
            opt_mass += cnt
        a = decode(bitstr, r)
        if a is None or not feasible(a, r):
            continue
        feas += cnt
        e, t = score(a, r)
        v = e - r["lam"] * t
        if best is None or v > best[2]:
            best = (round(e, 3), round(t, 3), round(v, 3), cnt)
    row = {"rung": r["name"], "nv": r["nv"], "two_qubit_gates": r["n_twoqubit_gates"],
           "where": where, "shots": shots,
           "feasible_frac": round(feas / shots, 4),
           "random_feasible_frac": r["random_feasible_frac"],
           "best_value": (best[2] if best else None),
           "ideal_best_value": iv,
           "value_ratio": (round(best[2] / iv, 4) if best and iv else None),
           "found_true_optimum": (opt_mass > 0),
           "optimum_prob": round(opt_mass / shots, 4)}
    print("[%s] %-14s nv=%2d 2q=%3d | feasible %.1f%% (rand %.1f%%) | best/ideal=%s "
          "| found_opt=%s (%.1f%% mass)"
          % (where, r["name"], r["nv"], r["n_twoqubit_gates"], 100 * row["feasible_frac"],
             100 * row["random_feasible_frac"],
             ("%.2f" % row["value_ratio"]) if row["value_ratio"] else "--",
             row["found_true_optimum"], 100 * row["optimum_prob"]))
    return row


def main():
    device_arn = os.environ.get("AMZN_BRAKET_DEVICE_ARN")
    hw = bool(device_arn) and "local" not in device_arn
    device = AwsDevice(device_arn) if hw else None
    print("device:", device_arn if hw else "LOCAL SIM ONLY (no real device ARN)")

    results = []
    for r in LADDER:
        circ = build_circuit(r)
        ideal = LocalSimulator().run(circ, shots=500).result()
        row_ideal = analyse(ideal.measurement_counts, 500, r, "ideal-sim")
        row_hw = None
        if hw:
            task = device.run(circ, shots=SHOTS)
            print("  submitted %s task %s" % (r["name"], task.id))
            res = task.result()
            row_hw = analyse(res.measurement_counts, SHOTS, r, "REAL-QPU")
        results.append({"ideal": row_ideal, "hardware": row_hw})
        save_job_result({"ladder": results})   # checkpoint after every rung

    print("\\nNOISE-SCALING LADDER DONE")


if __name__ == "__main__":
    main()
'''


if __name__ == "__main__":
    main()
