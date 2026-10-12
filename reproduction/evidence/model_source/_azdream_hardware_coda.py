#!/usr/bin/env python3
"""
_azdream_hardware_coda.py -- Track B "here it is on real hardware" coda.

Runs a DELIBERATELY TINY dose-resolved instance so a real QPU can execute it:
3 drugs x D=2 = 6 qubits (penalty encoding), shallow (p=1) QAOA. The design
follows the project's hardware discipline (PROJECT_GUIDE.md §9, DATASET_LOG.md
§7): the variational parameters are optimised ONCE on the noise-free simulator,
then the FINAL circuit is executed -- on ideal sim, on a depolarising noise model
(local preview), and optionally on a real Braket QPU (IonQ) -- and we compare the
sampled feasible front / best energy. This is a NISQ noise-scaling demonstration,
NOT an optimisation run; real QPU time is spent only on the final circuit.

Real-hardware path (AWS Braket, via the Pawsey Quantum Hub / project pawsey1287):
  pip install amazon-braket-pennylane-plugin
  # surface federated AWS creds in the ENV (never pasted into code):
  #   aws configure sso   (or export AWS_ACCESS_KEY_ID / _SECRET / _SESSION_TOKEN)
  python src/_azdream_hardware_coda.py \
      --device braket --arn arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1 \
      --shots 500 --submit     # --submit REQUIRED to actually spend budget

Default (no --submit): ideal + noisy-sim preview only -- free, local, no AWS.

IonQ (trapped-ion, all-to-all) is the right target: our cost Hamiltonian is
fully connected, so all-to-all connectivity avoids SWAP-routing depth blow-up.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pennylane as qml
from pennylane import numpy as pnp

from dose_instance import DoseScorer
from dose_qaoa import build_dose_qubo, dose_prior, _decode
from qaoa import _qubo_to_ising
from warmstart import prior_to_angles

# 3-drug sub-instance: AKT_1(0), MTOR_1(4), PIK3CB_PIK3CD(5) -- distinct MoA,
# strong efficacy + real synergy. penalty encoding => 3*2 = 6 qubits.
CODA_DRUGS = [0, 4, 5]
LAM = 0.6          # a mid-front toxicity weight
P_LAYERS = 1       # shallow: hardware-friendly


def build_circuit_pieces(scorer, lam):
    nv = scorer.nv
    Q = build_dose_qubo(scorer, lam)
    hh, JJ, _ = _qubo_to_ising(Q)
    coeffs, ops = [], []
    for v in range(nv):
        if hh[v] != 0:
            coeffs.append(float(hh[v])); ops.append(qml.PauliZ(v))
    for v in range(nv):
        for w in range(v + 1, nv):
            if JJ[v, w] != 0:
                coeffs.append(float(JJ[v, w])); ops.append(qml.PauliZ(v) @ qml.PauliZ(w))
    H_cost = qml.Hamiltonian(coeffs, ops)
    theta = np.asarray(prior_to_angles(dose_prior(scorer, lam)), dtype=float)
    return nv, H_cost, theta


def make_circuit(nv, H_cost, theta, p_layers, depol=0.0):
    """QAOA ansatz closure. depol>0 inserts a DepolarizingChannel on every wire
    after state prep and after each cost/mix layer -- a simple, version-proof
    local noise model for the hardware-decay preview (run on default.mixed)."""
    def noise():
        if depol > 0:
            for w in range(nv):
                qml.DepolarizingChannel(depol, wires=w)

    def prepare():
        for w in range(nv):
            qml.RY(theta[w], wires=w)

    def mix(beta):
        for v in range(nv):
            qml.RY(-theta[v], wires=v); qml.RZ(2.0 * beta, wires=v); qml.RY(theta[v], wires=v)

    def ansatz(params):
        prepare(); noise()
        for layer in range(p_layers):
            qml.ApproxTimeEvolution(H_cost, params[layer, 0], 1); noise()
            mix(params[layer, 1]); noise()
    return ansatz


def optimise_on_sim(nv, H_cost, ansatz, p_layers, steps=60, seed=0):
    dev = qml.device("lightning.qubit", wires=nv)

    @qml.qnode(dev, interface="autograd", diff_method="adjoint")
    def cost(params):
        ansatz(params); return qml.expval(H_cost)

    rng = pnp.random.default_rng(seed)
    params = pnp.array(rng.standard_normal((p_layers, 2)) * 0.1, requires_grad=True)
    opt = qml.AdamOptimizer(0.1)
    for _ in range(steps):
        params = opt.step(cost, params)
    return np.asarray(params)


def sample_front(scorer, nv, ansatz, params, dev, shots, ideal_probs=False):
    """Return (best_feasible (assign,E,T), n_feasible_sampled) from the device."""
    if ideal_probs:
        @qml.qnode(dev)
        def probs(p):
            ansatz(p); return qml.probs(wires=range(nv))
        pr = np.asarray(probs(params))
        order = np.argsort(-pr)[:64]
        cands = [_decode(int(i), nv) for i in order]
    else:
        @qml.qnode(dev)
        def sampler(p):
            ansatz(p); return qml.sample(wires=range(nv))
        raw = np.asarray(sampler(params)).reshape(-1, nv)
        keys, counts = np.unique(raw, axis=0, return_counts=True)
        cands = [list(map(int, keys[i])) for i in np.argsort(-counts)[:64]]

    best, bestval, nfeas = None, -1e18, 0
    for bits in cands:
        a = scorer.onehot_to_assign(bits)
        if a is None or not scorer.feasible(a):
            continue
        nfeas += 1
        e, t = scorer.objectives(a)
        if e - LAM * t > bestval:
            bestval, best = e - LAM * t, (a, e, t)
    return best, nfeas


def main():
    ap = argparse.ArgumentParser(description="Track B hardware coda (6-qubit IonQ demo).")
    ap.add_argument("--device", default="sim", choices=["sim", "braket"])
    ap.add_argument("--arn", default="arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1")
    ap.add_argument("--shots", type=int, default=500)
    ap.add_argument("--submit", action="store_true",
                    help="REQUIRED to actually send a metered job to a real QPU")
    ap.add_argument("--noise", default="0.0,0.002,0.005,0.01",
                    help="per-gate depolarising rates for the local noisy preview")
    args = ap.parse_args()

    full = DoseScorer(D=2)
    s = full.subset(CODA_DRUGS)
    nv, H_cost, theta = build_circuit_pieces(s, LAM)
    ansatz = make_circuit(nv, H_cost, theta, P_LAYERS)

    print("TRACK B HARDWARE CODA")
    print("  drugs:", list(s.drugs), " | qubits (penalty encoding):", nv,
          " | layers:", P_LAYERS, " | lambda:", LAM)

    params = optimise_on_sim(nv, H_cost, ansatz, P_LAYERS)

    # circuit resource report (what noise scales with)
    dev0 = qml.device("lightning.qubit", wires=nv)
    try:
        spec = qml.specs(qml.QNode(lambda p: (ansatz(p), qml.expval(H_cost))[1], dev0))(params)
        res = spec["resources"] if isinstance(spec, dict) else spec.resources
        g2 = res.gate_sizes.get(2, 0); depth = res.depth; ng = res.num_gates
        print("  circuit: %s gates, depth %s, two-qubit gates %s (noise scales with these)"
              % (ng, depth, g2))
    except Exception as e:  # noqa: BLE001
        print("  circuit: resource introspection unavailable (%s)" % e)

    # exact reference front on this tiny instance
    exact = [(a, e, t) for a, e, t in s.all_feasible()]
    import pareto
    pts = [(e, t) for _, e, t in exact]
    ref = (min(e for e, _ in pts), max(t for _, t in pts))
    hv_exact = pareto.hypervolume(pts, ref)
    print("  exact: %d feasible, front HV=%.4f" % (len(exact), hv_exact))

    # ideal-simulator sampled best
    best_ideal, _ = sample_front(s, nv, ansatz, params,
                                 qml.device("lightning.qubit", wires=nv), 0, ideal_probs=True)
    print("\n  IDEAL sim  -> best feasible: %s  E=%.3f T=%.3f"
          % (s.label(best_ideal[0]), best_ideal[1], best_ideal[2]))

    # local NOISY preview: quality decay vs per-layer depolarising rate
    print("\n  NOISY preview (local depolarising model, %d shots):" % args.shots)
    print("    depol   feasible/%d shots   best feasible found" % args.shots)
    for p in [float(x) for x in args.noise.split(",")]:
        if p <= 0:
            continue
        dev = qml.device("default.mixed", wires=nv, shots=args.shots)
        noisy = make_circuit(nv, H_cost, theta, P_LAYERS, depol=p)
        best_n, nf = sample_front(s, nv, noisy, params, dev, args.shots)
        tag = ("%s  E=%.3f T=%.3f" % (s.label(best_n[0]), best_n[1], best_n[2])) if best_n else "-- none --"
        print("    %.3f       %-16d %s" % (p, nf, tag))

    # real hardware (only with --submit AND creds present)
    if args.device == "braket":
        if not args.submit:
            print("\n  [braket] dry run -- pass --submit to send a metered job to", args.arn)
            return
        try:
            dev = qml.device("braket.aws.qubit", device_arn=args.arn, wires=nv, shots=args.shots)
        except Exception as e:  # noqa: BLE001
            print("\n  [braket] device init failed (%s). Need amazon-braket-pennylane-plugin "
                  "+ AWS creds in env." % e)
            return
        print("\n  [braket] submitting FINAL circuit to", args.arn, "shots=%d ..." % args.shots)
        best_hw, nf = sample_front(s, nv, ansatz, params, dev, args.shots)
        tag = ("%s  E=%.3f T=%.3f" % (s.label(best_hw[0]), best_hw[1], best_hw[2])) if best_hw else "NO feasible sampled"
        print("  [braket] feasible states sampled: %d | best: %s" % (nf, tag))


if __name__ == "__main__":
    main()
