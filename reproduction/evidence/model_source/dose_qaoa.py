"""Custom warm-start QAOA on the one-hot drug x dose QUBO (Track B).

Recovers the efficacy-toxicity Pareto front by SCALARISATION: for a grid of
weights lambda, build the QUBO for  maximise (E - lambda*T)  subject to the
constraints, run warm-start QAOA, take the returned feasible point. The
non-dominated set of those points is the QAOA front.

Reuses the project's existing machinery:
  - qaoa._qubo_to_ising  (QUBO -> Ising, unchanged)
  - warmstart.prior_to_angles  (prior c -> RY angles)
  - the Egger/Tate warm-start mixer prescription  RY(-theta).RZ(2beta).RY(theta)
    (identical to qaoa.py:_mix, generalised to n*D one-hot variables)

What is NEW vs the binary Track-A pipeline:
  - one-hot drug x dose variables (v = i*D + d) and a one-dose-per-drug penalty;
  - re-ranking that decodes one-hot bitstrings to dose assignments and scores
    BOTH objectives via DoseScorer.

Constraints encoded as QUBO penalties: one-dose-per-drug + cardinality (2..k).
MoA-diversity is NOT penalised here because clique #1 has one drug per MoA
class, so any feasible-size combo already spans >= 2 classes (the re-rank still
checks it via DoseScorer.feasible). For instances where drugs share a class a
MoA penalty would be added -- that is Paper-2 work.

NOTE (rigor): weighted-sum scalarisation recovers only the convex hull of the
front; epsilon-constraint / Chebyshev scalarisation (Paper-2) reaches concave
regions. Flagged in the write-up.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from dose_instance import Assign, DoseScorer
from qaoa import _qubo_to_ising
from warmstart import prior_to_angles

_DEVICE_NAME = None  # cached after first successful resolution


def _device(nv: int, shots: Optional[int] = None):
    """Resolve a PennyLane device.

    Order of preference (auto): lightning.gpu (Setonix GH200) -> lightning.qubit
    (CPU) -> default.qubit. Set env DRUGSEARCH_DEVICE to force one (e.g.
    'lightning.gpu' in an sbatch job so it fails loudly if the GPU stack is
    missing, rather than silently dropping to CPU). The resolved name is cached
    so the laptop doesn't re-probe lightning.gpu on every call.

    shots=None -> analytic (exact probabilities), used for training and for the
    small penalty tier. shots=<int> -> a finite-shot sampler, used when nv is
    large enough that a 2**nv probability vector is impractical (the 30-qubit XY
    tier).

    At the 30-qubit ceiling the lightning devices are switched to single precision
    (c_dtype=complex64): a complex128 statevector is 17 GB and adjoint-diff needs
    ~2.5x that, which OOMs a 96 GB GH200; complex64 halves every buffer for
    comfortable headroom, and the precision is irrelevant here because the front
    is re-ranked EXACTLY from the sampled bitstrings."""
    global _DEVICE_NAME

    def _mk(name):
        kw = {"wires": nv, "shots": shots}
        if name.startswith("lightning") and nv >= 28:
            kw["c_dtype"] = np.complex64
        return qml.device(name, **kw)

    forced = os.environ.get("DRUGSEARCH_DEVICE")
    if forced:
        return _mk(forced)
    if _DEVICE_NAME is not None:
        return _mk(_DEVICE_NAME)
    for name in ("lightning.gpu", "lightning.qubit", "default.qubit"):
        try:
            dev = _mk(name)
            _DEVICE_NAME = name
            return dev
        except Exception:
            continue
    raise RuntimeError("no PennyLane device could be created")


def active_device_name():
    """Best-effort name of the device that will be used (after one probe)."""
    forced = os.environ.get("DRUGSEARCH_DEVICE")
    if forced:
        return forced
    if _DEVICE_NAME is None:
        try:
            _device(1)
        except Exception:
            return "unknown"
    return _DEVICE_NAME


def _add_leq_sq_penalty(Q: np.ndarray, weights, cap: float, pen: float) -> np.ndarray:
    """Add  pen * (sum_v weights[v]*x_v - cap)^2  to QUBO Q, in place.

    A slack-FREE penalty encoding of the inequality  sum_v weights[v]*x_v <= cap
    (cf. unbalanced penalisation, Montanez-Barrera et al. 2022): quadratic in the
    binary x (x^2 = x), so it folds into Q's diagonal + off-diagonal with no extra
    qubits. The constant  pen*cap^2  is dropped (irrelevant to the argmin). This
    only GUIDES the sampler toward the feasible slice -- exact epsilon-feasibility
    is still enforced by the re-ranking filter, so the reported front is never
    distorted by the softness of this term.
    """
    u = np.asarray(weights, dtype=float).reshape(-1)
    nv = Q.shape[0]
    for v in range(nv):
        if u[v] != 0.0:
            Q[v, v] += pen * (u[v] * u[v] - 2.0 * cap * u[v])
    for v in range(nv):
        if u[v] == 0.0:
            continue
        for w in range(v + 1, nv):
            if u[w] != 0.0:
                Q[v, w] += pen * u[v] * u[w]
                Q[w, v] += pen * u[v] * u[w]
    return Q


def build_dose_qubo(scorer: DoseScorer, lam: float,
                    pen_onehot: Optional[float] = None,
                    pen_card: Optional[float] = None,
                    tox_cap: Optional[float] = None,
                    pen_eps: Optional[float] = None) -> np.ndarray:
    """QUBO Q with  x^T Q x = -(E - lam*T) + penalties, x in {0,1}^{n*D}.

    Linear:    c_v = h_flat[v] - lam * tox_flat[v]      (variable v = i*D + d)
    Quadratic: J[v,w]                                   (efficacy synergy)
    Penalties: one-dose-per-drug (forbid two doses of one drug) + cardinality.

    epsilon-constraint (opt-in): if tox_cap is given, add a soft  T <= tox_cap
    penalty so the sampler is steered into that toxicity slice. Call with lam=0
    to get the pure epsilon-constraint sub-problem  'maximise E s.t. T <= eps'.
    tox_cap=None reproduces the Stage-1 QUBO byte-for-byte.
    """
    n, D, nv = scorer.n, scorer.D, scorer.nv
    h_flat = scorer.h.reshape(-1)
    tox_flat = scorer.tox.reshape(-1)
    J = scorer.J
    c = h_flat - lam * tox_flat

    if pen_card is None or pen_onehot is None:
        scale = float(np.abs(c).max() + np.abs(J).sum(axis=1).max() + 1e-6)
        P = 3.0 * scale
        pen_card = P if pen_card is None else pen_card
        pen_onehot = P if pen_onehot is None else pen_onehot

    Q = np.zeros((nv, nv))
    # Objective (minimise -f): linear on diagonal, synergy split across off-diag.
    for v in range(nv):
        Q[v, v] -= c[v]
    for v in range(nv):
        for w in range(v + 1, nv):
            if J[v, w] != 0.0:
                Q[v, w] += -J[v, w] / 2.0
                Q[w, v] += -J[v, w] / 2.0

    # Cardinality penalty pen_card*(S-2)(S-3), S = sum_v x_v:
    #   -4*pen_card on each diagonal, +pen_card on each symmetric off-diagonal.
    for v in range(nv):
        Q[v, v] += pen_card * (-4.0)
    for v in range(nv):
        for w in range(v + 1, nv):
            Q[v, w] += pen_card
            Q[w, v] += pen_card

    # One-dose-per-drug: penalise picking two doses of the SAME drug.
    for i in range(n):
        for d1 in range(D):
            for d2 in range(d1 + 1, D):
                v, w = i * D + d1, i * D + d2
                Q[v, w] += pen_onehot
                Q[w, v] += pen_onehot

    # epsilon-constraint (opt-in): soft  T <= tox_cap  guide, no extra qubits.
    if tox_cap is not None:
        if pen_eps is None:
            scale = float(np.abs(c).max() + np.abs(J).sum(axis=1).max() + 1e-6)
            pen_eps = 1.0 * scale
        _add_leq_sq_penalty(Q, tox_flat, tox_cap, pen_eps)
    return Q


def dose_prior(scorer: DoseScorer, lam: float, target: float = 2.5,
               tau: float = 0.5) -> np.ndarray:
    """Warm-start prior c_v in [0,1]: bias toward variables with high E-lam*T,
    rescaled so the expected number of 'on' variables ~ target (the cardinality
    target). Mirrors the spirit of warmstart.build_*_prior."""
    h_flat = scorer.h.reshape(-1)
    tox_flat = scorer.tox.reshape(-1)
    g = h_flat - lam * tox_flat
    if g.std() > 1e-9:
        g = (g - g.mean()) / g.std()
    u = 1.0 / (1.0 + np.exp(-g / tau))
    s = u.sum()
    if s > 0:
        u = np.clip(u * (target / s), 1e-3, 1 - 1e-3)
    return u


def _decode(index: int, nv: int) -> List[int]:
    """PennyLane probs index -> bit list indexed by variable v (wire 0 = MSB)."""
    return [(index >> (nv - 1 - v)) & 1 for v in range(nv)]


def _sample_candidates(dev_nv, p_layers, H_cost, prepare, mix, params, top_k,
                       shots=None):
    """Top-k candidate bit-lists from the optimised QAOA state.

    shots=None: exact top-k of the 2**nv probability vector (small tiers).
    shots=<int>: the most-frequent finite-shot samples -- bounded memory, so the
    30-qubit XY tier never materialises a billion-entry probability vector. Both
    arms (penalty, XY) share this: they pass their own prepare/mix closures."""
    if shots:
        dev = _device(dev_nv, shots=shots)

        @qml.qnode(dev, diff_method=None)
        def sample_q(p):
            prepare()
            for layer in range(p_layers):
                qml.ApproxTimeEvolution(H_cost, p[layer, 0], 1)
                mix(p[layer, 1])
            return qml.sample(wires=range(dev_nv))

        raw = np.asarray(sample_q(params)).reshape(-1, dev_nv)
        keys, counts = np.unique(raw, axis=0, return_counts=True)
        order = np.argsort(-counts)[:top_k]
        return [list(map(int, keys[o])) for o in order]

    dev = _device(dev_nv)

    @qml.qnode(dev, diff_method=None)
    def probs_q(p):
        prepare()
        for layer in range(p_layers):
            qml.ApproxTimeEvolution(H_cost, p[layer, 0], 1)
            mix(p[layer, 1])
        return qml.probs(wires=range(dev_nv))

    probs = np.asarray(probs_q(params))
    top = np.argsort(-probs)[:top_k]
    return [_decode(int(idx), dev_nv) for idx in top]


def dose_qaoa_point(scorer: DoseScorer, lam: float, p_layers: int = 2,
                    steps: int = 50, lr: float = 0.1, seed: int = 0,
                    warmstart: bool = True, warm_mixer: bool = True,
                    top_k: int = 96, tox_cap: Optional[float] = None,
                    shots: Optional[int] = None):
    """Run warm-start QAOA for one scalarisation weight lam. Returns
    (best, feasible) where best = (assign, E, T, scalarised_value) is the top
    feasible string by the scalarised objective (or None), and feasible is the
    list of ALL distinct feasible (assign, E, T) found in the top-k bitstrings
    (the sampler's feasible support at this lambda).

    tox_cap (opt-in): steer the sampler toward the  T <= tox_cap  slice for
    epsilon-constraint search; combine with lam=0 for pure 'maximise E s.t.
    T <= eps'. None -> Stage-1 weighted-sum behaviour, unchanged.

    shots (opt-in): finite-shot readout of the optimised state instead of exact
    probabilities -- required once nv is large enough that a 2**nv vector is
    impractical (30-qubit XY tier). None -> exact top-k probabilities, unchanged."""
    nv = scorer.nv
    Q = build_dose_qubo(scorer, lam, tox_cap=tox_cap)
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

    if warmstart:
        theta = np.asarray(prior_to_angles(dose_prior(scorer, lam)), dtype=float)
    else:
        theta = None
    wm = bool(warm_mixer) and theta is not None

    def _prepare():
        if theta is None:
            for w in range(nv):
                qml.Hadamard(wires=w)
        else:
            for w in range(nv):
                qml.RY(theta[w], wires=w)

    def _mix(beta):
        if wm:
            for v in range(nv):
                qml.RY(-theta[v], wires=v)
                qml.RZ(2.0 * beta, wires=v)
                qml.RY(theta[v], wires=v)
        else:
            for v in range(nv):
                qml.RX(2.0 * beta, wires=v)

    dev = _device(nv)

    @qml.qnode(dev, interface="autograd", diff_method="adjoint")
    def expectation(params):
        _prepare()
        for layer in range(p_layers):
            qml.ApproxTimeEvolution(H_cost, params[layer, 0], 1)
            _mix(params[layer, 1])
        return qml.expval(H_cost)

    rng = pnp.random.default_rng(seed)
    params = pnp.array(rng.standard_normal((p_layers, 2)) * 0.1, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=lr)
    for _ in range(steps):
        params = opt.step(expectation, params)

    cands = _sample_candidates(dev_nv=nv, p_layers=p_layers, H_cost=H_cost,
                               prepare=_prepare, mix=_mix, params=params,
                               top_k=top_k, shots=shots)

    best = None
    best_val = -float("inf")
    feasible = []
    seen = set()
    for bits in cands:
        assign = scorer.onehot_to_assign(bits)
        if assign is None or not scorer.feasible(assign):
            continue
        e, t = scorer.objectives(assign)
        if assign not in seen:
            seen.add(assign)
            feasible.append((assign, e, t))
        val = e - lam * t
        if val > best_val:
            best_val, best = val, (assign, e, t, val)
    return best, feasible


def qaoa_pareto(scorer: DoseScorer, lambdas=None, p_layers: int = 2,
                steps: int = 50, seed: int = 0, warmstart: bool = True,
                warm_mixer: bool = True, verbose: bool = False):
    """Scalarisation sweep -> QAOA-recovered Pareto front.

    Pools ALL feasible states the sampler surfaces (top-k of every lambda), then
    returns their non-dominated set. This reflects what QAOA actually produces -- a
    distribution -- rather than one argmax per weight, and so populates the concave
    middle of the front that a pure weighted-sum argmax skips.
    """
    import pareto
    if lambdas is None:
        lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.15,
                   1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]
    pool = {}
    for lam in lambdas:
        best, feas = dose_qaoa_point(scorer, lam, p_layers=p_layers, steps=steps,
                                     seed=seed, warmstart=warmstart,
                                     warm_mixer=warm_mixer)
        for a, e, t in feas:
            pool[a] = (e, t)
        if verbose:
            tag = ("E=%.3f T=%.3f %s" % (best[1], best[2], scorer.label(best[0]))
                   if best else "no feasible")
            print("  lam=%.2f -> argmax %s | pooled feasible: %d" % (lam, tag, len(pool)))
    rows = [(a, e, t) for a, (e, t) in pool.items()]
    pts = [(e, t) for _, e, t in rows]
    return [rows[i] for i in pareto.pareto_sorted(pts)]
