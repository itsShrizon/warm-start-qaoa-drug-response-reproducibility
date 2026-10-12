"""Warm-start priors for QAOA on the drug-combination QUBO.

A *prior* here is a vector c in [0,1]^n giving, per drug, the probability that
the drug should be ON in the chosen combination. qaoa_search(warmstart=c) turns
each c_i into an RY rotation RY(theta_i)|0> with P(qubit i = 1) = c_i, and pairs
it with a custom mixer whose ground state is exactly that product state
(Egger 2021 / Tate 2023 -- the pairing is what makes warm-starting help).

Three constructors produce c:
  build_moa_prior      -- mechanism-of-action (MoA) diversity prior (the novel
                          contribution): rewards potent drugs that synergise
                          ACROSS MoA classes and penalises within-class
                          redundancy. Uses MoA-class structure the QUBO scalars
                          do not contain.
  build_generic_prior  -- structure-blind continuous relaxation of the QUBO
                          (Egger 2021 baseline; the bar the MoA prior must beat).
  build_scramble_prior -- the MoA prior with MoA labels permuted: same shape and
                          strength, biology destroyed. The causal control.

prior_to_angles converts any c into the per-qubit RY angles consumed by
qaoa_search. It lives here (not in qaoa.py) so qaoa.py depends on warmstart.py
one-directionally; build_generic_prior takes Q as an argument and therefore does
NOT import qaoa, so there is no import cycle.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, Optional, Sequence

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# --------------------------------------------------------------------------- #
# Angle conversion (consumed by qaoa_search)                                  #
# --------------------------------------------------------------------------- #
def prior_to_angles(c: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Per-qubit RY angle theta_i = 2*arcsin(sqrt(clip(c_i, eps, 1-eps))).

    RY(theta_i)|0> = sqrt(1-c_i)|0> + sqrt(c_i)|1>, so P(qubit i = 1) = c_i.
    Clipping keeps the angle in (0, pi) so the warm-start mixer's X-component
    never vanishes (a qubit at c_i in {0,1} would otherwise freeze).

    Note the consistency property used as the cold-QAOA regression test: a
    uniform c = 0.5 gives theta = pi/2, i.e. RY(pi/2)|0> = |+> (= Hadamard) and
    a warm mixer that reduces to the transverse-field mixer.
    """
    c = np.clip(np.asarray(c, dtype=float), eps, 1.0 - eps)
    return 2.0 * np.arcsin(np.sqrt(c))


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def _zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / (v.std() + 1e-9)


def _rescale_to_target(u: np.ndarray, target: float, eps: float,
                       passes: int = 2) -> np.ndarray:
    """Scale u so sum(c) ~= target, clipping to [eps, 1-eps].

    Clipping perturbs the sum, so we alternate scale/clip a couple of times; at
    n ~ 20 this lands sum(c) within ~0.05 of target. The prior stays
    interpretable as a soft cardinality-`target` selection.
    """
    c = np.asarray(u, dtype=float).copy()
    for _ in range(passes):
        s = c.sum()
        if s > 0:
            c = c * (target / s)
        c = np.clip(c, eps, 1.0 - eps)
    return c


# --------------------------------------------------------------------------- #
# MoA annotation loading                                                      #
# --------------------------------------------------------------------------- #
def load_moa(cell: str = "COLO320DM",
             drugs: Optional[Sequence[str]] = None,
             data_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load data/<cell>_moa.csv (built by src/_build_moa_annotation.py).

    The CSV has columns index,drug,target,moa_class,pubchem_cid,source and MUST
    be row-aligned to <cell>_drugs.txt (same as the h/J files). When `drugs` is
    given (the names from DataScorer) the alignment is asserted, mirroring the
    index-aligned _load in data_instance.py.
    Returns a dict with 'moa_class', 'target', 'pubchem_cid' as arrays of len n.
    """
    base = os.path.join(data_dir or DATA_DIR, cell.replace("/", "_"))
    rows = []
    with open(base + "_moa.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: int(r["index"]))
    names = [r["drug"] for r in rows]
    if drugs is not None:
        assert list(drugs) == names, (
            "MoA CSV drug order != drugs.txt order:\n"
            f"  moa : {names}\n  txt : {list(drugs)}"
        )
    return {
        "drug": np.array(names, dtype=object),
        "target": np.array([r.get("target", "") for r in rows], dtype=object),
        "moa_class": np.array([r["moa_class"] for r in rows], dtype=object),
        "pubchem_cid": np.array([r.get("pubchem_cid", "") for r in rows], dtype=object),
    }


# --------------------------------------------------------------------------- #
# Prior constructors                                                          #
# --------------------------------------------------------------------------- #
def build_moa_prior(
    h: np.ndarray,
    J: np.ndarray,
    moa_class: Sequence,
    w_p: float = 1.0,
    w_x: float = 1.0,
    w_r: float = 0.0,
    tau: float = 1.0,
    target: float = 2.5,
    eps: float = 1e-3,
) -> np.ndarray:
    """MoA-diversity-informed warm-start prior (the novel contribution).

    g_i = w_p*z(p_i) + w_x*z(x_i) - w_r*z(r_i), squashed by sigmoid(./tau) and
    rescaled so sum(c) ~= target, where
        p_i = h_i                                   monotherapy potency
        x_i = mean{ J_ij>0 : moa(j) != moa(i) }     cross-MoA synergy potential
        r_i = mean{ J_ij>0 : moa(j) == moa(i) }     within-MoA "redundancy"
    MoA-class structure enters via the cross/within split in x_i (and r_i): the
    information the QUBO scalars (h, J magnitudes) do not separate out.

    w_r defaults to 0. On COLO320DM the strongest measured synergy is within a
    single MoA class (BEZ-235 + MK-8669, both PI3K_AKT_mTOR; z(r)=3.2 for
    BEZ-235), so a within-class "redundancy" penalty wrongly demotes a genuine
    vertical-pathway synergy. We keep w_r as a studyable knob but leave it off:
    the prior's principle is "potent drug that synergises ACROSS mechanisms".
    """
    h = np.asarray(h, dtype=float)
    J = np.asarray(J, dtype=float)
    moa = np.asarray(moa_class, dtype=object)
    n = len(h)

    p = h.copy()
    x = np.zeros(n)
    r = np.zeros(n)
    for i in range(n):
        same_mask = np.array([j != i and moa[j] == moa[i] for j in range(n)])
        other_mask = np.array([j != i and moa[j] != moa[i] for j in range(n)])
        cross = J[i][other_mask]
        cross = cross[cross > 0]
        within = J[i][same_mask]
        within = within[within > 0]
        x[i] = cross.mean() if cross.size else 0.0
        r[i] = within.mean() if within.size else 0.0

    g = w_p * _zscore(p) + w_x * _zscore(x) - w_r * _zscore(r)
    u = 1.0 / (1.0 + np.exp(-g / tau))
    return _rescale_to_target(u, target, eps)


def build_generic_prior(
    Q: np.ndarray,
    iters: int = 400,
    eps: float = 1e-3,
    target: Optional[float] = None,
) -> np.ndarray:
    """Structure-blind warm-start: continuous relaxation of the QUBO.

    Solve  min_{c in [0,1]^n}  c^T Q c  by projected gradient descent
    (grad = (Q + Q^T) c). Pure numpy -- scipy is intentionally NOT a project
    dependency. This is the published Egger-2021 baseline the MoA prior must
    beat; it sees only Q (the h/J scalars), never the MoA partition.

    If `target` is given, the relaxed solution is rescaled toward that
    cardinality so it is comparable in sparsity to the MoA prior; by default the
    raw relaxed optimum is returned (clipped to [eps, 1-eps]).
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    G = Q + Q.T
    eta = 1.0 / (np.linalg.norm(G, 2) + 1e-12)
    c = 0.5 * np.ones(n)
    for _ in range(iters):
        c = np.clip(c - eta * (G @ c), 0.0, 1.0)
    if target is not None:
        return _rescale_to_target(c, target, eps)
    return np.clip(c, eps, 1.0 - eps)


def build_scramble_prior(
    h: np.ndarray,
    J: np.ndarray,
    moa_class: Sequence,
    scramble_seed: int = 12345,
    **prior_kwargs,
) -> np.ndarray:
    """Causal control: the MoA prior with MoA labels permuted.

    Preserves the prior's overall shape, strength and sum (potency still drives
    p_i) but destroys the biological cross/within-MoA partition. If the MoA
    prior beats THIS, the gain comes from correct biology, not merely from
    having a non-uniform potency-weighted prior.
    """
    rng = np.random.default_rng(scramble_seed)
    scrambled = np.asarray(moa_class, dtype=object)[rng.permutation(len(moa_class))]
    return build_moa_prior(h, J, scrambled, **prior_kwargs)
