#!/usr/bin/env python3
"""
_azdream_build_instance.py -- build the Track B dose-resolved instance.

Builds the constrained, dose-resolved, multi-objective instance for the exact
7-drug fully-measured CAMA-1 clique found by src/_azdream_profile.py:

    clique #1 = AKT_1, BCL2_BCL2L1, FASN, MAP2K_1, MTOR_1, PIK3CB_PIK3CD, SLC16A4

from the AZ-DREAM OI synergy table (QA-passing rows only). All 21 pairs are
measured (verified), so there are NO MISSING J entries -- the cleanest instance.

------------------------------------------------------------------------------
MODEL (v1 -- self-contained from the OI table; flagged choices upgrade-able later)
------------------------------------------------------------------------------
Variables: x_{i,d} = drug i administered at dose level d (one-hot per drug).
Dose levels are NORMALISED fractions of each drug's max tested concentration
(c_d in (0,1]); this is unit-consistent with the OI table's IC50, which is
reported in max-conc-normalised units (MAX_CONC == 1 for every CAMA-1 row).

  h_{i,d}  (efficacy, single-drug inhibition fraction in [0,1]):
      Hill curve from the OI monotherapy fit (median IC50_i, H_i, Einf_i):
        v_i(c) = Einf_i + (100 - Einf_i) / (1 + (c / IC50_i)^{H_i})   [% viability]
        h_{i,d} = (100 - v_i(c_d)) / 100
      (H_i ~ 0, e.g. SLC16A4, -> flat, h ~ 0: monotherapy-inert but still synergises.)

  J_{(i,d),(j,d')}  (efficacy, pairwise synergy):
      measured Loewe SYNERGY_SCORE_ij (mean over reps) / 100, scaled by the
      DOSE-FRACTION PRODUCT  c_d * c_d'  so synergy grows with administered dose
      and does NOT vanish for monotherapy-inert drugs (SLC16A4). Sign preserved
      (antagonistic pairs stay negative).  *modeling choice; upgrade = real
      dose-response surfaces from DREAM_OI_matrices_final.zip.*

  T_{i,d}  (toxicity, the DECLARED PROXY -- second Pareto objective, minimise):
      T_{i,d} = w_class(MoA_i) * c_d
      dose-magnitude (defensible: we have real doses) x a per-MoA-class weight.
      *PROXY, not measured; flagged in Limitations. Never presented as measured.*

Efficacy and toxicity share the same dose lever c_d, so higher doses raise both
synergy-efficacy and toxicity -> a genuine trade-off / Pareto front.

Writes (mirroring the data/COLO320DM_* convention; variable index v = i*D + d):
  data/CAMA1c7_drugs.txt           7 drug names (clique order)
  data/CAMA1c7_moa.csv             drug, moa_class, tox_weight
  data/CAMA1c7_D{D}_doses.csv      the D normalised dose fractions
  data/CAMA1c7_D{D}_h.csv          n x D efficacy-linear  (h_{i,d})
  data/CAMA1c7_D{D}_tox.csv        n x D toxicity-linear  (T_{i,d})
  data/CAMA1c7_D{D}_Jij.csv        (n*D) x (n*D) symmetric efficacy-quadratic
  data/CAMA1c7_D{D}_meta.json      provenance + parameters

Run (laptop, stdlib + numpy):
  python src/_azdream_build_instance.py --D 2
  python src/_azdream_build_instance.py --D 3
"""

import argparse
import csv
import json
import os
import statistics as st

import numpy as np

DEFAULT_DATASET = r"C:\Users\Shrizon\Desktop\Pawsey\Dataset"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CLIQUE = ["AKT_1", "BCL2_BCL2L1", "FASN", "MAP2K_1", "MTOR_1",
          "PIK3CB_PIK3CD", "SLC16A4"]
CELL = "CAMA-1"

# Per-MoA-class toxicity weights (relative, literature-informed PROXY).
# Broad signaling inhibitors weighted higher; metabolic targets lower.
# These are illustrative relative weights, flagged as a proxy -- the ordering
# matters more than the exact values, and they are easy to revise.
TOX_WEIGHT = {
    "AKT": 1.0,
    "MTOR": 1.0,
    "PIK3CB_PIK3CD": 1.2,   # pan-PI3K, broad
    "MAP2K": 1.1,           # MEK, known clinical tox
    "BCL2_BCL2L1": 0.9,
    "FASN": 0.6,            # metabolic
    "SLC16A4": 0.5,         # metabolic transporter
}
DEFAULT_TOX_WEIGHT = 1.0


def base_class(name):
    """Coarse mechanism family: strip a trailing _<digit> replicate suffix."""
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return name


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def dose_levels(D):
    """Normalised dose fractions in (0,1], geometric (half-log) ladder ending at 1.0."""
    if D == 1:
        return [1.0]
    if D == 2:
        return [0.1, 1.0]
    if D == 3:
        return [0.1, 0.32, 1.0]
    # general: geometric from 0.1 to 1.0
    return [round(10 ** (-1.0 + i / (D - 1)), 4) for i in range(D)]


def hill_inhibition(c, ic50, hill, einf):
    """Single-drug inhibition fraction in [0,1] from a Hill fit (Einf in %)."""
    if ic50 is None or ic50 <= 0 or hill is None or einf is None:
        return 0.0
    ratio = (c / ic50) ** hill
    viability = einf + (100.0 - einf) / (1.0 + ratio)   # %
    return max(0.0, min(1.0, (100.0 - viability) / 100.0))


def load_oi_clique(dataset):
    """Return (per-drug Hill medians, per-pair mean synergy) for the clique on CAMA-1."""
    path = os.path.join(dataset, "OI_combinations_synergy_scores_final.txt")
    hill = {d: {"IC50": [], "H": [], "Einf": []} for d in CLIQUE}
    syn = {}
    clique = set(CLIQUE)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["CELL_LINE"].strip() != CELL or r["QA"].strip() != "1":
                continue
            a, b = r["COMPOUND_A"].strip(), r["COMPOUND_B"].strip()
            if a not in clique or b not in clique or a == b:
                continue
            for d, ic, h, ei in ((a, "IC50_A", "H_A", "Einf_A"),
                                 (b, "IC50_B", "H_B", "Einf_B")):
                for key, col in (("IC50", ic), ("H", h), ("Einf", ei)):
                    v = f(r[col])
                    if v is not None:
                        hill[d][key].append(v)
            sc = f(r["SYNERGY_SCORE"])
            if sc is not None:
                syn.setdefault(frozenset((a, b)), []).append(sc)

    hill_med = {}
    for d in CLIQUE:
        H = hill[d]
        hill_med[d] = {k: (st.median(v) if v else None) for k, v in H.items()}
    syn_mean = {k: st.mean(v) for k, v in syn.items()}
    return hill_med, syn_mean


def load_moa(dataset):
    """Map clique drug -> mechanism class from Drug_info_release.csv (base_class)."""
    path = os.path.join(dataset, "Drug_info_release.csv")
    name_target = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            nm = (r.get("ChallengeName") or "").strip()
            if nm:
                name_target[nm] = (r.get("Target(Official Symbol)") or "").strip()
    return {d: base_class(d) for d in CLIQUE}


def build(D, dataset, out_dir):
    hill_med, syn_mean = load_oi_clique(dataset)
    moa = load_moa(dataset)
    doses = dose_levels(D)
    n = len(CLIQUE)

    # h_{i,d} and T_{i,d}
    h = np.zeros((n, D))
    tox = np.zeros((n, D))
    for i, drug in enumerate(CLIQUE):
        p = hill_med[drug]
        w = TOX_WEIGHT.get(moa[drug], DEFAULT_TOX_WEIGHT)
        for d, c in enumerate(doses):
            h[i, d] = hill_inhibition(c, p["IC50"], p["H"], p["Einf"])
            tox[i, d] = w * c

    # J_{(i,d),(j,d')} = (SYN_ij/100) * c_d * c_d', variable index v = i*D + d
    nv = n * D
    J = np.zeros((nv, nv))
    for i in range(n):
        for j in range(i + 1, n):
            s = syn_mean.get(frozenset((CLIQUE[i], CLIQUE[j])), 0.0) / 100.0
            for di, ci in enumerate(doses):
                for dj, cj in enumerate(doses):
                    val = s * ci * cj
                    vi, vj = i * D + di, j * D + dj
                    J[vi, vj] = val
                    J[vj, vi] = val

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "CAMA1c7")

    with open(base + "_drugs.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(CLIQUE) + "\n")
    with open(base + "_moa.csv", "w", encoding="utf-8", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["index", "drug", "moa_class", "tox_weight"])
        for i, drug in enumerate(CLIQUE):
            wtr.writerow([i, drug, moa[drug],
                          TOX_WEIGHT.get(moa[drug], DEFAULT_TOX_WEIGHT)])

    np.savetxt(base + "_D%d_doses.csv" % D, np.array(doses)[None, :],
               delimiter=",", fmt="%.4f")
    np.savetxt(base + "_D%d_h.csv" % D, h, delimiter=",", fmt="%.6f")
    np.savetxt(base + "_D%d_tox.csv" % D, tox, delimiter=",", fmt="%.6f")
    np.savetxt(base + "_D%d_Jij.csv" % D, J, delimiter=",", fmt="%.6f")

    meta = {
        "cell": CELL, "clique": CLIQUE, "n_drugs": n, "D": D,
        "doses_normalised": doses, "qubits_one_hot": nv,
        "moa_class": {d: moa[d] for d in CLIQUE},
        "tox_weight": {d: TOX_WEIGHT.get(moa[d], DEFAULT_TOX_WEIGHT) for d in CLIQUE},
        "synergy_score_mean": {("|".join(sorted(k))): round(v, 3)
                               for k, v in syn_mean.items()},
        "hill_median": {d: {k: (round(v, 4) if v is not None else None)
                            for k, v in hill_med[d].items()} for d in CLIQUE},
        "model_notes": {
            "efficacy_linear": "h_{i,d} = 1 - Hill viability(c_d); IC50/Einf in max-conc-normalised units",
            "efficacy_quad": "J = (SYNERGY_SCORE_ij/100) * c_d * c_d' (dose-fraction product; sign preserved)",
            "toxicity": "PROXY: T_{i,d} = w_class(MoA) * c_d (dose-magnitude x per-MoA weight)",
            "source": "AZ-DREAM OI table, CAMA-1, QA=1 only; all 21 clique pairs measured",
        },
    }
    with open(base + "_D%d_meta.json" % D, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    return h, tox, J, doses, moa, syn_mean


def main():
    ap = argparse.ArgumentParser(description="Build the Track B dose-resolved instance.")
    ap.add_argument("--D", type=int, default=2, help="dose levels per drug (2 or 3)")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--out", default=DATA_DIR)
    args = ap.parse_args()

    h, tox, J, doses, moa, syn = build(args.D, args.dataset, args.out)
    n, D = h.shape

    print("Built CAMA-1 7-drug instance, D=%d  (%d one-hot qubits)" % (D, n * D))
    print("doses (normalised fractions):", doses)
    print("\n%-16s %-16s %s" % ("drug", "moa_class", "  h_{i,d}  (efficacy linear)"))
    for i, drug in enumerate(CLIQUE):
        print("%-16s %-16s %s" % (drug, moa[drug],
              "  ".join("%.3f" % h[i, d] for d in range(D))))
    print("\n%-16s %-16s %s" % ("drug", "tox_weight", "  T_{i,d}  (toxicity linear)"))
    for i, drug in enumerate(CLIQUE):
        w = TOX_WEIGHT.get(moa[drug], DEFAULT_TOX_WEIGHT)
        print("%-16s %-16.1f %s" % (drug, w,
              "  ".join("%.3f" % tox[i, d] for d in range(D))))
    pos = int((J > 1e-9).sum() // 2)
    neg = int((J < -1e-9).sum() // 2)
    print("\nSynergy pairs: %d synergistic (J>0), %d antagonistic (J<0) at full dose"
          % (pos // (D * D), neg // (D * D)))
    print("J range: [%.4f, %.4f]" % (J.min(), J.max()))
    print("\nWrote data/CAMA1c7_D%d_*  +  CAMA1c7_drugs.txt / _moa.csv" % D)


if __name__ == "__main__":
    main()
