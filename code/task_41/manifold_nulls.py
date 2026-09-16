#!/usr/bin/env python3
# ==============================================================================
# AI ASSISTANCE DISCLAIMER
# In accordance with academic guidelines, the author acknowledges the use of
# generative artificial intelligence assistance (DeepMind's Antigravity AI assistant)
# for writing assistance, code development, data formatting, and proofreading the
# manuscript. Some analytical derivations, mathematical proofs, numerical simulations,
# and physical interpretations still need rigorous verification by the author.
# ==============================================================================
"""
Null models for the constructed manifold.

A clustering statistic is only informative against a null that keeps
everything except the ingredient under test.  Three are used:

  label   permute the species labels of J jointly on rows and columns.  J and
          n keep their exact structure; what is destroyed is the pairing
          between a type's interaction profile and its abundance / genotype.
  shuffle permute the off-diagonal entries of J.  Keeps the value distribution,
          destroys the c[a XOR b] * g[b] product structure.
  none    J = 0.  Keeps abundances, resource competition and the mutation
          kernel; removes interspecies interactions entirely.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, '.')
from tana_geometry import (community_jacobian, whitening_weights,
                           propagator_distance, hopkins, silhouette_profile)

DATA = os.environ.get("TANA_DATA", "data")
OUT = os.environ.get("TANA_OUT", "results")
PKILL, NU, L, WINDOW, NDRAW, SEED = 0.2, 5e-6, 20, (1, 3), 40, 20260912

GRID = [("baseline_std_tnm","R02_baseline",100,.10,.010),("C_50","R01_C50",50,.10,.010),
        ("C_200","R03_C200",200,.10,.010),("C_300","R04_C300",300,.10,.010),
        ("pmut_0005","R06_pmut0005",100,.10,.005),("pmut_002","R07_pmut002",100,.10,.020),
        ("mu_005","R08_mu005",100,.05,.010),("mu_02","R09_mu02",100,.20,.010),
        ("pmut_002_C_200","R10_pmut002_C200",200,.10,.020),
        ("C_200_pmut_0005","R11_C200_pmut0005",200,.10,.005),
        ("C_200_mu_005","R12_C200_mu005",200,.05,.010)]

def geom(J, n, sa, mu, pmut):
    N = n.sum()
    K, _ = community_jacobian(J, n, mu=mu, nu=NU, pkill=PKILL, pmut=pmut,
                              sa=sa, L=L, with_mutation=True)
    dn = np.clip(np.round(np.sqrt(n)), 1, n)
    D = propagator_distance(K, np.diag(-dn / N), whitening_weights(n), *WINDOW)
    s = silhouette_profile(D, ks=(2, 3, 4))
    return hopkins(D, seed=SEED), max(s.values())

def surrogate(J, rng, kind):
    S = len(J)
    if kind == "none":
        return np.zeros_like(J)
    if kind == "label":
        p = rng.permutation(S)
        return J[np.ix_(p, p)]
    off = ~np.eye(S, dtype=bool)
    Js = np.zeros_like(J); v = J[off].copy(); rng.shuffle(v); Js[off] = v
    return Js

out = {}
for tag, fid, C, mu, pmut in GRID:
    p = f"{DATA}/{fid}_tstar.json"
    if not os.path.exists(p): continue
    d = json.load(open(p))
    J = np.array(d["J_sub"], float)
    n = np.array([s["pop"] for s in d["species"]], float)
    sa = np.array([s["id"] for s in d["species"]], int)
    if len(n) < 10: continue
    H, s = geom(J, n, sa, mu, pmut)
    row = {"hopkins": H, "silhouette_max": s, "S": int(len(n)), "nulls": {}}
    rng = np.random.default_rng(SEED)
    for kind in ("label", "shuffle", "none"):
        hs, ss = [], []
        reps = 1 if kind == "none" else NDRAW
        for _ in range(reps):
            h2, s2 = geom(surrogate(J, rng, kind), n, sa, mu, pmut)
            hs.append(h2); ss.append(s2)
        row["nulls"][kind] = {
            "hopkins_mean": float(np.nanmean(hs)), "hopkins_sd": float(np.nanstd(hs)),
            "silhouette_mean": float(np.mean(ss)), "silhouette_sd": float(np.std(ss)),
            "hopkins_z": float((H - np.nanmean(hs)) / np.nanstd(hs)) if reps > 1 and np.nanstd(hs) > 0 else None,
            "silhouette_z": float((s - np.mean(ss)) / np.std(ss)) if reps > 1 and np.std(ss) > 0 else None}
    out[tag] = row
    lz = row["nulls"]["label"]
    print(f"{tag:18s} S={len(n):3d}  H={H:.3f} (label null {lz['hopkins_mean']:.3f}+-{lz['hopkins_sd']:.3f}, "
          f"z={lz['hopkins_z']:+.2f})   s_max={s:.3f} (null {lz['silhouette_mean']:.3f}, z={lz['silhouette_z']:+.2f})")

json.dump(out, open(f"{OUT}/manifold_nulls.json", "w"), indent=2, default=float)
print(f"\nwrote {OUT}/manifold_nulls.json")
