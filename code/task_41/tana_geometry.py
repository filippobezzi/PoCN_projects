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
tana_geometry.py
----------------
Shared estimators for the TaNa perturbation-response geometry analysis.

Everything the two write-ups report is computed here, so that a number can
only appear in a figure or a table if this module produced it.

Design notes on the three corrections that matter:

1.  The community Jacobian includes the mutation kernel.  Without it the
    82 halo types have p_off ~ 0 and K reduces to -I on the halo block, so
    exp(K t) at t = 1 is numerically e^{-t} I to ~1e-14 and the propagator
    carries no network information at all.  With the kernel, mutational
    influx from the reproducing core couples every halo row to the core and
    exp(K) retains structure at O(1e-2).

2.  Cluster-tendency statistics use a RELATIVE eigenvalue threshold.  An
    absolute 1e-6 cut is meaningless for a distance matrix whose entries are
    O(1e-4): every Gram eigenvalue falls below it and the estimator silently
    returns its fallback value.

3.  The sham null is built from independent ensemble MEANS of disjoint
    replica blocks, matching the estimator being tested.  Drawing single
    replicas (with replacement) instead inflates the null variance ~60-fold
    and plants duplicate rows that Ward resolves as perfect clusters.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
import scipy.stats as st
# sklearn is imported lazily inside silhouette_profile() to avoid a hard
# dependency for scripts that never call that function.

__all__ = [
    "hamming_matrix", "mutation_kernel", "community_jacobian",
    "whitening_weights", "response_distance", "propagator_distance",
    "self_term_predictor", "hopkins", "silhouette_profile",
    "partial_spearman", "sham_null_distances", "ward_labels",
]

# --------------------------------------------------------------------------
# model-level quantities
# --------------------------------------------------------------------------

def hamming_matrix(sa: np.ndarray) -> np.ndarray:
    """Bitwise Hamming distance on the {0,1}^L hypercube between genotype ids."""
    sa = np.asarray(sa, dtype=np.int64)
    return np.array([[bin(int(a) ^ int(b)).count("1") for b in sa] for a in sa],
                    dtype=np.float64)


def mutation_kernel(sa: np.ndarray, pmut: float, L: int = 20) -> np.ndarray:
    """M[c, a] = P(offspring of c is of type a) = pmut^h (1-pmut)^(L-h)."""
    h = hamming_matrix(sa)
    return pmut ** h * (1.0 - pmut) ** (L - h)


def community_jacobian(J, n, *, mu, nu, pkill, pmut=None, sa=None, L=20,
                       with_mutation=True):
    """
    Jacobian of the mean-field abundance dynamics

        dn_a/dt = -n_a + (1/pkill) sum_c n_c p_off(H_c) M(c -> a),

    evaluated at the qESS state n.  Returns (K, diagnostics).

    With ``with_mutation=False`` the mutation kernel is dropped, which is the
    form used in the first draft of this analysis; it is kept only so the
    write-up can quantify what that omission costs.
    """
    J = np.asarray(J, float)
    n = np.asarray(n, float)
    S = len(n)
    N = n.sum()

    H_I = (J @ n) / N
    H = H_I - mu * N - nu * N ** 2
    poff = 1.0 / (1.0 + np.exp(-H))
    dpoff = poff * (1.0 - poff)

    # dH_c/dn_b = J(c,b)/N - H_I(c)/N - mu - 2 nu N
    dH = J / N - np.outer(H_I, np.ones(S)) / N - mu - 2.0 * nu * N

    if with_mutation:
        if sa is None or pmut is None:
            raise ValueError("with_mutation=True needs sa and pmut")
        M = mutation_kernel(sa, pmut, L)          # M[c, a]
        MT = M.T                                   # MT[a, c] = M(c -> a)
        K = -np.eye(S) + (1.0 / pkill) * (poff[None, :] * MT
                                          + (MT * (n * dpoff)[None, :]) @ dH)
    else:
        K = (n[:, None] / pkill) * dpoff[:, None] * dH
        np.fill_diagonal(K, np.diag(K) + (poff / pkill - 1.0))

    ev = la.eigvals(K)
    sv = la.svdvals(K)
    P1 = la.expm(K)
    core = np.where(poff > pkill)[0]
    halo = np.setdiff1d(np.arange(S), core)
    offdiag_halo = (np.abs(P1[np.ix_(halo, halo)]
                           - np.diag(np.diag(P1[np.ix_(halo, halo)]))).max()
                    if len(halo) > 1 else 0.0)
    diag = dict(
        S=S, N=float(N),
        n_core=int(len(core)), core_idx=core.tolist(),
        H_crit=float(np.log(pkill / (1.0 - pkill))),
        spectral_abscissa=float(ev.real.max()),
        numerical_abscissa=float(np.linalg.eigvalsh((K + K.T) / 2).max()),
        eig_abs_sorted=np.sort(np.abs(ev))[::-1].tolist(),
        d_eff_singular=float(sv.sum() ** 2 / (sv ** 2).sum()),
        d_eff_eigen=float(np.abs(ev).sum() ** 2 / (np.abs(ev) ** 2).sum()),
        nonnormality=float(np.linalg.norm(K - K.T) / np.linalg.norm(K)),
        # how much structure the one-generation propagator actually retains
        propagator_core_block_max=float(np.abs(P1[np.ix_(core, core)]).max())
        if len(core) else 0.0,
        propagator_halo_offdiag_max=float(offdiag_halo),
        poff=poff.tolist(), H=H.tolist(),
    )
    return K, diag


# --------------------------------------------------------------------------
# response geometry
# --------------------------------------------------------------------------

def whitening_weights(n: np.ndarray) -> np.ndarray:
    """w_a = 1/sqrt(n_a): equalises the demographic noise variance per coordinate."""
    return 1.0 / np.sqrt(np.maximum(1.0, np.asarray(n, float)))


def response_distance(delta_x, weights, t_low=1, t_high=3):
    """Time-averaged weighted Euclidean distance between response vectors."""
    delta_x = np.asarray(delta_x)
    S = delta_x.shape[0]
    w2 = np.asarray(weights, float) ** 2
    D = np.zeros((S, S))
    for t in range(t_low, t_high + 1):
        a = delta_x[:, t, :]
        d = a[:, None, :] - a[None, :, :]
        D += np.sqrt((w2[None, None, :] * d ** 2).sum(-1))
    D /= float(t_high - t_low + 1)
    np.fill_diagonal(D, 0.0)
    return D


def propagator_distance(K, delta_x0, weights, t_low=1, t_high=3):
    """Same metric applied to the linear prediction u_i(t) = exp(K t)(-e_i) dn_i."""
    S = K.shape[0]
    w2 = np.asarray(weights, float) ** 2
    D = np.zeros((S, S))
    for t in range(t_low, t_high + 1):
        pred = delta_x0 @ la.expm(K * t).T
        d = pred[:, None, :] - pred[None, :, :]
        D += np.sqrt((w2[None, None, :] * d ** 2).sum(-1))
    D /= float(t_high - t_low + 1)
    np.fill_diagonal(D, 0.0)
    return D


def self_term_predictor(delta_n, n):
    """
    The distance matrix that would be obtained if each perturbation decayed
    in place with no coupling whatsoever:

        D0(i,j) = sqrt( dn_i^2 / n_i + dn_j^2 / n_j )   (up to a factor e^{-t}/N)

    This is the trivial amplitude content of any response-distance matrix and
    must be partialled out before a correlation is read as network structure.
    """
    a = np.asarray(delta_n, float) ** 2 / np.maximum(np.asarray(n, float), 1.0)
    D = np.sqrt(a[:, None] + a[None, :])
    np.fill_diagonal(D, 0.0)
    return D


# --------------------------------------------------------------------------
# geometry diagnostics
# --------------------------------------------------------------------------

def _mds_coords(D, max_dim=4, rel_tol=1e-10):
    """Classical MDS coordinates, with a RELATIVE eigenvalue cut."""
    D2 = np.asarray(D, float) ** 2
    G = -0.5 * (D2 - D2.mean(0, keepdims=True) - D2.mean(1, keepdims=True) + D2.mean())
    vals, vecs = la.eigh(G)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    scale = max(vals.max(), np.finfo(float).tiny)
    keep = vals > rel_tol * scale
    d = int(min(max_dim, keep.sum()))
    if d < 2:
        return None, vals
    return vecs[:, :d] * np.sqrt(vals[:d]), vals


def hopkins(D, n_samples=30, seed=0, max_dim=4, n_repeats=25, return_sd=False):
    """
    Hopkins statistic, averaged over ``n_repeats`` independent sampling draws.

    A single draw with m = 30 of S = 86 points has a sampling spread of ~0.05,
    which is the same size as the differences between the models being compared,
    so a one-draw value is not reportable.  The mean over repeats is returned;
    pass return_sd=True to get (mean, sd) and quote the spread.
    """
    vals = [_hopkins_once(D, n_samples, seed + r, max_dim) for r in range(n_repeats)]
    vals = np.array(vals, float)
    if np.all(np.isnan(vals)):
        return (np.nan, np.nan) if return_sd else np.nan
    m, s = float(np.nanmean(vals)), float(np.nanstd(vals))
    return (m, s) if return_sd else m


def _hopkins_once(D, n_samples=30, seed=0, max_dim=4):
    """
    Hopkins cluster-tendency statistic on the classical-MDS embedding of D.

    H ~ 0.5  : points are indistinguishable from a uniform sample (no tendency)
    H > 0.75 : strong tendency to clump
    H < 0.25 : regular / lattice-like

    Returns np.nan when the embedding has fewer than two usable dimensions,
    rather than a default value that would be indistinguishable from a real
    measurement of 0.5.
    """
    D = np.asarray(D, float)
    S = D.shape[0]
    if S < 10:
        return np.nan
    X, _ = _mds_coords(D, max_dim=max_dim)
    if X is None:
        return np.nan
    m = min(n_samples, S // 2)
    rng = np.random.default_rng(seed)
    idx = rng.choice(S, m, replace=False)
    rnd = rng.uniform(X.min(0), X.max(0), size=(m, X.shape[1]))
    u = np.array([np.linalg.norm(X - rnd[i], axis=1).min() for i in range(m)])
    w = np.empty(m)
    for k, i in enumerate(idx):
        d = np.linalg.norm(X - X[i], axis=1)
        d[i] = np.inf
        w[k] = d.min()
    tot = u.sum() + w.sum()
    return float(u.sum() / tot) if tot > 0 else np.nan


def ward_labels(D, k):
    return sch.fcluster(sch.ward(ssd.squareform(np.asarray(D, float), checks=False)),
                        k, "maxclust")


def silhouette_profile(D, ks=range(2, 9)):
    """
    Mean silhouette for each k.  Returns nan where the partition degenerates
    (a near-constant distance matrix collapses Ward to a single cluster), so a
    degenerate case is never silently reported as a low score.
    """
    from sklearn.metrics import silhouette_score  # lazy import
    D = np.asarray(D, float)
    out = {}
    for k in ks:
        lab = ward_labels(D, k)
        if len(np.unique(lab)) < 2 or len(np.unique(lab)) >= len(lab):
            out[int(k)] = float("nan")
        else:
            out[int(k)] = float(silhouette_score(D, lab, metric="precomputed"))
    return out


def partial_spearman(x, y, controls):
    """Spearman correlation of x and y after linearly removing the controls' ranks."""
    rx, ry = st.rankdata(x), st.rankdata(y)
    Z = np.column_stack([st.rankdata(c) for c in controls] + [np.ones(len(x))])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def sham_null_distances(sham_traj, weights, n_draws=200, t_low=1, t_high=3, seed=0):
    """
    Demographic-noise floor for the response-distance estimator.

    Each draw splits the R unperturbed replicas into two disjoint halves, then
    builds a pseudo-response for every target as the difference of two
    independent ensemble means drawn WITHOUT replacement from the first half.
    This matches the estimator under test: the measured delta_x is also a
    difference of ensemble means, so the null carries the same variance.

    Yields one null distance matrix per draw.
    """
    sham_traj = np.asarray(sham_traj)
    R, T1, S = sham_traj.shape
    rng = np.random.default_rng(seed)
    half = R // 2
    block = max(1, half // 2)
    for _ in range(n_draws):
        perm = rng.permutation(R)
        g1, g2 = perm[:half], perm[half:]
        m2 = sham_traj[g2].mean(0)
        ds = np.empty((S, T1, S), dtype=np.float32)
        for i in range(S):
            pick = rng.choice(g1, block, replace=False)
            ds[i] = sham_traj[pick].mean(0) - m2
        yield response_distance(ds, weights, t_low, t_high)
