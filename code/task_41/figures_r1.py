#!/usr/bin/env python3
# ==============================================================================
# AI ASSISTANCE DISCLAIMER
# In accordance with academic guidelines, the author acknowledges the use of
# generative artificial intelligence assistance (DeepMind's Antigravity AI assistant)
# for writing assistance, code development, data formatting, and proofreading the
# manuscript. Some analytical derivations, mathematical proofs, numerical simulations,
# and physical interpretations still need rigorous verification by the author.
# ==============================================================================
"""Figures for write-up 1: constructing the Jacobian manifold and testing it."""
import json, os, sys
import numpy as np
import scipy.linalg as la
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
from tana_geometry import (community_jacobian, whitening_weights, response_distance,
                           propagator_distance, self_term_predictor, hopkins,
                           silhouette_profile, sham_null_distances, ward_labels,
                           partial_spearman)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA = os.environ.get("TANA_DATA",
    os.path.join(REPO_DIR, "data", "task_41") if os.path.exists(os.path.join(REPO_DIR, "data", "task_41")) else "data")
OUT = os.environ.get("TANA_OUT",
    DATA if os.path.exists(os.path.join(DATA, "results.json")) else "results")
FIG = os.environ.get("TANA_FIG",
    os.path.join(REPO_DIR, "latex", "task_41", "fig") if os.path.exists(os.path.join(REPO_DIR, "latex", "task_41", "fig"))
    else os.path.join(REPO_DIR, "latex", "images") if os.path.isdir(os.path.join(REPO_DIR, "latex", "images"))
    else "fig")
os.makedirs(FIG, exist_ok=True)
PKILL, MU, NU, PMUT, L, W, SEED = 0.2, .10, 5e-6, .01, 20, (1, 3), 20260912

R = json.load(open(f"{OUT}/results.json"))
NUL = json.load(open(f"{OUT}/manifold_nulls.json"))
ck = json.load(open(f"{DATA}/checkpoint_microstate_tstar.json"))
J = np.array(ck["interaction_submatrix"], float)
pop = np.array([s["population"] for s in ck["extant_species"]], float)
sa = np.array([s["sa"] for s in ck["extant_species"]], int)
S, N = len(pop), pop.sum()
w = whitening_weights(pop); tri = np.triu_indices(S, 1)
K, kd = community_jacobian(J, pop, mu=MU, nu=NU, pkill=PKILL, pmut=PMUT, sa=sa, L=L)
Kn, kdn = community_jacobian(J, pop, mu=MU, nu=NU, pkill=PKILL, with_mutation=False)
H = np.array(kd["H"]); poff = np.array(kd["poff"])
HCRIT = np.log(PKILL / (1 - PKILL))
DES = {"halo_shift": np.minimum(pop, 5),
       "amplitude_matched": np.clip(np.round(np.sqrt(pop)), 1, pop)}
ENS_PATHS = {"halo_shift": os.path.join(DATA, "perturbation_responses_R60.npz"),
             "amplitude_matched": os.path.join(DATA, "perturbation_responses_matched_R60.npz")}
ENS = ENS_PATHS
LBL = {"halo_shift": r"halo shift  $\Delta n=-\min(n,5)$",
       "amplitude_matched": r"amplitude-matched  $\Delta n=-\mathrm{round}\sqrt{n}$"}


def tiers(p):
    return np.where(p >= 0.05 * N, "Core", np.where(p > 5, "Sub-core", "Halo"))


T = tiers(pop)

# ---------------------------------------------------------------- figure 1
fig, ax = plt.subplots(2, 2, figsize=(7.1, 4.8))
stats = np.loadtxt(f"{DATA}/C_200_stats.dat")
t, Np, Sp = stats[:, 0], stats[:, 1], stats[:, 2]
a = panel(tidy(ax[0, 0]), "a")
a.axvspan(805, 6000, color=C1, alpha=.08, lw=0)
a.plot(t, Np, color=INK, lw=1.0, label=r"$N(t)$")
a.axvline(3402, color=C2, lw=1.2, ls="--")
a.set_xlabel("generation"); a.set_ylabel(r"$N(t)$")
a.legend(loc="upper right")

b = panel(tidy(ax[0, 1]), "b")
tr = np.genfromtxt(f"{DATA}/C_200_top5_trace.tsv", names=True, dtype=None, encoding=None)
cols = [c for c in tr.dtype.names if c not in ("t", "N", "S")]
for c, col in zip(cols, CAT5):
    b.plot(tr["t"], tr[c], color=col, lw=.8, label=c.lstrip("f"))
b.axvline(3402, color=INK2, lw=.8, ls="--")
b.set_xlim(700, 6000); b.set_ylim(0, 560)
b.set_xlabel("generation"); b.set_ylabel(r"$n_a(t)$")
b.legend(ncol=3, loc="upper center", title="genotype (plateau residents)",
         title_fontsize=6.5, columnspacing=1.0, handletextpad=.5)

c = panel(tidy(ax[1, 0], grid="both"), "c")
order = np.argsort(-pop)
for tier in ("Core", "Sub-core", "Halo"):
    m = T[order] == tier
    c.scatter(np.arange(S)[m] + 1, pop[order][m], s=11, color=TIER[tier],
              label=f"{tier} ($S={m.sum()}$)", zorder=3, lw=0)
c.set_yscale("log"); c.set_xlabel("abundance rank"); c.set_ylabel(r"$n_a$")
c.legend(loc="upper right")

d = panel(tidy(ax[1, 1], grid="both"), "d")
for tier in ("Core", "Sub-core", "Halo"):
    m = T == tier
    d.scatter(pop[m], H[m], s=11, color=TIER[tier], label=tier, zorder=3, lw=0)
d.axhline(HCRIT, color=INK, lw=1.0, ls="--")
d.set_xscale("log"); d.set_xlabel(r"$n_a$"); d.set_ylabel(r"$H_a$")
d.set_ylim(-185, 22); d.legend(loc="lower right")
# the core/halo fitness gap is ~100 units wide, so H_crit is invisible at full
# scale: inset the neighbourhood of the threshold.
ins = d.inset_axes([.42, .50, .55, .44])
for tier in ("Core", "Sub-core", "Halo"):
    m = T == tier
    ins.scatter(pop[m], H[m], s=13, color=TIER[tier], zorder=3, lw=0)
ins.axhline(HCRIT, color=INK, lw=1.0, ls="--")
ins.set_xscale("log"); ins.set_ylim(-2.2, .4); ins.set_xlim(40, 700)
ins.set_facecolor("#fafaf8")
for sp_ in ("top", "right"): ins.spines[sp_].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/r1_operating_point.pdf"); plt.close(fig)
print("r1_operating_point")

# ---------------------------------------------------------------- figure 2
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.6), gridspec_kw=dict(wspace=.34))
a = panel(tidy(ax[0], grid="both"), "a")
for lab, dg, col, mk in (("with mutation kernel", kd, C1, "o"),
                         ("mutation term dropped", kdn, C2, "s")):
    v = np.array(dg["eig_abs_sorted"])
    a.plot(np.arange(1, len(v) + 1), v, mk, color=col, ms=2.6, lw=0, label=lab)
a.set_yscale("log"); a.set_xlabel("index"); a.set_ylabel(r"$|\lambda|$")
a.axhline(1, color=INK2, lw=.7, ls=":")
a.legend(loc="center right")

for k, (lab, Kx, dg) in enumerate([("with mutation kernel", K, kd),
                                   ("mutation term dropped", Kn, kdn)]):
    b = panel(ax[k + 1], "bc"[k])
    P = np.abs(la.expm(Kx))
    im = b.imshow(np.log10(np.maximum(P, 1e-18)), cmap=SEQ, vmin=-16, vmax=0,
                  interpolation="nearest")
    b.set_xticks([]); b.set_yticks([])
    b.set_xlabel(r"$\log_{10}|e^{K}|_{ab}$", fontsize=7.5)
    plt.colorbar(im, ax=b, fraction=.046, pad=.03)
fig.tight_layout(); fig.savefig(f"{FIG}/r1_propagator.pdf"); plt.close(fig)
print("r1_propagator")

# ---------------------------------------------------------------- figure 3
regs = list(NUL.keys())
x = np.arange(len(regs))
fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.1), gridspec_kw=dict(wspace=.26))
a = panel(tidy(ax[0]), "a")
a.bar(x - .22, [NUL[r]["hopkins"] for r in regs], .4, color=C1,
      label="measured manifold", zorder=3)
a.errorbar(x + .22, [NUL[r]["nulls"]["label"]["hopkins_mean"] for r in regs],
           yerr=[NUL[r]["nulls"]["label"]["hopkins_sd"] for r in regs], fmt="s",
           color=NULLC, ms=3.4, lw=1, capsize=2, label="label-permuted null", zorder=4)
a.plot(x, [NUL[r]["nulls"]["none"]["hopkins_mean"] for r in regs], "^",
       color=C2, ms=4, lw=0, label=r"$J=0$ control (no interactions)", zorder=5)
a.axhline(.5, color=INK, lw=.8, ls="--")
a.set_xticks(x); a.set_xticklabels(regs, rotation=40, ha="right", fontsize=6.2)
a.set_ylabel(r"Hopkins $H$"); a.set_ylim(.30, 1.06)
a.legend(loc="lower center", ncol=1, fontsize=6.4, columnspacing=1.0,
         handletextpad=.4, bbox_to_anchor=(.45, -.02))

b = panel(tidy(ax[1]), "b")
b.bar(x - .22, [NUL[r]["silhouette_max"] for r in regs], .4, color=C1,
      label="measured manifold", zorder=3)
b.errorbar(x + .22, [NUL[r]["nulls"]["label"]["silhouette_mean"] for r in regs],
           yerr=[NUL[r]["nulls"]["label"]["silhouette_sd"] for r in regs], fmt="s",
           color=NULLC, ms=3.4, lw=1, capsize=2, label="label-permuted null", zorder=4)
b.plot(x, [NUL[r]["nulls"]["none"]["silhouette_mean"] for r in regs], "^",
       color=C2, ms=4, lw=0, label=r"$J=0$ control", zorder=5)
b.set_xticks(x); b.set_xticklabels(regs, rotation=40, ha="right", fontsize=6.2)
b.set_ylabel(r"$\max_k s(k)$"); b.set_ylim(0, 1.62)
b.legend(loc="upper right", ncol=1, fontsize=6.4, columnspacing=1.0, handletextpad=.4)
fig.tight_layout(); fig.savefig(f"{FIG}/r1_manifold_vs_null.pdf"); plt.close(fig)
print("r1_manifold_vs_null")

# ---------------------------------------------------------------- figure 4
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.9), gridspec_kw=dict(wspace=.42))
for k, tag in enumerate(DES):
    dn = DES[tag]
    DJ = propagator_distance(K, np.diag(-dn / N), w, *W)
    D0 = self_term_predictor(dn, pop)
    a = panel(tidy(ax[k], grid="both"), "ab"[k])
    a.scatter(D0[tri], DJ[tri], s=3, alpha=.25, color=C1, lw=0, rasterized=True)
    a.set_xlabel(r"amplitude term  $D_0(i,j)$")
    a.set_ylabel(r"manifold  $D_{\rm Jac}(i,j)$" if k == 0 else "")
    a.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    a.yaxis.get_offset_text().set_fontsize(6.4)
    a.locator_params(axis="x", nbins=4)
c = panel(tidy(ax[2]), "c")
# Hopkins is elevated above a uniform point set for every construction that
# shares the propagator geometry - including one with no interactions at all -
# so the elevation is a property of the construction, not of the network.
rng = np.random.default_rng(SEED)
dnm = DES["amplitude_matched"]
DJm = propagator_distance(K, np.diag(-dnm / N), w, *W)
Kz, _ = community_jacobian(np.zeros_like(J), pop, mu=MU, nu=NU, pkill=PKILL,
                           pmut=PMUT, sa=sa, L=L)
DJz = propagator_distance(Kz, np.diag(-dnm / N), w, *W)
perm = rng.permutation(S)
Kp, _ = community_jacobian(J[np.ix_(perm, perm)], pop, mu=MU, nu=NU, pkill=PKILL,
                           pmut=PMUT, sa=sa, L=L)
DJp = propagator_distance(Kp, np.diag(-dnm / N), w, *W)
Dmeas = response_distance(np.load(ENS_PATHS["amplitude_matched"])["delta_x"], w, *W)
Drand = np.zeros((S, S)); Drand[tri] = rng.random(len(tri[0])); Drand += Drand.T
Xu = rng.random((S, 4)); Dunif = ssd.squareform(ssd.pdist(Xu))
items = [("manifold", DJm, C1), (r"manifold, $J=0$", DJz, C2),
         ("manifold,\nlabels permuted", DJp, C2),
         ("measured\nresponse", Dmeas, C3),
         ("random\nmetric", Drand, MUTED), ("uniform\npoints", Dunif, MUTED)]
ys, es = [], []
for lab, M, col in items:
    m_, s_ = hopkins(M, seed=SEED, return_sd=True); ys.append(m_); es.append(s_)
c.bar(np.arange(len(items)), ys, .62, yerr=es, capsize=2.5,
      color=[i[2] for i in items], zorder=3)
c.axhline(.5, color=INK, lw=.8, ls="--")
c.set_xticks(np.arange(len(items)))
c.set_xticklabels([i[0].replace("\\n", " ") for i in items], fontsize=6.0,
                  rotation=38, ha="right")
c.set_ylabel(r"Hopkins $H$"); c.set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig(f"{FIG}/r1_amplitude_dominance.pdf"); plt.close(fig)
print("r1_amplitude_dominance")

# ---------------------------------------------------------------- figure 5

fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.65), gridspec_kw=dict(wspace=.42))
bars = []
for k, tag in enumerate(DES):
    z = np.load(ENS[tag])
    D = response_distance(z["delta_x"], w, *W)
    dn = DES[tag]
    DJ = propagator_distance(K, np.diag(-dn / N), w, *W)
    m = R["measurement"][tag]
    a = panel(tidy(ax[k], grid="both"), "ab"[k])
    a.scatter(DJ[tri], D[tri], s=3, alpha=.25, color=C1, lw=0, rasterized=True)
    sl, ic, *_ = st.linregress(DJ[tri], D[tri])
    xs = np.linspace(DJ[tri].min(), DJ[tri].max(), 50)
    a.plot(xs, ic + sl * xs, color=C2, lw=1.4, label="least-squares fit")
    a.set_xlabel(r"manifold prediction  $D_{\rm Jac}(i,j)$")
    a.set_ylabel(r"measured  $D(i,j)$" if k == 0 else "")
    a.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    a.xaxis.get_offset_text().set_fontsize(6.4)
    a.yaxis.get_offset_text().set_fontsize(6.4)
    a.locator_params(axis="x", nbins=4)
    a.legend(loc="lower right")
    bars.append((m["rho_vs_manifold"], m["rho_vs_manifold_partial"],
                 m["null"]["rho_sd"], m["null"]["rho_partial_sd"]))
c = panel(tidy(ax[2]), "c")
xx = np.arange(2)
c.bar(xx - .2, [b[0] for b in bars], .38, yerr=[b[2] for b in bars], capsize=2,
      color=C1, label="raw  " + r"$\rho(D,D_{\rm Jac})$", zorder=3)
c.bar(xx + .2, [b[1] for b in bars], .38, yerr=[b[3] for b in bars], capsize=2,
      color=C2, label=r"partial, $D_0$ removed", zorder=3)
c.axhline(0, color=INK2, lw=.8)
c.set_xticks(xx); c.set_xticklabels(["halo\nshift", "amplitude\nmatched"], fontsize=7)
c.set_ylabel("Spearman correlation")
c.set_ylim(-.10, .74); c.set_xlim(-.62, 1.62)
c.legend(loc="upper center", fontsize=6.4, ncol=1, bbox_to_anchor=(.52, 1.03))
fig.tight_layout(); fig.savefig(f"{FIG}/r1_reliability.pdf"); plt.close(fig)
print("r1_reliability")

# ---------------------------------------------------------------- figure 6
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.65), gridspec_kw=dict(wspace=.36))
a = panel(tidy(ax[0]), "a")
ks = list(range(2, 9))
for k, tag in enumerate(DES):
    m = R["measurement"][tag]
    a.plot(ks, [m["silhouette"][str(q)] for q in ks], "o-", color=CAT3[k], label=LBL[tag])
    if k == 0:
        nm = [m["null"]["silhouette"][str(q)]["mean"] for q in ks]
        ns = [m["null"]["silhouette"][str(q)]["sd"] for q in ks]
        a.plot(ks, nm, "s--", color=NULLC, label="sham null (mean $\\pm2\\sigma$)")
        a.fill_between(ks, np.array(nm) - 2 * np.array(ns), np.array(nm) + 2 * np.array(ns),
                       color=NULLC, alpha=.22, lw=0)
a.set_xlabel("number of clusters $k$"); a.set_ylabel("mean silhouette $s(k)$")
a.legend(loc="upper left", fontsize=6.6)

z = np.load(ENS["amplitude_matched"])
D = response_distance(z["delta_x"], w, *W)
b = panel(ax[1], "b")
Z = sch.ward(ssd.squareform(D, checks=False))
o = sch.dendrogram(Z, no_plot=True)["leaves"]
im = b.imshow(D[np.ix_(o, o)], cmap=SEQ, interpolation="nearest")
b.set_xticks([]); b.set_yticks([]); b.set_xlabel(r"$D(i,j)$, amplitude-matched", fontsize=7.5)
plt.colorbar(im, ax=b, fraction=.046, pad=.03)
DJ = propagator_distance(K, np.diag(-DES["amplitude_matched"] / N), w, *W)
c = panel(ax[2], "c")
im = c.imshow(DJ[np.ix_(o, o)], cmap=SEQ, interpolation="nearest")
c.set_xticks([]); c.set_yticks([]); c.set_xlabel(r"$D_{\rm Jac}(i,j)$", fontsize=7.5)
plt.colorbar(im, ax=c, fraction=.046, pad=.03)
fig.tight_layout(); fig.savefig(f"{FIG}/r1_clustering.pdf"); plt.close(fig)
print("r1_clustering")
