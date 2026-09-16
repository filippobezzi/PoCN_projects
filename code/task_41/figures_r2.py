#!/usr/bin/env python3
# ==============================================================================
# AI ASSISTANCE DISCLAIMER
# In accordance with academic guidelines, the author acknowledges the use of
# generative artificial intelligence assistance (DeepMind's Antigravity AI assistant)
# for writing assistance, code development, data formatting, and proofreading the
# manuscript. Some analytical derivations, mathematical proofs, numerical simulations,
# and physical interpretations still need rigorous verification by the author.
# ==============================================================================
"""Figures for write-up 2: why the measurement cannot resolve the manifold,
and what the manifold does show about coevolution across a qESS transition."""
import json, os, sys
import numpy as np
import scipy.linalg as la
import scipy.spatial.distance as ssd
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import *
from tana_geometry import (community_jacobian, whitening_weights, response_distance,
                           propagator_distance, mutation_kernel, hamming_matrix,
                           self_term_predictor)

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
RES = json.load(open(f"{OUT}/results.json"))

ck = json.load(open(f"{DATA}/checkpoint_microstate_tstar.json"))
J = np.array(ck["interaction_submatrix"], float)
pop = np.array([s["population"] for s in ck["extant_species"]], float)
sa = np.array([s["sa"] for s in ck["extant_species"]], int)
S, N = len(pop), pop.sum()
w = whitening_weights(pop); tri = np.triu_indices(S, 1)
K, kd = community_jacobian(J, pop, mu=MU, nu=NU, pkill=PKILL, pmut=PMUT, sa=sa, L=L)
H = np.array(kd["H"]); poff = np.array(kd["poff"])
H_I = (J @ pop) / N
HCRIT = np.log(PKILL / (1 - PKILL))
core = poff > PKILL
T = np.where(pop >= 0.05 * N, "Core", np.where(pop > 5, "Sub-core", "Halo"))
dn_m = np.clip(np.round(np.sqrt(pop)), 1, pop)

# ---------------------------------------------------------------- mechanism
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.7), gridspec_kw=dict(wspace=.40))

a = panel(tidy(ax[0], grid="both"), "a")
pen = MU * N + NU * N ** 2
o = np.argsort(-pop)
a.scatter(np.arange(S) + 1, H_I[o], s=12, color=C1, lw=0, zorder=3,
          label=r"cooperative gain $H_I$")
a.axhline(pen + HCRIT, color=C2, lw=1.4,
          label=r"survival line $\mu N+\nu N^2+H_{\rm crit}$")
a.set_yscale("symlog", linthresh=1); a.set_xlabel("abundance rank")
a.set_ylabel(r"per-capita energy $H_I$"); a.set_ylim(-2, 900)
a.legend(loc="upper right", fontsize=6.2)

b = panel(tidy(ax[1], grid="both"), "b")
M = mutation_kernel(sa, PMUT, L)
pred = (1 / PKILL) * ((pop * poff) @ M)
h = ~core
rho_h = st.spearmanr(pred[h], pop[h]).statistic
jit = np.random.default_rng(SEED).normal(0, .035, h.sum())
b.scatter(pred[h] * np.exp(jit), pop[h] * np.exp(jit), s=13, color=C3, lw=0,
          alpha=.75, zorder=3, label="non-core types")
lim = [.6, 20]
b.plot(lim, lim, color=INK, lw=1.0, ls="--", label="parameter-free prediction")
b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(*lim); b.set_ylim(*lim)
b.set_xlabel(r"predicted $n_h^\ast$ (no fitted parameters)")
b.set_ylabel(r"observed $n_h$")
b.legend(loc="lower right", fontsize=6.2)

c = panel(tidy(ax[2], grid="both"), "c")
U = np.vstack([(np.diag(-dn_m / N) @ la.expm(K * t).T) * w for t in (1, 2, 3)])
sv = la.svdvals(U); sv /= sv.sum()
z = np.load(f"{DATA}/perturbation_responses_matched_R60.npz")
Dm = np.vstack([z["delta_x"][:, t, :] * w for t in (1, 2, 3)])
sm = la.svdvals(Dm); sm /= sm.sum()
c.plot(np.arange(1, 21), np.cumsum(sv)[:20], "o-", color=C1,
       label="manifold prediction")
c.plot(np.arange(1, 21), np.cumsum(sm)[:20], "s-", color=C2, label="measured response")
c.axhline(.9, color=INK, lw=.8, ls="--")
n90 = int(np.searchsorted(np.cumsum(sv), .9) + 1)
m90 = int(np.searchsorted(np.cumsum(sm), .9) + 1)
c.set_xlabel("singular value index"); c.set_ylabel("cumulative share")
c.set_ylim(0, 1.05)
c.legend(loc="lower right", fontsize=6.4)
fig.tight_layout(); fig.savefig(f"{FIG}/r2_mechanism.pdf"); plt.close(fig)
print("r2_mechanism")

# ---------------------------------------------------------------- geometry
DJ = propagator_distance(K, np.diag(-dn_m / N), w, *W)
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.8), gridspec_kw=dict(wspace=.46))
D2 = DJ ** 2
G = -0.5 * (D2 - D2.mean(0, keepdims=True) - D2.mean(1, keepdims=True) + D2.mean())
ev, evec = la.eigh(G); idx = np.argsort(ev)[::-1]; ev, evec = ev[idx], evec[:, idx]
Y = evec[:, :2] * np.sqrt(np.maximum(ev[:2], 0))
a = panel(tidy(ax[0], grid="both"), "a")
for tier in ("Halo", "Sub-core", "Core"):
    m = T == tier
    a.scatter(Y[m, 0], Y[m, 1], s=(28 if tier == "Core" else 12), color=TIER[tier],
              label=tier, lw=0, alpha=.85, zorder=3)
a.set_xlabel(r"$y_1$"); a.set_ylabel(r"$y_2$")
a.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
a.xaxis.get_offset_text().set_fontsize(6); a.yaxis.get_offset_text().set_fontsize(6)
a.legend(loc="lower right", fontsize=6.6)

b = panel(tidy(ax[1], grid="both"), "b")
DH = hamming_matrix(sa)
hs = DH[tri].astype(int)
parts = [DJ[tri][hs == k] for k in range(1, 10)]
bp = b.boxplot(parts, positions=range(1, 10), widths=.62, showfliers=False,
               patch_artist=True, medianprops=dict(color=INK, lw=1.1))
for pch in bp["boxes"]:
    pch.set_facecolor(C1); pch.set_alpha(.55); pch.set_edgecolor(INK2); pch.set_lw(.7)
b.set_xlabel(r"$D_H$ [bits]")
b.set_ylabel(r"$D_{\rm Jac}(i,j)$")
b.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
b.yaxis.get_offset_text().set_fontsize(6)

c = panel(tidy(ax[2], grid="both"), "c")
# each type's response over the window is one vector in R^{3S}; the core
# responses span a 4-dimensional subspace of it.
EXP = [la.expm(K * t) for t in (1, 2, 3)]
def resp(i):
    return np.concatenate([E[:, i] * w for E in EXP])
Pcols = np.column_stack([resp(c) for c in np.where(core)[0]])   # (3S, 4)
Q, _ = np.linalg.qr(Pcols)
frac = []
for i in np.where(~core)[0]:
    u = resp(i)
    frac.append(float(np.linalg.norm(Q.T @ u) ** 2 / np.linalg.norm(u) ** 2))
frac = np.array(frac)
c.hist(frac, bins=np.linspace(0, .3, 25), color=C3, alpha=.85, zorder=3)
c.set_xlim(0, .3)
c.axvline(np.median(frac), color=INK, lw=1.2, ls="--")
c.set_xlabel("overlap fraction")
c.set_ylabel("number of types")
fig.tight_layout(); fig.savefig(f"{FIG}/r2_geometry.pdf"); plt.close(fig)
print("r2_geometry")

# ---------------------------------------------------------------- transition
tr = json.load(open(f"{DATA}/mu_02_transition.json"))
Ju = np.array(tr["J_union"], float)
ids = [s["id"] for s in tr["union_species"]]
ix = {s: i for i, s in enumerate(ids)}
MUt, PMt = tr["mu"], tr["pmut"]
tl = tr["timeline"]
gens = np.array([g["t"] for g in tl])
Npop = np.array([g["N"] for g in tl], float)

lam, Fen = [], []
for g in tl:
    cnt = {int(k): v for k, v in g["counts"].items()}
    e = list(cnt); n_ = np.array([cnt[s] for s in e], float)
    Je = Ju[np.ix_([ix[s] for s in e], [ix[s] for s in e])]
    Kg, dg = community_jacobian(Je, n_, mu=MUt, nu=NU, pkill=PKILL, pmut=PMt,
                                sa=np.array(e), L=L, with_mutation=True)
    lam.append(dg["spectral_abscissa"])
    Fen.append(float(n_ @ Je @ n_) / max(n_.sum(), 1))
lam, Fen = np.array(lam), np.array(Fen)

ACT = {615444: "core 1", 205879: "core 2", 594964: "invader",
       549908: "bridge", 747028: "successor"}
present = [g for g in tl if all(str(k) in g["counts"] for k in ACT)]
ref = present[len(present) // 2] if present else tl[len(tl) // 2]
rc = {int(k): v for k, v in ref["counts"].items()}
# Singletons dominate the whitened metric and hide the orbit, so the embedding
# is built on the types that actually carry biomass at the reference generation,
# plus the five named actors.
re_ = [s for s, v in rc.items() if v >= 3 or s in ACT]
rn = np.array([rc[s] for s in re_], float)
Jr = Ju[np.ix_([ix[s] for s in re_], [ix[s] for s in re_])]
Kr, _ = community_jacobian(Jr, rn, mu=MUt, nu=NU, pkill=PKILL, pmut=PMt,
                           sa=np.array(re_), L=L, with_mutation=True)
dnr = np.clip(np.round(np.sqrt(rn)), 1, rn)
DJr = propagator_distance(Kr, np.diag(-dnr / rn.sum()), whitening_weights(rn), *W)
D2 = DJr ** 2
G = -0.5 * (D2 - D2.mean(0, keepdims=True) - D2.mean(1, keepdims=True) + D2.mean())
ev, evec = la.eigh(G); idx = np.argsort(ev)[::-1]
Yr = (evec[:, idx][:, :2] * np.sqrt(np.maximum(ev[idx][:2], 0)))
coord = {s: Yr[i] for i, s in enumerate(re_)}

cen, sig, sig_fix = [], [], []
# control: hold membership at the types resident before the invasion, so that a
# change in dispersion cannot be produced by turnover of who is in the average
fixed = {int(k) for k in tl[0]["counts"]} & set(re_)
for g in tl:
    cnt = {int(k): v for k, v in g["counts"].items()}
    for store, keys in ((0, None), (1, fixed)):
        ww, cv = 0.0, np.zeros(2)
        for s, p in cnt.items():
            if s in coord and (keys is None or s in keys):
                ww += p; cv += p * coord[s]
        if ww == 0:
            (cen if store == 0 else sig_fix).append(np.nan * np.ones(2) if store == 0 else np.nan)
            if store == 0: sig.append(np.nan)
            continue
        cv /= ww
        var = sum(p * np.sum((coord[s] - cv) ** 2) for s, p in cnt.items()
                  if s in coord and (keys is None or s in keys)) / ww
        if store == 0:
            cen.append(cv); sig.append(np.sqrt(var))
        else:
            sig_fix.append(np.sqrt(var))
cen = np.array(cen); sig = np.array(sig); sig_fix = np.array(sig_fix)

fig, ax = plt.subplots(2, 2, figsize=(7.1, 5.0))
PH = [(3820, 3835, C1, "resident qESS"), (3836, 3839, C2, "invasion & crash"),
      (3840, 3843, C4, "reorganisation"), (3844, 3870, C3, "successor qESS")]
a = panel(tidy(ax[0, 0]), "a")
for lo, hi, cc, lb in PH:
    a.axvspan(lo, hi, color=cc, alpha=.13, lw=0)
a.plot(gens, Npop, color=INK, lw=1.3, label=r"$N(t)$", zorder=4)
a.set_xlabel("generation"); a.set_ylabel(r"$N(t)$")
a2 = a.twinx(); a2.plot(gens, lam, color=C2, lw=1.3, label=r"${\rm Re}\,\lambda_{\max}$")
a2.axhline(0, color=INK2, lw=.8, ls=":")
a2.set_ylabel(r"${\rm Re}\,\lambda_{\max}$", color=C2)
a2.tick_params(axis="y", colors=C2); a2.spines["top"].set_visible(False)
from matplotlib.lines import Line2D
a.legend([Line2D([], [], color=INK, lw=1.3), Line2D([], [], color=C2, lw=1.3)],
         [r"$N(t)$", r"${\rm Re}\,\lambda_{\max}$"], loc="center left", fontsize=6.6)

b = panel(tidy(ax[0, 1], grid="both"), "b")
ok = ~np.isnan(cen[:, 0])
seg = np.stack([cen[ok][:-1], cen[ok][1:]], axis=1)
lc = LineCollection(seg, cmap="plasma", array=gens[ok][:-1], lw=2.0, zorder=3)
b.add_collection(lc)
if 594964 in coord and 747028 in coord:
    b.plot([coord[594964][0], coord[747028][0]],
           [coord[594964][1], coord[747028][1]],
           color=INK2, ls="--", lw=1.1, zorder=2, alpha=0.85)
ACT_ROLE = {615444: "ancestors", 205879: "ancestors", 594964: "invader",
            549908: "bridge", 747028: "successor"}
ROLE_COLOR = {"ancestors": C1, "invader": C2, "bridge": C4, "successor": C3}
roles_plotted = set()
for s, role in ACT_ROLE.items():
    if s in coord:
        lbl = role if role not in roles_plotted else None
        roles_plotted.add(role)
        b.scatter(*coord[s], s=48, color=ROLE_COLOR[role], zorder=5,
                  edgecolor=INK, lw=.7, label=lbl)
b.legend(loc="lower left", fontsize=6.3, frameon=True, framealpha=0.88,
         edgecolor="none", handletextpad=0.3, borderpad=0.3)
b.autoscale_view(); b.set_xlabel(r"$y_1$"); b.set_ylabel(r"$y_2$")
b.margins(.16)
plt.colorbar(lc, ax=b, fraction=.046, pad=.03, label="generation")

c = panel(tidy(ax[1, 0], grid="both"), "c")
keys = [s for s in ACT if s in coord]
hh, dd, lbl = [], [], []
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        hh.append(bin(keys[i] ^ keys[j]).count("1"))
        dd.append(np.linalg.norm(coord[keys[i]] - coord[keys[j]]))
        lbl.append(f"{ACT[keys[i]]}–{ACT[keys[j]]}")
allh = hamming_matrix(np.array(re_))
t2 = np.triu_indices(len(re_), 1)
alld = ssd.squareform(ssd.pdist(Yr))
c.scatter(allh[t2] + np.random.default_rng(1).normal(0, .09, len(t2[0])), alld[t2],
          s=5, color=MUTED, alpha=.35, lw=0, label="all extant pairs", rasterized=True)
c.scatter(hh, dd, s=34, color=C2, zorder=4, label="transition actors", lw=0)
c.set_xlabel(r"genomic Hamming distance [bits]")
c.set_ylabel("latent response distance")
c.legend(loc="upper left", fontsize=6.6)

d = panel(tidy(ax[1, 1]), "d")
for lo, hi, cc, _ in PH:
    d.axvspan(lo, hi, color=cc, alpha=.13, lw=0)
d.plot(gens, sig, color=C1, lw=1.5, label="all types present")
sig_fix = np.asarray(sig_fix, float)
d.plot(gens, sig_fix, color=C2, lw=1.5, ls="--",
       label="membership held at the pre-invasion set")
gone = np.where(np.array(sig_fix) < 1e-8)[0]
if len(gone):
    sig_fix[gone] = np.nan
d.set_yscale("log"); d.set_ylim(8e-5, 40)
d.set_xlabel("generation"); d.set_ylabel(r"latent dispersion $\Sigma(t)$")
d.legend(loc="center left", fontsize=6.2)
_r_all = np.nanmax(sig) / sig[0]
_r_fix = np.nanmax(sig_fix) / sig_fix[0]
fig.tight_layout(); fig.savefig(f"{FIG}/r2_transition.pdf"); plt.close(fig)
print("r2_transition")

json.dump({"halo_balance_rho": float(rho_h),
           "halo_balance_median_ratio": float(np.median(pred[h] / pop[h])),
           "rank90_predicted": n90, "rank90_measured": m90,
           "core_direction_fraction_median": float(np.median(frac)),
           "rho_DJac_hamming": float(st.spearmanr(DJ[tri], DH[tri]).statistic),
           "transition_lambda_max": float(np.max(lam)),
           "transition_lambda_resident": float(np.median(lam[gens <= 3835])),
           "transition_N_min": float(Npop.min()),
           "transition_reference_generation": int(ref["t"]),
           "sigma_peak_all": float(np.nanmax(sig)),
           "sigma_peak_fixed": float(np.nanmax(sig_fix)),
           "sigma_base_all": float(np.nanmedian(sig[gens <= 3835])),
           "sigma_base_fixed": float(np.nanmedian(sig_fix[gens <= 3835]))},
          open(f"{OUT}/r2_numbers.json", "w"), indent=2)
print("wrote results/r2_numbers.json")
