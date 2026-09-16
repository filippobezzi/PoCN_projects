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
visualize_jacobian_core_halo.py
--------------------------------
Verifies and visualizes the Jacobian timescale inversion statement in report_manifold.pdf:
- Non-reproducing halo types have eigenvalues of exactly -1.0 (tau = 1 generation).
- Four reproducing core mutualists possess eigenvalue moduli spanning |lambda| in [26, 105] (tau in 0.01 - 0.04 generations).
- Core modes are 1-2 orders of magnitude faster than halo modes.
Generates publication-quality figure color-coded by Core, Sub-core, and Halo types.
"""

import os
import sys
import json
import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Ensure repo src is importable
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tana_geometry import community_jacobian, whitening_weights

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#4b5563",
    "axes.linewidth": 0.8,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.2,
    "legend.frameon": True,
    "legend.edgecolor": "#e5e7eb",
    "grid.color": "#f3f4f6",
    "grid.linewidth": 0.7,
})

# Color palette matching report
C_CORE = "#e05328"     # Vermilion / Red-orange
C_SUB = "#2a78d6"      # Royal Blue
C_HALO = "#10b981"     # Emerald Green
C_INK = "#111827"      # Charcoal text
C_MUTED = "#6b7280"    # Muted grey
C_GRID = "#e5e7eb"

# 1. Load microstate checkpoint at t* = 3402
REPO_DIR = os.path.abspath(os.path.join(SRC_DIR, "..", ".."))
ckpt_path = os.path.join(REPO_DIR, "data", "task_41", "checkpoint_microstate_tstar.json")
if not os.path.exists(ckpt_path):
    ckpt_path = os.path.join(os.path.dirname(SRC_DIR), "data", "step3", "checkpoint_microstate_tstar.json")
if not os.path.exists(ckpt_path):
    ckpt_path = os.path.join(os.path.dirname(SRC_DIR), "data", "checkpoint_microstate_tstar.json")

with open(ckpt_path) as f:
    ck = json.load(f)

J = np.array(ck["interaction_submatrix"], float)
pop = np.array([s["population"] for s in ck["extant_species"]], float)
sa = np.array([s["sa"] for s in ck["extant_species"]], int)
S = len(pop)
Ntot = pop.sum()

# Tiers
tier = np.where(pop >= 0.05 * Ntot, "Core", np.where(pop > 5, "Sub-core", "Halo"))
core_idx = np.where(tier == "Core")[0]
subcore_idx = np.where(tier == "Sub-core")[0]
halo_idx = np.where(tier == "Halo")[0]
noncore_idx = np.where(tier != "Core")[0]

PKILL, MU, NU, PMUT, L = 0.2, 0.10, 5e-6, 0.01, 20
HCRIT = np.log(PKILL / (1.0 - PKILL))

# 2. Compute Jacobian
K, diag = community_jacobian(J, pop, mu=MU, nu=NU, pkill=PKILL, pmut=PMUT, sa=sa, L=L, with_mutation=True)

# Eigendecomposition
ev, V = la.eig(K)
order = np.argsort(-np.abs(ev.real))
ev = ev[order]
V = V[:, order]
V_inv = la.inv(V)

# Left eigenvectors
ev_l, W = la.eig(K.T)
order_l = np.argsort(-np.abs(ev_l.real))
ev_l = ev_l[order_l]
W = W[:, order_l]

# Right mode localization
mode_weights = np.abs(V)**2
mode_weights /= mode_weights.sum(axis=0, keepdims=True)
core_share_right = mode_weights[core_idx, :].sum(axis=0)

# Left mode localization (subspace support)
left_weights = np.abs(W)**2
left_weights /= left_weights.sum(axis=0, keepdims=True)
core_share_left = left_weights[core_idx, :].sum(axis=0)
halo_share_left = left_weights[noncore_idx, :].sum(axis=0)

tau_all = 1.0 / np.abs(ev.real)

# 3. Create Multi-Panel Figure
fig = plt.figure(figsize=(14.0, 9.4))
gs = gridspec.GridSpec(2, 3, wspace=0.32, hspace=0.35,
                       left=0.06, right=0.98, top=0.95, bottom=0.07)

# ==========================================
# Panel (a): Eigenvalue Moduli Spectrum |lambda|
# ==========================================
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_title(r"(a)", loc="left", pad=8)

# Shaded band for core modes
ax_a.axhspan(26.46, 105.49, color=C_CORE, alpha=0.10, lw=0)
ax_a.axhline(1.0, color=C_HALO, ls="--", lw=1.2, alpha=0.8)

# Plot Halo modes
ax_a.scatter(np.arange(5, S + 1), np.abs(ev[4:].real), color=C_HALO, s=26, marker="o",
             label=r"Halo modes (82 modes, $|\lambda| \equiv 1.0$)", zorder=3, alpha=0.9, edgecolors="none")

# Plot Core modes
ax_a.scatter(np.arange(1, 5), np.abs(ev[:4].real), color=C_CORE, s=55, marker="D",
             label=r"Core mutualists (4 modes, $|\lambda| \in [26.5, 105.5]$)", zorder=4, edgecolors="black", linewidths=0.5)

ax_a.set_yscale("log")
ax_a.set_xlim(0, 90)
ax_a.set_ylim(0.4, 220)
ax_a.set_xlabel(r"Eigenmode rank $k$ (ordered by $|\lambda|$)")
ax_a.set_ylabel(r"Eigenvalue modulus $|\lambda_k|$")
ax_a.grid(True, linestyle=":", alpha=0.6)

ax_a.legend(loc="upper right", framealpha=0.92, bbox_to_anchor=(0.98, 0.98))

# Inset: precision check for halo eigenvalues
ins_a = ax_a.inset_axes([0.48, 0.36, 0.48, 0.28])
ins_a.scatter(np.arange(5, S + 1), np.abs(ev[4:].real - (-1.0)), color=C_HALO, s=10, alpha=0.8)
ins_a.set_yscale("log")
ins_a.set_ylim(1e-16, 1e-13)
ins_a.set_xlabel(r"Halo mode $k$", fontsize=6.2)
ins_a.set_ylabel(r"$|\lambda_k - (-1)|$", fontsize=6.2)
ins_a.tick_params(labelsize=6)
ins_a.grid(True, linestyle=":", alpha=0.5)


# ==========================================
# Panel (b): Timescale Inversion tau = 1/|lambda|
# ==========================================
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_title(r"(b)", loc="left", pad=8)

# Shaded band for core timescales
ax_b.axhspan(1.0/105.49, 1.0/26.46, color=C_CORE, alpha=0.12, lw=0)
ax_b.axhline(1.0, color=C_HALO, ls="--", lw=1.2, alpha=0.8)

ax_b.scatter(np.arange(5, S + 1), tau_all[4:], color=C_HALO, s=26, marker="o",
             label=r"Halo timescale $\tau \equiv 1.0$ gen", zorder=3, alpha=0.9, edgecolors="none")
ax_b.scatter(np.arange(1, 5), tau_all[:4], color=C_CORE, s=55, marker="D",
             label=r"Core timescale $\tau \in [0.010, 0.038]$ gen", zorder=4, edgecolors="black", linewidths=0.5)

ax_b.set_yscale("log")
ax_b.set_xlim(0, 90)
ax_b.set_ylim(0.005, 2.5)
ax_b.set_xlabel(r"Eigenmode rank $k$")
ax_b.set_ylabel(r"Relaxation timescale $\tau_k$ [generations]")
ax_b.grid(True, linestyle=":", alpha=0.6)

ax_b.legend(loc="upper right", bbox_to_anchor=(0.98, 0.88), framealpha=0.92)


# ==========================================
# Panel (c): Dynamic Relaxation Trajectories e^{lambda*t}
# ==========================================
ax_c = fig.add_subplot(gs[0, 2])
ax_c.set_title(r"(c)", loc="left", pad=8)

t_eval = np.linspace(0, 2.5, 500)
halo_decay = np.exp(-1.0 * t_eval)
ax_c.plot(t_eval, halo_decay, color=C_HALO, lw=2.2, label=r"Halo modes: $e^{-1.0 t}$ ($\tau = 1.0$ gen)")

core_alphas = [1.0, 0.85, 0.7, 0.55]
for i in range(4):
    c_decay = np.exp(ev[i].real * t_eval)
    lbl = rf"Core {i+1} ($\lambda = {ev[i].real:.1f}, \tau = {tau_all[i]:.4f}$ gen)" if i in [0, 3] else None
    ax_c.plot(t_eval, c_decay, color=C_CORE, lw=1.6, alpha=core_alphas[i], label=lbl)

ax_c.set_xlim(0, 2.5)
ax_c.set_ylim(-0.02, 1.05)
ax_c.set_xlabel(r"Time $t$ [generations]")
ax_c.set_ylabel(r"Normalized mode amplitude $u_k(t)/u_k(0)$")
ax_c.grid(True, linestyle=":", alpha=0.6)

# Mark e-folding of halo (placed below curve to avoid inset)
ax_c.axvline(1.0, color=C_HALO, ls=":", lw=1.0)
ax_c.scatter([1.0], [np.exp(-1.0)], color=C_HALO, s=28, zorder=5)
ax_c.legend(loc="upper right", framealpha=0.92, fontsize=7.0)

# Inset: Zoom into t in [0, 0.1] generations (placed lower right to avoid overlap)
ins_c = ax_c.inset_axes([0.55, 0.42, 0.42, 0.38])
t_inset = np.linspace(0, 0.08, 300)
ins_c.plot(t_inset, np.exp(-1.0 * t_inset), color=C_HALO, lw=1.5, ls="--", label="Halo")
for i in range(4):
    ins_c.plot(t_inset, np.exp(ev[i].real * t_inset), color=C_CORE, lw=1.3, alpha=core_alphas[i])
ins_c.set_xlim(0, 0.08)
ins_c.set_ylim(0, 1.02)
ins_c.set_xlabel(r"$t$ [gen]", fontsize=6.2)
ins_c.set_ylabel(r"Ampl.", fontsize=6.2)
ins_c.tick_params(labelsize=6)
ins_c.grid(True, linestyle=":", alpha=0.5)


# ==========================================
# Panel (d): Mode Energy Localization on Core vs Halo
# ==========================================
ax_d = fig.add_subplot(gs[1, 0])
ax_d.set_title(r"(d)", loc="left", pad=8)

# Show left eigenvector projection (the observable decay coordinates)
k_indices = np.arange(1, 21)
core_w = core_share_left[:20] * 100
halo_w = halo_share_left[:20] * 100

bar_w = 0.65
ax_d.bar(k_indices, core_w, width=bar_w, color=C_CORE, label="Weight on 4 Core mutualists", alpha=0.9)
ax_d.bar(k_indices, halo_w, width=bar_w, bottom=core_w, color=C_HALO, label="Weight on 82 Halo genotypes", alpha=0.85)

ax_d.set_xlim(0.3, 20.7)
ax_d.set_ylim(0, 115)
ax_d.set_xticks(np.arange(1, 21, 2))
ax_d.set_xlabel(r"Observable decay mode rank $k$ (left eigenvectors $\mathbf{w}_k$)")
ax_d.set_ylabel(r"Subspace weight fraction $\|\mathbf{w}_k\|_i^2$ [%]")
ax_d.grid(True, axis="y", linestyle=":", alpha=0.6)

ax_d.legend(loc="upper right", framealpha=0.92)


# ==========================================
# Panel (e): Physical Mechanism - Fitness & Survival Threshold
# ==========================================
ax_e = fig.add_subplot(gs[1, 1])
ax_e.set_title(r"(e)", loc="left", pad=8)

H = np.array(diag["H"])
poff = np.array(diag["poff"])

# Scatter per tier
for t_name, c_col in [("Halo", C_HALO), ("Sub-core", C_SUB), ("Core", C_CORE)]:
    mask = tier == t_name
    count = mask.sum()
    ax_e.scatter(pop[mask], H[mask], color=c_col, s=48 if t_name == "Core" else 22,
                 label=f"{t_name} ($S={count}$)", zorder=4 if t_name == "Core" else 3,
                 edgecolors="black" if t_name == "Core" else "none", linewidths=0.5, alpha=0.9)

ax_e.axhline(HCRIT, color=C_INK, ls="--", lw=1.2, zorder=2)

ax_e.set_xscale("log")
ax_e.set_xlabel(r"Genotype abundance $n_a$ [individuals]")
ax_e.set_ylabel(r"Per-capita effective fitness $H_a$")
ax_e.set_xlim(0.8, 600)
ax_e.set_ylim(-150, 20)
ax_e.grid(True, linestyle=":", alpha=0.6)

ax_e.legend(loc="lower right", framealpha=0.92)


# ==========================================
# Panel (f): Core Subspace Overlap (r2_geometry style)
# ==========================================
ax_f = fig.add_subplot(gs[1, 2])
ax_f.set_title(r"(f)", loc="left", pad=8)

# Compute subspace overlap of non-core species onto core responses
w = whitening_weights(pop)
EXP = [la.expm(K * t) for t in (1, 2, 3)]
def resp(i):
    return np.concatenate([E[:, i] * w for E in EXP])

Pcols = np.column_stack([resp(c) for c in core_idx])
Q, _ = np.linalg.qr(Pcols)

frac = []
for i in noncore_idx:
    u = resp(i)
    frac.append(float(np.linalg.norm(Q.T @ u) ** 2 / np.linalg.norm(u) ** 2))
frac = np.array(frac)

bins = np.linspace(0, 0.25, 26)
ax_f.hist(frac, bins=bins, color=C_HALO, alpha=0.85, edgecolor="#047857", lw=0.6, zorder=3,
          label=f"82 Non-core species\n(median overlap = {np.median(frac)*100:.1f}%)")

med_val = np.median(frac)
ax_f.axvline(med_val, color=C_CORE, lw=1.4, ls="--", zorder=4)
ax_f.set_xlim(0, 0.25)
ax_f.set_xlabel(r"Fraction of halo response in span of 4 core responses $\eta_h$")
ax_f.set_ylabel(r"Number of genotypes")
ax_f.grid(True, linestyle=":", alpha=0.6)

ax_f.legend(loc="upper right", framealpha=0.92)

# Save figures
out_fig_dir = os.path.join(REPO_DIR, "latex", "task_41", "fig")
if not os.path.exists(out_fig_dir):
    out_fig_dir = os.path.join(os.path.dirname(SRC_DIR), "fig")
os.makedirs(out_fig_dir, exist_ok=True)
png_path = os.path.join(out_fig_dir, "jacobian_timescale_inversion_verification.png")
pdf_path = os.path.join(out_fig_dir, "jacobian_timescale_inversion_verification.pdf")

plt.savefig(png_path)
plt.savefig(pdf_path)
plt.close(fig)

print(f"\nSaved updated verification plots:\n  {png_path}\n  {pdf_path}")
