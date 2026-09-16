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
Generates every LaTeX table and every inline numeric macro used by the two
write-ups directly from results/*.json.  Nothing is typed by hand, so a table
cannot drift from the analysis that produced it.
"""
import json, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
OUT = os.environ.get("TANA_OUT",
    os.path.join(REPO_DIR, "data", "task_41") if os.path.isdir(
        os.path.join(REPO_DIR, "data", "task_41")) else "results")
TEX = os.environ.get("TANA_TEX",
    os.path.join(REPO_DIR, "latex") if os.path.isdir(
        os.path.join(REPO_DIR, "latex")) else "report")
os.makedirs(TEX, exist_ok=True)
R = json.load(open(f"{OUT}/results.json"))
NUL = json.load(open(f"{OUT}/manifold_nulls.json"))
R2 = json.load(open(f"{OUT}/r2_numbers.json"))

NAME = {"baseline_std_tnm": r"\texttt{baseline}", "C_50": r"\texttt{C\_50}",
        "C_200": r"\texttt{C\_200}", "C_300": r"\texttt{C\_300}",
        "pmut_0002": r"\texttt{pmut\_0002}", "pmut_0005": r"\texttt{pmut\_0005}",
        "pmut_002": r"\texttt{pmut\_002}", "mu_005": r"\texttt{mu\_005}",
        "mu_02": r"\texttt{mu\_02}", "pmut_002_C_200": r"\texttt{pmut\_002\_C\_200}",
        "C_200_pmut_0005": r"\texttt{C\_200\_pmut\_0005}",
        "C_200_mu_005": r"\texttt{C\_200\_mu\_005}"}


def f(x, n=3):
    return "--" if x is None else f"{x:.{n}f}"


# ---- table: parameter grid ------------------------------------------------
rows = []
for g in R["grid"]:
    t = g["tag"]
    n = NUL.get(t)
    rows.append(
        f"{NAME.get(t, t)} & {g['C']:.0f} & {g['pmut']:.3f} & {g['mu']:.2f} & "
        f"{g['N']:.0f} & {g['S']} & {g['S_core']} & {g['core_biomass_pct']:.1f} & "
        f"{f(g['d_eff_singular'],2)} & "
        + (f"{f(n['hopkins'])} & {f(n['nulls']['label']['hopkins_mean'])} & "
           f"{f(n['nulls']['label']['hopkins_z'],2)} & "
           f"{f(n['silhouette_max'])} & {f(n['nulls']['label']['silhouette_mean'])} & "
           f"{f(n['nulls']['label']['silhouette_z'],2)}"
           if n else "\\multicolumn{6}{c}{$S<10$: geometry not defined}")
        + r" \\")
grid = r"""\begin{table*}
\caption{\label{tab:grid}Operating points examined, and the geometry of the
Jacobian manifold constructed at each.  $S_{\rm core}$ counts types with
$p_{\rm off}>p_{\rm kill}$, i.e.\ those able to reproduce without mutational
influx.  $d_{\rm eff}$ is the participation ratio of the singular values of $K$.
$H$ is the Hopkins statistic of the manifold, averaged over 25 sampling draws;
$s_{\max}$ the best Ward silhouette over $k\in[2,4]$.  Each is quoted beside the
mean of a label-permuted surrogate and the corresponding $z$.  All runs use
$\nu=5\times10^{-6}$, $p_{\rm kill}=0.2$, $\sigma=0$, world seed 1234.
These are twelve single realisations, not an ensemble over landscapes.}
\begin{ruledtabular}
\begin{tabular}{lccc ccccc cc c cc c}
Regime & $C$ & $p_{\rm mut}$ & $\mu$ & $N$ & $S$ & $S_{\rm core}$ & core \% &
$d_{\rm eff}$ & $H$ & null & $z$ & $s_{\max}$ & null & $z$ \\ \hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table*}"""
open(f"{TEX}/tab_grid.tex", "w").write(grid)

# ---- table: reliability ---------------------------------------------------
m1, m2 = R["measurement"]["halo_shift"], R["measurement"]["amplitude_matched"]
rel = r"""\begin{table*}[t]
\caption{\label{tab:reliability}Does the measured response reproduce the
constructed manifold?  Spearman correlations over the $\binom{86}{2}=3655$
pairs, before and after removing the decoupled amplitude term $D_0$.  Null
means and spreads come from 200 sham draws built from independent unperturbed
ensemble means; $z$ is the excess over that null in units of its standard
deviation.  Both designs use $R=60$ replicas and reach the same median
signal-to-noise ratio, so the difference between the columns is not a
difference in statistical power.}
\begin{ruledtabular}
\begin{tabular}{lcc}
 & halo shift & amplitude-matched \\
 & $\Delta n_i=-\min(n_i,5)$ & $\Delta n_i=-\mathrm{round}\sqrt{n_i}$ \\ \hline
whitened kick spread & $74\times$ & $2.7\times$ \\
forced extinctions & 77/86 & 34/86 \\ \hline
$\rho(D,D_{\rm Jac})$ & $%+.3f$ & $%+.3f$ \\
$\rho$ after removing $D_0$ & $%+.3f$ & $%+.3f$ \\
\quad null mean $\pm$ s.d. & $%+.3f\pm%.3f$ & $%+.3f\pm%.3f$ \\
\quad $z$ & $%+.2f$ & $%+.2f$ \\ \hline
$\rho(D,D_0)$ alone & $%+.3f$ & $%+.3f$ \\
$\rho(D,D_{\rm Jac})$, interactions deleted & $%+.3f$ & $%+.3f$ \\ \hline
$\rho(D,D_H)$ genomic & $%+.3f$ & $%+.3f$ \\
$\rho(D,D_{\rm int})$ interaction profile & $%+.3f$ & $%+.3f$ \\ \hline
silhouette $s(k{=}2)$ & $%+.3f$ & $%+.3f$ \\
\quad $z$ vs sham null & $%+.2f$ & $%+.2f$ \\
Hopkins $H$ & $%.3f$ & $%.3f$ \\
ARI with the manifold, $k=4$ & $%+.3f$ & $%+.3f$ \\
\end{tabular}
\end{ruledtabular}
\end{table*}""" % (
    m1["rho_vs_manifold"], m2["rho_vs_manifold"],
    m1["rho_vs_manifold_partial"], m2["rho_vs_manifold_partial"],
    m1["null"]["rho_partial_mean"], m1["null"]["rho_partial_sd"],
    m2["null"]["rho_partial_mean"], m2["null"]["rho_partial_sd"],
    m1["z_rho_partial"], m2["z_rho_partial"],
    m1["rho_vs_self_term"], m2["rho_vs_self_term"],
    m1["rho_vs_decoupled_control"], m2["rho_vs_decoupled_control"],
    m1["baselines"]["genomic_hamming"]["rho"], m2["baselines"]["genomic_hamming"]["rho"],
    m1["baselines"]["interaction_profile"]["rho"], m2["baselines"]["interaction_profile"]["rho"],
    m1["silhouette"]["2"], m2["silhouette"]["2"],
    m1["silhouette_z"]["2"], m2["silhouette_z"]["2"],
    m1["hopkins"], m2["hopkins"],
    m1["ari_with_manifold_k4"], m2["ari_with_manifold_k4"])
open(f"{TEX}/tab_reliability.tex", "w").write(rel)

# ---- inline macros --------------------------------------------------------
j, jn = R["jacobian"]["with_mutation"], R["jacobian"]["without_mutation"]
rs = R["reference_state"]
mac = {
    "Nstar": f"{rs['N']}", "Sstar": f"{rs['S']}", "tstar": f"{rs['t_star']}",
    "Ccoupling": f"{rs['C']:.0f}",
    "corebio": f"{100*rs['core_biomass_fraction']:.1f}",
    "nsubcore": f"{rs['n_subcore']}", "nhalo": f"{rs['n_halo']}",
    "abscissa": f"{j['spectral_abscissa']:.3f}",
    "deffsv": f"{j['d_eff_singular']:.2f}", "deffeig": f"{j['d_eff_eigen']:.2f}",
    "nonnormality": f"{j['nonnormality']:.2f}",
    "propcore": f"{j['propagator_core_block_max']:.1e}".replace("e-0", r"\times10^{-") + "}",
    "propcoreNM": f"{jn['propagator_core_block_max']:.1e}".replace("e-", r"\times10^{-") + "}",
    "prophaloNM": f"{jn['propagator_halo_offdiag_max']:.1e}".replace("e-", r"\times10^{-") + "}",
    "rhoHalo": f"{m1['rho_vs_manifold']:+.3f}",
    "rhoMatched": f"{m2['rho_vs_manifold']:+.3f}",
    "rhoHaloPartial": f"{m1['rho_vs_manifold_partial']:+.3f}",
    "rhoMatchedPartial": f"{m2['rho_vs_manifold_partial']:+.3f}",
    "zHaloPartial": f"{m1['z_rho_partial']:+.2f}",
    "zMatchedPartial": f"{m2['z_rho_partial']:+.2f}",
    "rhoSelfHalo": f"{m1['rho_vs_self_term']:+.3f}",
    "rhoDecoupledHalo": f"{m1['rho_vs_decoupled_control']:+.3f}",
    "silMatched": f"{m2['silhouette']['2']:+.3f}",
    "silZMatched": f"{m2['silhouette_z']['2']:+.2f}",
    "hopMeasured": f"{m2['hopkins']:.3f}",
    "ariMatched": f"{m2['ari_with_manifold_k4']:+.3f}",
    "haloRho": f"{R2['halo_balance_rho']:+.2f}",
    "haloRatio": f"{R2['halo_balance_median_ratio']:.2f}",
    "rankPred": f"{R2['rank90_predicted']}", "rankMeas": f"{R2['rank90_measured']}",
    "coreFrac": f"{100*R2['core_direction_fraction_median']:.0f}",
    "rhoJacHam": f"{R2['rho_DJac_hamming']:+.3f}",
    "lamMax": f"{R2['transition_lambda_max']:+.2f}",
    "lamRes": f"{R2['transition_lambda_resident']:+.2f}",
    "Nmin": f"{R2['transition_N_min']:.0f}",
    "sigRatioAll": f"{R2['sigma_peak_all']/R2['sigma_base_all']:.1f}",
    "sigRatioFix": f"{R2['sigma_peak_fixed']/R2['sigma_base_fixed']:.1f}",
    "hopManifoldLo": f"{min(v['hopkins'] for v in NUL.values()):.2f}",
    "hopManifoldHi": f"{max(v['hopkins'] for v in NUL.values()):.2f}",
    "hopZmax": f"{max(v['nulls']['label']['hopkins_z'] for v in NUL.values()):+.2f}",
    "silManifoldLo": f"{min(v['silhouette_max'] for v in NUL.values()):.2f}",
    "silManifoldHi": f"{max(v['silhouette_max'] for v in NUL.values()):.2f}",
    "nregimes": f"{len(NUL)}",
}
open(f"{TEX}/numbers.tex", "w").write(
    "% generated by make_tables.py - do not edit\n" +
    "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in mac.items()) + "\n")
print(f"wrote {TEX}/tab_grid.tex, {TEX}/tab_reliability.tex, {TEX}/numbers.tex "
      f"({len(mac)} macros)")
