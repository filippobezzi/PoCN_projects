# PoCN_projects
number:
  + 6
  + 41

name: 
  + Dynamics on networks
  + Functional clusters in the Tangled Nature model through perturbation-response geometry.

score:
  + 0.3
  + 1.2

description:
  + I examine the hypothesis by which transient synchronisation in the Kuramoto model reveals hierarchical community structure in complex networks. In the linearised regime around the steady state, I prove an exact analytical identity demonstrating that the pairwise order parameter $\rho_{ij}(t) = \langle \cos(\theta_i - \theta_j) \rangle$ is equivalent to a diffusion-map kernel $\exp(-d_t^2(i,j)/2)$, where $d_t(i,j)$ is the network's diffusion distance at time $t$. We validate this identity on a synthetic two-level hierarchical network ($N=256$, $z_{\mathrm{in}1}=13, z_{\mathrm{in}2}=4, z_{\mathrm{out}}=1$) using RK4 numerical simulations. By comparing the Laplacian spectrum against a degree-preserving null model and computing the Inverse Participation Ratio (IPR), I separate delocalised structural modes from localised bulk noise. I show that spectral gaps dictate transient plateaus in single-linkage component counts. Furthermore, I analyse two key failure mechanisms of synchronisation: targeted degree heterogeneity (hubs), where adding 8 hubs ($\sim 3\%$ of nodes) destroys plateau formation by placing hubs at the diffusion centroid, and structural noise injection (increasing $z_{\mathrm{out}}$), showing that plateau visibility degrades before the spectral gaps collapse.
  +  I explore the physical mechanisms governing perturbation-response geometry and phase-space dynamics across qESS transitions. At a mutualistic qESS plateau ($N^{\*}, S^{\*}$), where synergistic mutualism overcomes carrying capacity mortality, global competition imposes an exact survival threshold: four core mutualists concentrate most of biomass, while non-autonomous halo genotypes persist through mutational leakage from the core. Linearising the community Jacobian $K$ reveals a spectral timescale inversion: halo linear responses are governed by single-node demographic relaxation ($\lambda = -1.0$) rather than collective core modes ($|\lambda| \in [26, 105]$). Across $5{,}220$ stochastic replicas, an amplitude-matched protocol ($\Delta n_i = -\mathrm{round}\sqrt{n_i}$) eliminates demographic kick artefacts, uncovering a continuous response manifold smoother than demographic noise, rather than functional clustering. Finally, tracking an evolutionary transition reveals that stochastic mutational occupancy drives a linear instability ($\mathrm{Re}(\lambda_{\max}) > 0$), triggering deterministic population collapse and latent centroid migration. A fixed-membership control confirms that phase-space delocalisation surges independent of taxonomic turnover.

---

## Repository Structure

```
PoCN_projects/
├── report.pdf                         # Official unified report (compiled via report_template_v3)
├── report_task_06.pdf                 # Dedicated manuscript for Task 6 (PRL format)
├── report_task_41_manifold.pdf        # Dedicated manuscript for Task 41 (PRA format)
├── supplementary_task_41_manifold.pdf # Supplementary Information for Task 41
├── latex/                             # Official LaTeX report source (report_template_v3)
│   ├── main.tex                       # Master report file
│   ├── base/                          # Style and package definitions
│   ├── sections/                      # Front page, task chapters, and appendix
│   │   ├── front_page.tex             # Official UniPd cover page
│   │   ├── task1.tex                  # Task 6 chapter
│   │   ├── task2.tex                  # Task 41 chapter
│   │   └── appendix.tex               # Mathematical proofs and derivations
│   ├── images/                        # UniPd logos and all publication figures
│   ├── bibliography.bib               # Unified BibTeX database
│   └── standalone_reports/            # Dedicated standalone RevTeX sources
│       ├── task_06/                   # Standalone Task 6 PRL LaTeX manuscript
│       └── task_41/                   # Standalone Task 41 PRA LaTeX manuscript & SI
├── code/                              # Standalone source code organized by task
│   ├── task_06/                       # Standalone Base R scripts (00_common.R, run_all.R, etc.)
│   └── task_41/                       # C++ simulation model and Python analysis scripts
└── data/                              # Processed output datasets (no raw simulation dumps)
    ├── task_06/                       # Pre-computed simulation results & network edgelists (.rds, .csv)
    └── task_41/                       # Processed checkpoints, transition series & perturbation tensors (.json, .npz)
```

## Reproduction Instructions

### Task 6: Synchronization Dynamics (R)
All scripts are written in standard Base R (no third-party package dependencies required):
```bash
cd code/task_06
# Run the complete analysis pipeline and render all figures:
Rscript run_all.R

# Or generate/update all figures from pre-computed data:
Rscript -e 'source("plot_figures.R"); plot_all()'
```

### Task 41: Tangled Nature Model (C++ / Python)
- **Compile C++ Simulator**:
  ```bash
  cd code/task_41
  clang++ -std=c++11 -O3 -o tangled_nature tangled_nature.cpp
  ```
- **Reproduce Manifold Figures**:
  ```bash
  cd code/task_41
  python3 figures_r2.py
  python3 visualize_jacobian_core_halo.py
  ```

## LaTeX Compilation

### Official Unified Report (`report.pdf`)
To recompile the official report abiding by `report_template_v3`:
```bash
cd latex
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cp main.pdf ../report.pdf
```

### Dedicated Standalone RevTeX Manuscripts
- **Task 6 (PRL style)**:
  ```bash
  cd latex/standalone_reports/task_06
  pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
  ```
- **Task 41 (PRA style & SI)**:
  ```bash
  cd latex/standalone_reports/task_41
  pdflatex report_manifold.tex && bibtex report_manifold && pdflatex report_manifold.tex && pdflatex report_manifold.tex
  pdflatex supplementary_manifold.tex && bibtex supplementary_manifold && pdflatex supplementary_manifold.tex && pdflatex supplementary_manifold.tex
  ```
