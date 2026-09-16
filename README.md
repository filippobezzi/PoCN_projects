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
  +  I explore the physical mechanisms governing perturbation-response geometry and phase-space dynamics across qESS transitions. At a mutualistic qESS plateau ($N^*, S^*$), where synergistic mutualism overcomes carrying capacity mortality, global competition imposes an exact survival threshold: four core mutualists concentrate most of biomass, while non-autonomous halo genotypes persist through mutational leakage from the core. Linearising the community Jacobian $K$ reveals a spectral timescale inversion: halo linear responses are governed by single-node demographic relaxation ($\lambda = -1.0$) rather than collective core modes ($|\lambda| \in [26, 105]$). Across $5{,}220$ stochastic replicas, an amplitude-matched protocol ($\Delta n_i = -\mathrm{round}\sqrt{n_i}$) eliminates demographic kick artefacts, uncovering a continuous response manifold smoother than demographic noise, rather than functional clustering. Finally, tracking an evolutionary transition reveals that stochastic mutational occupancy drives a linear instability ($\mathrm{Re}(\lambda_{\max}) > 0$), triggering deterministic population collapse and latent centroid migration. A fixed-membership control confirms that phase-space delocalisation surges independent of taxonomic turnover.
