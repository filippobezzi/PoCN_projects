# ============================================================================
#  40_T5_zout_sweep.R
#
#  T5 -- Robustness to structural noise: quantify how cross-community edges degrades plateau formation in synchronization time-scales
#  
#  T3 showed synchronization dynamics successfully reveals topological scales on clean networks. 
#  T4 showed that degree heterogeneity (hubs) breaks the method because hubs sit at the centroid of diffusion space. 
#  T5 addresses a different question: How robust is synchronization dynamics against structural noise?
#  We destroy the coarse, inter-community structure by adding random cross-community links.
#  Fix total degree at 18 and sweep inter-community degree z_out (1 to 14).
#  Tracks structural eigenvalues, spectral gap, plateau width, and NMI.
# 
#  Outputs:  F8_zout_sweep.png
#            ../data/T5_sweep.csv
#
#  Disclaimer: AI was used for generating efficient code, 
#  suggesting numerical improvements, and designing plotting functions. 
# ============================================================================

source("00_common.R")
dir.create(file.path("..", "data"), showWarnings = FALSE, recursive = TRUE)

K      <- 1.0
NREP   <- 3L # fast execution
zouts  <- seq(1, 14, by = 1)
ts     <- 10^seq(-2.4, 1.2, length.out = 70)
EPS    <- 0.05


gap  <- matrix(NA_real_, length(zouts), NREP)   # spectral gap
plat <- matrix(NA_real_, length(zouts), NREP)   # plateau width
nmiv <- matrix(NA_real_, length(zouts), NREP)   # recovery of the 4 large comms
nmi6 <- matrix(NA_real_, length(zouts), NREP)   # recovery of the 16 small comms
lamc <- matrix(NA_real_, length(zouts), NREP)   # coarse (4-communities) structural eigenvalue
lamf <- matrix(NA_real_, length(zouts), NREP)   # fine (16-communities) structural eigenvalue
bulk <- matrix(NA_real_, length(zouts), NREP)   # bulk edge from the null model

for (iz in seq_along(zouts)) {
  zo <- zouts[iz]
  # keep the 13:4 ratio between the two internal scales while z_out grows,
  # so that only the coarse structure is being degraded
  rest <- 18 - zo
  z1 <- rest * 13 / 17
  z2 <- rest * 4 / 17
  for (r in seq_len(NREP)) {
    net <- make_hier(z1, z2, zo, seed = 100 * iz + r)
    sp  <- spec(laplacian(net$A, "comb"))
    gap[iz, r]  <- sp$lambda[5] / sp$lambda[4]
    lamc[iz, r] <- sp$lambda[4]
    lamf[iz, r] <- sp$lambda[5]

    # Null model on every realisation
    spn <- spec(laplacian(config_null(net$A, 10L, seed = 200 * iz + r), "comb"))
    bulk[iz, r] <- spn$lambda[2]   # null model's algebraic connectivity 

    H  <- lapply(ts, function(t) slink_heights(diff_dist(dmap(sp, t, K, 1))))
    nc <- sapply(H, ncomp_at, eps = EPS)
    idx <- which(nc == 4)               # get time steps where 4 communities are detected
    plat[iz, r] <- if (length(idx) >= 2) log10(ts[max(idx)]) - log10(ts[min(idx)]) else 0      # measure plateau width

    # Best possible NMI
    b4 <- 0; b16 <- 0
    for (t in ts[seq(1, length(ts), by = 1)]) {
      D <- diff_dist(dmap(sp, t, K, 1))
      hc <- hclust(as.dist(D), "average") # build hierarchical tree based on diffusion distance
      # cut tree at different community levels to find best possible recovery
      hc4 <- cutree(hc, k = 4)
      hc16 <- cutree(hc, k = 16)
      b4  <- max(b4,  nmi(hc4, net$high))    # best 4-community recovery
      b16 <- max(b16, nmi(hc16, net$low))     # best 16-community recovery
    }
    nmiv[iz, r] <- b4
    nmi6[iz, r] <- b16
  }
  cat(sprintf("  z_out = %2d : gap = %.2f  plateau = %.2f  NMI4 = %.2f  NMI16 = %.2f\n",
              zo, mean(gap[iz, ]), mean(plat[iz, ]), mean(nmiv[iz, ]), mean(nmi6[iz, ])))
}

mrow <- function(M) rowMeans(M, na.rm = TRUE)
srow <- function(M) apply(M, 1, sd, na.rm = TRUE)

write.csv(data.frame(z_out = zouts,
                     lambda4 = mrow(lamc), lambda5 = mrow(lamf),
                     bulk_edge = mrow(bulk),
                     gap_ratio = mrow(gap), gap_sd = srow(gap),
                     plateau_width = mrow(plat), plateau_sd = srow(plat),
                     nmi4 = mrow(nmiv), nmi4_sd = srow(nmiv),
                     nmi16 = mrow(nmi6), nmi16_sd = srow(nmi6)),
          file.path("..", "data", "T5_sweep.csv"), row.names = FALSE)

saveRDS(list(
  zouts = zouts,
  lamc = lamc,
  lamf = lamf,
  bulk = bulk,
  gap = gap,
  plat = plat,
  nmiv = nmiv,
  nmi6 = nmi6
), file.path("..", "data", "T5_res.rds"))

cat("[T5] computation done -> data/T5_res.rds\n")
