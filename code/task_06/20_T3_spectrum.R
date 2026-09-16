# ============================================================================
#  20_T3_spectrum.R
#
#  T3 -- Spectral properties of the Kuramoto Laplacian & Community Plateaus
#
#  Outputs:  F2_spectrum.png
#            F_spectrum_vs_eigenvectors.png
#            F3_plateaus.png
#            ../data/T3_plateaus.csv
#            ../data/T3_res.rds
#
#  Disclaimer: AI was used for generating efficient code, 
#  suggesting numerical improvements, and designing plotting functions. 
# ============================================================================

source("00_common.R")
dir.create(file.path("..", "data"), showWarnings = FALSE, recursive = TRUE)

K <- 1.0
net <- make_hier(13, 4, 1, seed = 1L)
sp  <- spec(laplacian(net$A, "comb"))

# Degree-preserving null model (keeps degrees, destroys correlation)
An  <- config_null(net$A, nswap_per_edge = 20L, seed = 11L)
spn <- spec(laplacian(An, "comb"))

bulk_edge <- spn$lambda[2]
cat(sprintf("[T3] %s : lambda_2..5 = %s | null lambda_2 = %.2f\n",
            net$label, paste(sprintf("%.3f", sp$lambda[2:5]), collapse = " "),
            bulk_edge))

# Localisation: IPR per mode
ipr_vals <- ipr(sp$V)

# Mode Lifetime & Eigenvector Matrix Data
nz <- sp$lambda[-1]
ranks <- seq_along(nz) + 1

M <- 30
W <- sp$V[, 2:(M + 1)] # modes alpha = 2..31
W <- W / rep(apply(abs(W), 2, max), each = 256) # rescale each mode to [-1, 1]

cols_c <- c("#c15f3c", "#3c6ec1", "#8a8580")
modes_c <- c(2, 8, 120)

# Plateaus
ts   <- 10^seq(-3.0, 1.0, length.out = 90)
epss <- c(0.01, 0.05, 0.10)              # ball radii
Tvals <- T_of_eps(epss)

H <- lapply(ts, function(t) slink_heights(diff_dist(dmap(sp, t, K, 1))))
ncomp_mat <- sapply(epss, function(e) sapply(H, ncomp_at, eps = e))

predw <- function(sp, m) c(1 / (K * sp$lambda[m + 1]), 1 / (K * sp$lambda[m]))
w4  <- predw(sp, 4)
w16 <- predw(sp, 16)

# Measured plateau lengths at eps = 0.05
nc <- ncomp_mat[, 2]
plateau_width <- function(m) {
  idx <- which(nc == m)
  if (length(idx) < 2) return(0)
  log10(ts[max(idx)]) - log10(ts[min(idx)])
}

tab <- data.frame(
  network = net$label,
  scale   = c(4, 16),
  lambda_m   = c(sp$lambda[4],  sp$lambda[16]),
  lambda_mp1 = c(sp$lambda[5],  sp$lambda[17]),
  predicted_log10_width = c(log10(sp$lambda[5] / sp$lambda[4]),
                            log10(sp$lambda[17] / sp$lambda[16])),
  measured_log10_width  = c(plateau_width(4), plateau_width(16))
)

write.csv(tab, file.path("..", "data", "T3_plateaus.csv"), row.names = FALSE)
cat("\n[T3] predicted vs measured plateau widths (log10 t):\n")
print(tab, digits = 3)

saveRDS(list(
  net = net,
  K = K,
  sp = sp,
  spn = spn,
  bulk_edge = bulk_edge,
  ipr_vals = ipr_vals,
  nz = nz,
  ranks = ranks,
  M = M,
  W = W,
  cols_c = cols_c,
  modes_c = modes_c,
  ts = ts,
  epss = epss,
  Tvals = Tvals,
  ncomp_mat = ncomp_mat,
  w4 = w4,
  w16 = w16
), file.path("..", "data", "T3_res.rds"))

cat("[T3] computation done -> data/T3_res.rds\n")
