# ============================================================================
#  10_T2_validation.R
#
#  T2 -- Synthetic networks and validation of the central identity
#
#        rho_ij(t) = < cos(theta_i - theta_j) >  =  exp( -1/2 d_t(i,j)^2 )
#
#  Integrate FULL NONLINEAR Kuramoto model and compare against LINEAR prediction.  
#
#  Outputs:  F1_rho_heatmaps.png     (F1)
#
#  Disclaimer: AI was used for generating efficient code, 
#  suggesting numerical improvements, and designing plotting functions. 
# ============================================================================

source("00_common.R")
dir.create(file.path("..", "data"), showWarnings = FALSE, recursive = TRUE)

QUICK <- FALSE                        # TRUE -> ~20 s, FALSE -> ~2-4 min
M     <- if (QUICK) 100L else 400L    # ensemble size
DTMAX <- if (QUICK) 0.02  else 0.01   # RK4 step ceiling
K     <- 1.0                          # coupling; sets the unit of time
SIGMA <- 0.15                         # s.d. of the Gaussian initial phases

# focus on the early evolution of synchrony rather than slow approach to steady-state
times <- 10^seq(-2, 0.7, length.out = 30)     # logarithmic spacing for t in [0.01,5]

net <- make_hier(13, 4, 1, seed = 1L)

cat(sprintf("[T2] network %s   mean degree %.2f\n", net$label, mean(net$k)))
set.seed(21L)
Th0 <- matrix(rnorm(M * net$N, 0, SIGMA), M, net$N)
cat("  simulating (Gaussian ICs)...\n")
rho_sim <- simulate_rho(net$A, K, Th0, times, DTMAX)

res <- list(net = net, times = times, rho_sim = rho_sim)

show_at <- c(15, 20, 26)                    # three indices into `times` (t = 0.201, 0.587, 2.126)
diss <- lapply(show_at, function(s) log10(pmax(1 - rho_sim[[s]], 1e-16)))
zl <- range(unlist(diss), finite = TRUE)  # absolute range for colour scale

sp <- spec(laplacian(net$A, "comb"))
errs <- numeric(length(times))
for (s in seq_along(times)) {
  rho_lin <- rho_theory(diff_dist(dmap(sp, times[s], K, SIGMA)))
  errs[s] <- sqrt(mean((rho_sim[[s]] - rho_lin)^2))
}
err_df <- data.frame(t = times, rms_error = errs)

saveRDS(list(
  net = net,
  show_at = show_at,
  times = times,
  diss = diss,
  zl = zl,
  err_df = err_df
), file.path("..", "data", "T2_res.rds"))

write.csv(err_df, file.path("..", "data", "T2_validation.csv"), row.names = FALSE)
cat("[T2] computation done -> data/T2_res.rds\n")
