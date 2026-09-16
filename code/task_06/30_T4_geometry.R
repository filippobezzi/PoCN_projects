# ============================================================================
#  30_T4_geometry.R
#
#  T4 -- Diffusion geometry: the (t, T) plane, embedding dimension, and hub test.
#
#  Outputs:  F4_t_T_plane.png
#            F5_embedding.png
#            F6_hubs_test.png
#            ../data/T4_hubtest.csv
#
#  Disclaimer: AI was used for generating efficient code, 
#  suggesting numerical improvements, and designing plotting functions. 
# ============================================================================

source("00_common.R")
dir.create(file.path("..", "data"), showWarnings = FALSE, recursive = TRUE)

K <- 1.0
net <- make_hier(13, 4, 1, seed = 1L)
sp  <- spec(laplacian(net$A, "comb"))

ts   <- 10^seq(-2.4, 1.0, length.out = 110)   # time values
epsg <- 10^seq(-2.3, -0.3, length.out = 80)   # epsilon values (thresholds)

H  <- lapply(ts, function(t) slink_heights(diff_dist(dmap(sp, t, K, 1))))
NC <- t(sapply(H, function(h) ncomp_at(h, epsg)))

ts_p   <- 10^seq(-3.0, 1.0, length.out = 90)
epss_p <- c(0.01, 0.05, 0.10)
Tvals_p <- T_of_eps(epss_p)

H_p <- lapply(ts_p, function(t) slink_heights(diff_dist(dmap(sp, t, K, 1))))
ncomp_mat_p <- sapply(epss_p, function(e) sapply(H_p, ncomp_at, eps = e))

# Panel 3: Effective embedding dimension d_eff(t)
tv <- 10^seq(-2.4, 1.0, length.out = 200)
de <- sapply(tv, function(t) d_eff(sp$lambda, t, K))
w4  <- c(1 / (K * sp$lambda[5]),  1 / (K * sp$lambda[4]))
w16 <- c(1 / (K * sp$lambda[17]), 1 / (K * sp$lambda[16]))

brks <- c(0.9, 1.5, 2.5, 4.5, 8.5, 16.5, 32.5, 64.5, 128.5, 260)
cols <- hcl.colors(length(brks) - 1, "Spectral")

# Hub Distance Test across 3 time steps
hn <- add_hubs(make_hier(13, 4, 1, seed = 1L), nhub = 8L, extra = 50L, seed = 5L)
hubs <- hn$hubs
sp_c <- spec(laplacian(hn$A, "comb"))
emb_radius <- function(Psi) {
  P  <- Psi[, -1, drop = FALSE]
  ct <- colMeans(P)
  sqrt(rowSums((P - rep(ct, each = nrow(P)))^2))
}
gm  <- function(a, b) sqrt(a * b)
tsh <- c(gm(1 / (K * sp_c$lambda[17]), 1 / (K * sp_c$lambda[16])),
         gm(1 / (K * sp_c$lambda[5]),  1 / (K * sp_c$lambda[4])),
         5 / (K * sp_c$lambda[2]))
labels_t <- c("16-comm window", "4-comm window", "fully merged")

rad_matrix <- matrix(0, nrow = hn$N, ncol = 3)
for (j in 1:3) {
  tt  <- tsh[j]
  Psi <- dmap(sp_c, tt, K, 1)
  rr  <- emb_radius(Psi)
  rad_matrix[, j] <- rr
}

write.csv(data.frame(
  node = seq_len(hn$N),
  degree = hn$k,
  is_hub = seq_len(hn$N) %in% hubs,
  radius_t16 = rad_matrix[, 1],
  radius_t4  = rad_matrix[, 2],
  radius_tlate = rad_matrix[, 3]),
  file.path("..", "data", "T4_hubtest.csv"), row.names = FALSE)

saveRDS(list(
  ts = ts,
  epsg = epsg,
  NC = NC,
  ts_p = ts_p,
  epss_p = epss_p,
  Tvals_p = Tvals_p,
  ncomp_mat_p = ncomp_mat_p,
  tv = tv,
  de = de,
  w4 = w4,
  w16 = w16,
  brks = brks,
  cols = cols,
  hn = hn,
  hubs = hubs,
  sp_c = sp_c,
  tsh = tsh,
  labels_t = labels_t,
  rad_matrix = rad_matrix
), file.path("..", "data", "T4_geometry_res.rds"))

cat("[T4] computation done -> data/T4_geometry_res.rds\n")
