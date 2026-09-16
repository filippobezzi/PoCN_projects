# ============================================================================
#  30_T4_hub_damage.R
#
#  T4 -- What hubs actually cost.
#
#  Hubs cause failure of the method in two distinct ways.
#
#  FEW hubs -> CONTAMINATION. The plateau shifts from n_comp = 4 to
#              n_comp = 5 or 6, i.e. four communities plus a couple of hub
#              singletons. The count no longer equals the number of
#              communities, so it no longer identifies a topological scale.
#
#  MORE hubs -> BRIDGING. Because a hub sits at the centroid, it is roughly
#               equidistant from EVERY cluster. The eps at which it joins one
#               cluster is essentially the eps at which it joins all of them,
#               so under single linkage all communities merge. The plateau is destroyed.
#
#  The test sweeps the number of hubs and, for each, searches the whole
#  (t, eps) plane for the LONGEST plateau.
#
#  Disclaimer: AI was used for generating efficient code, 
#  suggesting numerical improvements, and designing plotting functions. 
# ============================================================================

source("00_common.R")
dir.create(file.path("..", "data"), showWarnings = FALSE, recursive = TRUE)

K     <- 1.0
ts    <- 10^seq(-2.4, 1.2, length.out = 70)
epsg  <- 10^seq(-2.3, -0.3, length.out = 40)
nhubs <- c(0, 2, 4, 8, 16)

base <- make_hier(13, 4, 1, seed = 1L)

#' longest run of a constant value in a vector, restricted to plausible
#' community counts, returned as (value, run length in grid steps)
best_plateau <- function(NC) {                    # input: 70x40 matrix of n_comp over (t,eps)
  best <- list(val = NA_integer_, len = 0L)
  for (j in seq_len(ncol(NC))) {
    r <- rle(NC[, j])                             # run length encoding over t for each eps: contains number of comp at each t
    ok <- which(r$values >= 2 & r$values <= 40)   # choose values close to (4-16)
    if (!length(ok)) next                         # skip if no values in the range
    k <- ok[which.max(r$lengths[ok])]             # find the longest run in the range
    if (r$lengths[k] > best$len) best <- list(val = r$values[k], len = r$lengths[k]) # update best
  }
  best
}

planes <- list(); res <- data.frame()
for (nh in nhubs) {
  net <- if (nh == 0) base else add_hubs(base, nh, 50L, seed = 5L)
  sp  <- spec(laplacian(net$A, "comb"))
  H   <- lapply(ts, function(t) slink_heights(diff_dist(dmap(sp, t, K, 1))))
  NC  <- t(sapply(H, function(h) ncomp_at(h, epsg)))
  planes[[as.character(nh)]] <- NC
  bp  <- best_plateau(NC) # the longest plateau that exists anywhere on the (t,eps) plane
  # the longest plateau at exactly 4 communities, anywhere on the plane
  len4 <- max(c(0, apply(NC, 2, function(v) { r <- rle(v); k <- which(r$values == 4)
                                              if (length(k)) max(r$lengths[k]) else 0 })))
  l4   <- sp$lambda[4]
  l5   <- sp$lambda[5]
  gap4 <- l5 / l4
  res <- rbind(res, data.frame(n_hubs = nh, mean_degree = mean(net$k),
                               lambda_4 = l4, lambda_5 = l5, gap_4 = gap4,
                               best_n = bp$val, best_len = bp$len, len_at_4 = len4))
  cat(sprintf("[T4c] %2d hubs : gap_4 = %.3f ; longest plateau n_comp = %s (%d steps) ; at n=4 : %d steps\n",
              nh, gap4, bp$val, bp$len, len4))
}

write.csv(res, file.path("..", "data", "T4c_hub_damage.csv"), row.names = FALSE)

saveRDS(list(
  ts = ts,
  epsg = epsg,
  nhubs = nhubs,
  planes = planes,
  res = res
), file.path("..", "data", "T4_hub_damage_res.rds"))

cat("[T4_hub_damage] computation done -> data/T4_hub_damage_res.rds\n")
