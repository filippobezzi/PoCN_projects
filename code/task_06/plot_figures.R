# ============================================================================
#  plot_figures.R
#
#  Unified Plotting Module for Hierarchical Kuramoto Sync Dynamics
#  Reads pre-computed data from ../data/*.rds and outputs figures to ../figures/
#
#  Disclaimer: AI was used for generating efficient code, 
#  suggesting numerical improvements, and designing plotting functions. 
# ============================================================================

source("00_common.R")

# ---------------------------------------------------------------------------
# F1: Adjacency and Synchronization Dissimilarity Heatmaps
# ---------------------------------------------------------------------------
plot_F1 <- function() {
  res_file <- file.path("..", "data", "T2_res.rds")
  if (!file.exists(res_file)) stop("Missing data: run 10_T2_validation.R first")
  d <- readRDS(res_file)
  
  show_at <- if (!is.null(d$show_at)) d$show_at else c(15, 20, 26)
  times   <- d$times
  diss    <- if (!is.null(d$diss)) d$diss else lapply(show_at, function(s) log10(pmax(1 - d$rho_sim[[s]], 1e-16)))
  zl      <- if (!is.null(d$zl)) d$zl else range(unlist(diss), finite = TRUE)
  
  open_png("F1_rho_heatmaps.png", w = 2100, h = 680, res = 300, pointsize = 11)
  layout(matrix(c(1, 2, 3, 4, 5, 5, 5, 5), nrow = 2, byrow = TRUE), heights = c(1, 0.28))
  par(mar = c(2.8, 2.8, 1.2, 0.8), family = "serif")
  
  # Panel 1: True structure (Adjacency Matrix A)
  mat_image(d$net$A, col = c("white", "black"), zlim = c(0, 1))
  mtext("Adjacency A", side = 1, line = 1.3, cex = 1.0)
  mtext("node (ordered)", side = 2, line = 1.2, cex = 1.0)
  abline(v = c(64, 128, 192) + 0.5, col = "grey30", lwd = 0.5)
  abline(h = 256 - c(64, 128, 192) + 0.5, col = "grey30", lwd = 0.5)
  
  # Panels 2-4: Synchronization heatmaps
  for (ii in seq_along(show_at)) {
    mat_image(diss[[ii]], col = pal_seq(64), zlim = zl)
    mtext(sprintf("t = %.3f", times[show_at[ii]]), side = 1, line = 1.3, cex = 1.0)
    abline(v = c(64, 128, 192) + 0.5, col = "grey30", lwd = 0.5)
    abline(h = 256 - c(64, 128, 192) + 0.5, col = "grey30", lwd = 0.5)
  }
  color_bar(zl, pal_seq(64))
  dev.off()
  cat("[PLOT] F1_rho_heatmaps.png generated\n")
}

# ---------------------------------------------------------------------------
# F2: Spectrum vs Null Model & IPR Localisation
# ---------------------------------------------------------------------------
plot_F2 <- function() {
  res_file <- file.path("..", "data", "T3_res.rds")
  if (!file.exists(res_file)) stop("Missing data: run 20_T3_spectrum.R first")
  d <- readRDS(res_file)
  
  open_png("F2_spectrum.png", w = 2100, h = 800, res = 300, pointsize = 11)
  par(mfrow = c(1, 2), mar = c(4.2, 4.4, 1.5, 1.2), cex.axis = 1.0, cex.lab = 1.0, family = "serif")
  
  # Left: Spectral density
  br <- seq(0, max(c(d$sp$lambda, d$spn$lambda)) * 1.02, length.out = 45)
  hn <- hist(d$spn$lambda, breaks = br, plot = FALSE)
  hr <- hist(d$sp$lambda,  breaks = br, plot = FALSE)
  
  plot(hn$mids, hn$density, type = "s", lwd = 2, col = "grey55",
       xlab = expression(lambda), ylab = "density",
       ylim = c(0, max(c(hn$density, hr$density))))
  lines(hr$mids, hr$density, type = "s", lwd = 2, col = "#e31a1c")
  abline(v = d$bulk_edge, lty = 2, col = "black")
  rug(d$sp$lambda[d$sp$lambda < d$bulk_edge & d$sp$lambda > 1e-8], col = "#e31a1c", lwd = 2)
  legend("topleft", bty = "n", cex = 0.9, lwd = 2,
         col = c("#e31a1c", "grey55"), lty = c(1, 1),
         legend = c("13-4", "null"))
  
  # Right: Localisation (IPR per mode)
  plot(seq_along(d$sp$lambda), d$ipr_vals, log = "y", pch = 16, cex = 0.8,
       col = ifelse(d$sp$lambda < d$bulk_edge, "#e31a1c", "grey50"),
       xlab = "eigenvalue rank", ylab = "IPR")
  abline(h = 1 / 256, lty = 3, col = "grey30")
  text(210, 1 / 256, "1/N (delocalised)", pos = 3, cex = 0.85, col = "grey30")
  legend("top", bty = "n", cex = 0.9, pch = 16, horiz = TRUE,
         col = c("#e31a1c", "grey50"),
         legend = c("structural", "bulk"))
  
  dev.off()
  cat("[PLOT] F2_spectrum.png generated\n")
}

# ---------------------------------------------------------------------------
# F3: Spectrum vs Eigenvectors (Lifetimes, Matrix Heatmap, Subscripts v_2, v_8, v_120)
# ---------------------------------------------------------------------------
plot_F3 <- function() {
  res_file <- file.path("..", "data", "T3_res.rds")
  if (!file.exists(res_file)) stop("Missing data: run 20_T3_spectrum.R first")
  d <- readRDS(res_file)
  
  open_png("F3_eigenvectors.png", w = 2100, h = 850, res = 300, pointsize = 11)
  layout(matrix(c(1, 2, 3, 1, 2, 4, 1, 2, 5), nrow = 3, byrow = TRUE),
         widths = c(1.0, 1.7, 1.2))
  
  # Panel 1 (Left): Mode Lifetimes
  par(mar = c(4.2, 4.4, 1.5, 1.0), cex.axis = 1.0, cex.lab = 1.0, family = "serif")
  plot(1 / (d$K * d$nz), d$ranks, log = "xy", pch = 16, cex = 0.8, col = "#2b2926",
       xlab = expression(1 / (K * lambda[i])~~"(mode lifetime)"),
       ylab = "rank i")
  rect(1 / (d$K * d$sp$lambda[5]), 0.8, 1 / (d$K * d$sp$lambda[4]), 4.2, col = adjustcolor("#c15f3c", 0.2), border = NA)
  rect(1 / (d$K * d$sp$lambda[17]), 4.8, 1 / (d$K * d$sp$lambda[16]), 16.2, col = adjustcolor("#8a8580", 0.25), border = NA)
  points(1 / (d$K * d$nz), d$ranks, pch = 16, cex = 0.8, col = "#2b2926")
  
  text(1 / (d$K * d$sp$lambda[3]), 2.2, "ranks 2-4\n(4)", cex = 0.85, col = "#c15f3c", pos = 2)
  text(1 / (d$K * d$sp$lambda[10]), 9.0, "ranks 5-16\n(16)", cex = 0.85, col = "#4f4b46", pos = 2)
  
  # Panel 2 (Center): Eigenvector Heatmap Matrix
  par(mar = c(4.2, 4.4, 1.5, 1.5), cex.axis = 1.0, cex.lab = 1.0, family = "serif")
  image(seq(2, d$M + 1), seq_len(256), t(d$W[256:1, ]), col = pal_div(64), zlim = c(-1, 1),
        xlab = expression(mode~index~~alpha), ylab = "node (ordered)", axes = FALSE)
  axis(1, at = c(2, 5, 10, 16, 20, 25, 30))
  axis(2, at = c(1, 64, 128, 192, 256), labels = c(1, 64, 128, 192, 256))
  box()
  abline(h = 256 - c(64, 128, 192) + 0.5, col = "black", lwd = 1.2)
  abline(h = 256 - seq(16, 240, by = 16) + 0.5, col = adjustcolor("black", 0.4), lwd = 0.5)
  abline(v = c(4.5, 16.5), col = "#c15f3c", lwd = 1.8)
  
  # Panel 3 (Right): Individual profiles
  labs_c <- list("(4)",
                 "(16)",
                 "(bulk)")
  ylabs_c <- list(expression(v[2]), expression(v[8]), expression(v[120]))
  
  for (idx in 1:3) {
    par(mar = c(if (idx == 3) 4.2 else 1.2, 4.4, if (idx == 1) 1.5 else 0.5, 1.0), cex.axis = 1.0, cex.lab = 1.0, family = "serif")
    v_col <- d$sp$V[, d$modes_c[idx]]
    v_col <- v_col / max(abs(v_col))
    plot(seq_len(256), v_col, type = "l", col = d$cols_c[idx], lwd = 1.2,
         ylim = c(-1.15, 1.15), xlim = c(1, 256), axes = FALSE,
         xlab = if (idx == 3) "node index (ordered)" else "",
         ylab = ylabs_c[[idx]])
    axis(2, at = c(-1, 0, 1))
    if (idx == 3) axis(1, at = c(1, 64, 128, 192, 256))
    box()
    abline(v = c(64, 128, 192) + 0.5, col = "grey40", lwd = 0.8)
    if (d$modes_c[idx] == 8) abline(v = seq(16, 240, by = 16) + 0.5, col = "grey75", lwd = 0.4)
    legend("topright", bty = "n", cex = 0.85, legend = labs_c[[idx]], text.col = d$cols_c[idx])
  }
  
  dev.off()
  cat("[PLOT] F3_eigenvectors.png generated\n")
}

# ---------------------------------------------------------------------------
# F4: Combined Plateaus, (t, eps) Plane, and Embedding Dimension
# ---------------------------------------------------------------------------
plot_F4 <- function() {
  res_file <- file.path("..", "data", "T4_geometry_res.rds")
  if (!file.exists(res_file)) stop("Missing data: run 30_T4_geometry.R first")
  d <- readRDS(res_file)
  
  open_png("F3_F4_F5_plateaus_geometry.png", w = 2100, h = 900, res = 300, pointsize = 11)
  layout(matrix(c(1, 2, 3, 4, 4, 4), nrow = 2, byrow = TRUE), heights = c(1, 0.25))
  
  # Panel 1 (Left): n_comp(t) plateaus
  par(mar = c(4.2, 4.4, 1.5, 1.2), cex.axis = 0.9, cex.lab = 1.0, family = "serif")
  plot(NA, xlim = range(d$ts_p), ylim = c(1, 300), log = "xy",
       xlab = "t", ylab = "number of components")
  rect(d$w4[1],  0.5, d$w4[2],  400, col = "#cfe8f3", border = NA)
  rect(d$w16[1], 0.5, d$w16[2], 400, col = "#fde3cf", border = NA)
  box()
  cols_p <- c("#08519c", "#e31a1c", "#4daf4a")
  for (j in seq_along(d$epss_p)) lines(d$ts_p, d$ncomp_mat_p[, j], lwd = 2, col = cols_p[j])
  abline(h = c(4, 16), lty = 3, col = "grey30")
  legend("bottomleft", bty = "n", cex = 0.85, lwd = 1.5, col = cols_p,
         legend = parse(text = sprintf("epsilon == '%.2f'", d$epss_p)))
  legend("topright", bty = "n", cex = 0.85, fill = c("#cfe8f3", "#fde3cf"),
         border = NA, legend = c("4", "16"))
  
  # Panel 2 (Center): (t, eps) plane contour map
  par(mar = c(4.2, 4.4, 1.5, 1.4), cex.axis = 0.9, cex.lab = 1.0, family = "serif")
  image(log10(d$ts), log10(d$epsg), d$NC, breaks = d$brks, col = d$cols,
        xlab = expression(log[10]~t), ylab = expression(log[10]~epsilon))
  box()
  contour(log10(d$ts), log10(d$epsg), d$NC, levels = 4,
          add = TRUE, lwd = 2.2, lty = 1, col = "black", drawlabels = FALSE)
  contour(log10(d$ts), log10(d$epsg), d$NC, levels = 16,
          add = TRUE, lwd = 2.2, lty = 2, col = "black", drawlabels = FALSE)
  legend("topright", bty = "n", cex = 0.85, lty = c(1, 2), lwd = 2.2, col = c("black", "black"),
         legend = c("4", "16"))
  
  # Panel 3 (Right): Effective embedding dimension d_eff(t)
  par(mar = c(4.2, 4.4, 1.5, 1.4), cex.axis = 0.9, cex.lab = 1.0, family = "serif")
  plot(d$tv, d$de, log = "xy", type = "l", lwd = 2, col = "#e31a1c",
       xlab = "t", ylab = expression(d[eff](t)))
  rect(d$w4[1], 0.5, d$w4[2], 400, col = adjustcolor("#cfe8f3", 0.4), border = NA)
  rect(d$w16[1], 0.5, d$w16[2], 400, col = adjustcolor("#fde3cf", 0.5), border = NA)
  lines(d$tv, d$de, lwd = 2, col = "#e31a1c")
  abline(h = c(3, 15), lty = 3, col = "grey30")
  axis(4, at = c(3, 15), labels = c("3", "15"), cex.axis = 0.85)
  legend("topright", bty = "n", cex = 0.85, fill = c("#cfe8f3", "#fde3cf"),
         border = NA, legend = c("4", "16"))
  
  # Bottom colorbar for Panel 2 (Center)
  par(mar = c(1.8, 4.4, 0.2, 4.4), family = "serif")
  image(seq_along(d$cols), 1, matrix(seq_along(d$cols), ncol = 1), col = d$cols,
        axes = FALSE, xlab = "", ylab = "")
  axis(1, at = seq_along(d$cols), cex.axis = 0.95, mgp = c(1.5, 0.4, 0),
       labels = c("1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", ">128"))
  box()
  dev.off()
  cat("[PLOT] F3_F4_F5_plateaus_geometry.png generated\n")
}

# ---------------------------------------------------------------------------
# F6: Hub Distance Test Across 3 Time Steps
# ---------------------------------------------------------------------------
plot_F6 <- function() {
  res_file <- file.path("..", "data", "T4_geometry_res.rds")
  if (!file.exists(res_file)) stop("Missing data: run 30_T4_geometry.R first")
  d <- readRDS(res_file)
  
  ncomps <- c("16", "4", "1")
  open_png("F6_hubs_test.png", w = 2100, h = 750, res = 300, pointsize = 11)
  par(mfrow = c(1, 3), mar = c(4.2, 4.4, 1.5, 1.0), cex.axis = 0.95, cex.lab = 1.0, family = "serif")
  for (j in 1:3) {
    tt  <- d$tsh[j]
    rr  <- d$rad_matrix[, j]
    plot(d$hn$k, rr, log = "xy", pch = 16, cex = 0.8, col = "grey45", xlab = expression(k[i]), ylab = "distance from centroid")
    points(d$hn$k[d$hubs], rr[d$hubs], pch = 16, cex = 1.3, col = "#e31a1c")
    legend("bottomleft", bty = "n", cex = 0.85,
           legend = c(sprintf("t = %.3f (%s)", tt, ncomps[j]),
                      "hubs"),
           pch = c(NA, 16), col = c("black", "#e31a1c"))
  }
  dev.off()
  cat("[PLOT] F6_hubs_test.png generated\n")
}

# ---------------------------------------------------------------------------
# F7: Hub Damage and Plateau Destruction
# ---------------------------------------------------------------------------
plot_F7 <- function() {
  res_file <- file.path("..", "data", "T4_hub_damage_res.rds")
  if (!file.exists(res_file)) stop("Missing data: run 30_T4_hub_damage.R first")
  d <- readRDS(res_file)
  
  open_png("F7_hub_damage.png", w = 2100, h = 750, res = 300, pointsize = 11)
  par(mfrow = c(1, 3), mar = c(4.2, 4.4, 1.5, 1.0), cex.axis = 0.95, cex.lab = 1.0, family = "serif")
  
  # Panel 1 (Left): Spectral gap ratio vs number of hubs
  plot(d$res$n_hubs, d$res$gap_4, type = "b", pch = 16, lwd = 2, col = "#08519c",
       xlim = c(-1, 17), ylim = c(1, 4),
       xlab = "number of hubs added",
       ylab = expression(spectral~gap~ratio~~lambda[5]/lambda[4]))
  text(d$res$n_hubs, d$res$gap_4, labels = sprintf("%.2f", d$res$gap_4),
       pos = 3, cex = 0.85, xpd = TRUE)
  abline(h = 1, lty = 3, col = "grey40")
  
  # Panel 2 (Center): (t, eps) plane after adding 8 hubs
  brks <- c(0.9, 1.5, 2.5, 4.5, 8.5, 16.5, 32.5, 64.5, 128.5, 260)
  cols <- hcl.colors(length(brks) - 1, "Spectral")
  image(log10(d$ts), log10(d$epsg), d$planes[["8"]], breaks = brks, col = cols,
        xlab = expression(log[10]~t), ylab = expression(log[10]~epsilon))
  box()
  contour(log10(d$ts), log10(d$epsg), d$planes[["8"]], levels = 4,
          add = TRUE, lwd = 2.2, lty = 1, col = "black", drawlabels = FALSE)
  contour(log10(d$ts), log10(d$epsg), d$planes[["8"]], levels = 16,
          add = TRUE, lwd = 2.2, lty = 2, col = "black", drawlabels = FALSE)
  legend("topright", bty = "n", cex = 0.85, lty = c(1, 2), lwd = 2.2, col = c("black", "black"),
         legend = c("4", "16"))
  title(main = "8 hubs added", cex.main = 1.0)

  # Panel 3 (Right): Plateau duration at 4 communities vs number of hubs
  plot(d$res$n_hubs, d$res$len_at_4, type = "b", pch = 16, lwd = 2, col = "#e31a1c",
       xlim = c(-1, 17), ylim = c(0, 12),
       xlab = "number of hubs added",
       ylab = "longest 4-community plateau (steps)")
  text(d$res$n_hubs, d$res$len_at_4, labels = sprintf("n=%s", d$res$best_n),
       pos = 3, cex = 0.85, xpd = TRUE)
  dev.off()
  cat("[PLOT] F7_hub_damage.png generated\n")
}

# ---------------------------------------------------------------------------
# F8: Structural Noise Sweep (z_out Sweep)
# ---------------------------------------------------------------------------
plot_F8 <- function() {
  res_file <- file.path("..", "data", "T5_res.rds")
  if (!file.exists(res_file)) stop("Missing data: run 40_T5_zout_sweep.R first")
  d <- readRDS(res_file)
  
  mrow <- function(M) rowMeans(M, na.rm = TRUE)
  srow <- function(M) apply(M, 1, sd, na.rm = TRUE)
  
  open_png("F8_zout_sweep.png", w = 2100, h = 750, res = 300, pointsize = 11)
  par(mfrow = c(1, 3), mar = c(4.2, 4.4, 1.5, 1.0), cex.axis = 0.95, cex.lab = 1.0, family = "serif")
  
  # Left: Structural eigenvalues vs bulk null model
  plot(d$zouts, mrow(d$lamc), type = "b", pch = 16, lwd = 2, col = "#e31a1c",
       ylim = c(0, max(mrow(d$bulk), na.rm = TRUE) * 1.1),
       xlab = expression(z[out]), ylab = expression(lambda))
  lines(d$zouts, mrow(d$lamf), type = "b", pch = 17, lwd = 2, col = "#ff7f00")
  lines(d$zouts, mrow(d$bulk), type = "b", pch = 1, lwd = 2, col = "grey40", lty = 2)
  legend("bottomright", bty = "n", cex = 0.85, lwd = 2,
         col = c("#e31a1c", "#ff7f00", "grey40"),
         lty = c(1, 1, 2), pch = c(16, 17, 1),
         legend = c(expression(lambda[4]~~"(meso)"),
                    expression(lambda[5]~~"(micro)"),
                    expression(lambda[bulk]~~"(null)")))
  
  # Center: Spectral gap ratio
  m <- mrow(d$gap); s <- srow(d$gap)
  plot(d$zouts, m, type = "b", pch = 16, lwd = 2, col = "#377eb8",
       ylim = c(1, max(m + s)), xlab = expression(z[out]),
       ylab = expression(lambda[5] / lambda[4]))
  arrows(d$zouts, m - s, d$zouts, m + s, angle = 90, code = 3, length = 0.02, col = "#377eb8")
  abline(h = 1, lty = 3)
  
  # Right: Scale separation vs recovery (NMI)
  par(mar = c(4.2, 4.4, 1.5, 1.0), cex.axis = 0.95, cex.lab = 1.0, family = "serif")
  m1 <- mrow(d$plat); m2 <- mrow(d$nmiv); m3 <- mrow(d$nmi6)
  m1 <- m1 / max(m1, 1e-9)
  plot(d$zouts, m1, type = "b", pch = 16, lwd = 2, col = "#4daf4a",
       ylim = c(0, 1.05), xlab = expression(z[out]),
       ylab = "plateau width / NMI")
  lines(d$zouts, m2, type = "b", pch = 17, lwd = 2, col = "#984ea3")
  lines(d$zouts, m3, type = "b", pch = 15, lwd = 2, col = "#ff7f00")
  legend("topright", bty = "n", cex = 0.85, lwd = 2, pch = c(16, 17, 15),
         col = c("#4daf4a", "#984ea3", "#ff7f00"),
         legend = c("plateau width (4)", "NMI (4)", "NMI (16)"))
  dev.off()
  cat("[PLOT] F8_zout_sweep.png generated\n")
}

# ---------------------------------------------------------------------------
# Master plot function
# ---------------------------------------------------------------------------
plot_all <- function() {
  cat("[PLOT] Generating all figures...\n")
  plot_F1()
  plot_F2()
  plot_F3()
  plot_F4()
  plot_F6()
  plot_F7()
  plot_F8()
  cat("[PLOT] All figures successfully updated in ../figures/\n")
}

# Allow command line execution: Rscript plot_figures.R [F1|F2|F3|F4|F6|F7|F8|all]
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  target <- tolower(args[1])
  if (target %in% c("f1", "f1_rho_heatmaps")) plot_F1()
  else if (target %in% c("f2", "f2_spectrum")) plot_F2()
  else if (target %in% c("f3", "f3_eigenvectors")) plot_F3()
  else if (target %in% c("f4", "f3_f4_f5_plateaus_geometry")) plot_F4()
  else if (target %in% c("f6", "f6_hubs_test")) plot_F6()
  else if (target %in% c("f7", "f7_hub_damage")) plot_F7()
  else if (target %in% c("f8", "f8_zout_sweep")) plot_F8()
  else if (target == "all") plot_all()
  else stop("Unknown plot target: ", args[1])
}
