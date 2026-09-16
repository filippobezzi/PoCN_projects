# ============================================================================
#  90_optional_detectability.R      *** OPTIONAL -- NOT PART OF THE PROJECT ***
#
#  Run only if you have time and want an EXTERNAL benchmark for the z_out
#  sweep.  It is not needed to read F8, and nothing in the report depends on it.
#
#      Rscript 90_optional_detectability.R
#
#  ---------------------------------------------------------------------------
#  What it adds, in one paragraph
#
#  F8 is an INTERNAL comparison: plateau vs gap vs NMI, all measured by us, all
#  on the same networks.  It establishes an ordering (plateau fails first) but
#  it cannot say whether the last of them, NMI, fails "early" or "on time",
#  because there is nothing to compare it to.
#
#  The detectability threshold supplies that missing reference.  For a block
#  model with q groups and mean degree c there is a sharp point below which NO
#  algorithm at all can beat random guessing as N -> infinity -- the community
#  labels exist, but the realised graph carries no usable trace of them.  For
#  the standard model this is the Kesten-Stigum point
#
#        | c_in - c_out |  =  q sqrt(c) ,
#
#  in the convention where c_in / q is a node's expected number of neighbours
#  in its own group and c_out / q the expected number in each other group.
#
#  With q = 4, c = 18, c_in = 4 (18 - z_out), c_out = 4 z_out / 3, the threshold
#  sits at z_out ~ 10.3.  Compare that with F8: the plateau dies at ~2, the gap
#  at ~3-4, recovery at ~7.  So the paper's method needs roughly five times less
#  inter-community leakage than the information-theoretic limit allows.
#
#  CAVEATS, both important, state both if you use this:
#    - Kesten-Stigum is an N -> infinity result.  At N = 256 the transition is
#      smeared and shifted to smaller z_out; our measured collapse at ~7 rather
#      than 10.3 is consistent with that, not evidence against the bound.
#    - It is derived for a SINGLE-level block model.  Ours is two-level, so
#      applying it to the coarse level is an approximation.
#  Plot it as a reference line, never as a prediction.
#
#  Reference: A. Decelle, F. Krzakala, C. Moore, L. Zdeborova,
#  Phys. Rev. E 84, 066106 (2011).
#
#  Disclaimer: AI was used for generating efficient code, 
#  suggesting numerical improvements, and designing plotting functions. 
# ============================================================================

source("00_common.R")

q    <- 4
cbar <- 18

ks_gap <- function(zout) {
  c_in  <- q * (18 - zout)          # affinity, own group
  c_out <- q * zout / (q - 1)       # affinity, each other group
  c_in - c_out
}
ks_rhs    <- q * sqrt(cbar)
zout_star <- uniroot(function(z) ks_gap(z) - ks_rhs, c(0.1, 17))$root

cat(sprintf("\n[optional] Kesten-Stigum threshold:  z_out* = %.2f\n", zout_star))
cat(sprintf("[optional] measured in F8: plateau dies ~2, gap ~3-4, NMI ~7\n"))
cat(sprintf("[optional] => the method needs ~%.0fx less leakage than the limit allows\n\n",
            zout_star / 2))

zs <- seq(0.5, 16, by = 0.25)
open_png("F11_optional_detectability.png", w = 1600, h = 1300, res = 205)
par(mar = c(4.5, 4.6, 3.0, 1))
plot(zs, ks_gap(zs), type = "l", lwd = 2.5, col = "#08519c",
     xlab = expression(z[out]), ylab = expression(c[inn] - c[out]),
     main = "Kesten-Stigum reference (optional)")
abline(h = ks_rhs, lwd = 2, lty = 2, col = "#e31a1c")
abline(v = zout_star, lty = 3, lwd = 2)
abline(v = c(2, 3.5, 7), lty = 3, col = c("#4daf4a", "#377eb8", "#984ea3"), lwd = 2)
legend("topright", bty = "n", cex = 0.75, lwd = 2,
       col = c("#08519c", "#e31a1c", "black", "#4daf4a", "#377eb8", "#984ea3"),
       lty = c(1, 2, 3, 3, 3, 3),
       legend = c("signal", expression(q*sqrt(c)),
                  sprintf("KS threshold (%.1f)", zout_star),
                  "our plateau dies (~2)", "our gap closes (~3.5)",
                  "our NMI collapses (~7)"))
dev.off()

# If T5 has been run, append the reference column to its output.
f <- file.path("..", "data", "T5_sweep.csv")
if (file.exists(f)) {
  d <- read.csv(f)
  d$ks_threshold <- zout_star
  write.csv(d, f, row.names = FALSE)
  cat("[optional] ks_threshold column appended to data/T5_sweep.csv\n")
}
cat("[optional] done -> F11\n")
