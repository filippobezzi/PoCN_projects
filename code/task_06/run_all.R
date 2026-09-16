# ============================================================================
#  run_all.R  --  run the whole pipeline
#
#  Usage, from inside this directory:
#      Rscript run_all.R
#  or, from an interactive session:
#      setwd("src_R"); source("run_all.R")
#
#  Base R only.  No packages required.
#  Total runtime on a laptop:  ~5 min with QUICK = FALSE in 10_T2, ~1 min with
#  QUICK = TRUE.  T3/T4/T5 are pure linear algebra and take seconds.
#
#  Figures are written to ../figures/ , numerical results to ../data/ .
#
#  Disclaimer: AI was used for generating efficient code, 
#  suggesting numerical improvements, and designing plotting functions. 
# ============================================================================

t0 <- Sys.time()
scripts <- c("10_T2_validation.R",
             "20_T3_spectrum.R",
             "30_T4_geometry.R",
             "30_T4_hub_damage.R",
             "40_T5_zout_sweep.R",
             "50_export_edgelists.R")

# 90_optional_detectability.R is deliberately NOT in this list.  It is an
# optional external benchmark; run it by hand if you want it:
#     Rscript 90_optional_detectability.R

for (s in scripts) {
  cat("\n=====================================================\n")
  cat("  ", s, "\n")
  cat("=====================================================\n")
  source(s, echo = FALSE)
}

cat("\n=====================================================\n")
cat("   Rendering all plots via plot_figures.R\n")
cat("=====================================================\n")
source("plot_figures.R")
plot_all()

cat(sprintf("\nAll done in %.1f min.\n",
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))
cat("Figures:\n"); print(list.files(file.path("..", "figures"), pattern = "^F"))
