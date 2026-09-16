# ============================================================================
#  50_export_edgelists.R
#
#  Export the networks as edge lists in the format
#     node_from,node_to,weight
#  plus node metadata where the nodes carry labels.
#  (ground-truth community labels for the two levels of the hierarchy)
#
#  Disclaimer: AI was used for generating efficient code, 
#  suggesting numerical improvements, and designing plotting functions. 
# ============================================================================

source("00_common.R")
dir.create(file.path("..", "data"), showWarnings = FALSE, recursive = TRUE)

export <- function(net, stem) {
  el <- which(upper.tri(net$A) & net$A > 0, arr.ind = TRUE)
  write.csv(data.frame(node_from = el[, 1], node_to = el[, 2], weight = 1L),
            file.path("..", "data", paste0(stem, "_edges.csv")), row.names = FALSE)
  write.csv(data.frame(node = seq_len(net$N),
                       community_16 = net$low,
                       community_4  = net$high,
                       degree       = net$k),
            file.path("..", "data", paste0(stem, "_nodes.csv")), row.names = FALSE)
  cat(sprintf("  %-16s N=%d  E=%d  <k>=%.2f\n", stem, net$N, nrow(el), mean(net$k)))
}

cat("[export]\n")
export(make_hier(13, 4, 1, seed = 1L), "hier_13-4")
export(add_hubs(make_hier(13, 4, 1, seed = 1L), 8L, 50L, seed = 5L), "hier_13-4_hubs")
cat("[export] done\n")
