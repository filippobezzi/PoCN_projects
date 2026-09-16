# ============================================================================
#  00_common.R
#  Re-derived and extension of paper: Synchronisation as diffusion geometry (Arenas, Diaz-Guilera, Perez-Vicente,
#  PRL 96, 114102 (2006)).
#  
#  Disclaimer: AI was used for generating efficient code, 
#  suggesting numerical improvements, and designing plotting functions. 
# ============================================================================

# ---------------------------------------------------------------------------
# 1. NETWORKS
# ---------------------------------------------------------------------------

#' Two-level hierarchical network of the paper.
#'
#' N = 256 nodes, 16 first-level communities of 16 nodes, grouped 4-into-1
#' into 4 second-level communities of 64.  Each node has, in expectation,
#'   zin1 edges inside its own small community  (16-1=15 possible candidates)
#'   zin2 edges to the rest of its large community (16x3=48 possible candidates)
#'   zout edges to the rest of the network        (64x3=192 possible candidates)
#' so that the expected degree is zin1 + zin2 + zout.

make_hier <- function(zin1 = 13, zin2 = 4, zout = 1, seed = 1L) {
  N <- 256L; nin1 <- 16L; nin2 <- 4L
  in1 <- rep(seq_len(16L), each = 16L)   # first-level label,  1..16
  in2   <- rep(seq_len(4L),  each = 64L)   # second-level label, 1..4

  p1 <- zin1 / 15
  p2 <- zin2 / 48
  p3 <- zout / 192

  same_in1 <- outer(in1, in1, "==")
  same_in2 <- outer(in2, in2, "==")
  P <- ifelse(same_in1, p1, ifelse(same_in2, p2, p3)) # final 256x256 probability matrix

  set.seed(seed)
  U <- matrix(runif(N * N), N, N)
  U[lower.tri(U, diag = TRUE)] <- 1        # only upper triangle relevant to generate A
  A <- (U < P) * 1
  A <- A + t(A)                            # symmetrise
  diag(A) <- 0

  degree_sequence <- list(A = A, low = in1, high = in2, k = rowSums(A), N = N,
                          label = sprintf("%g-%g", zin1, zin2))
  return(degree_sequence)
}

#' ---------------------------------------------------------------------------
#' 1.1 Add hubs to an existing hierarchical network.
#''
#'' CONTROLLED perturbation, and the control is the point. 
#'' Mantain the planted community structure
#'' attach `extra` new links from each of `nhub` chosen nodes
#'' to uniformly random nodes across the network.
#'' Added degree per hub < 50.

add_hubs <- function(net, nhub = 8L, extra = 50L, seed = 5L) {
  set.seed(seed)
  A <- net$A; N <- net$N
  per_high <- nhub %/% 4L
  hubs <- integer(0)
  for (b in 1:4) {
    hubs <- c(hubs, sample(which(net$high == b), per_high))
  }
  for (h in hubs) {
    hub_neighbors <- sample(setdiff(seq_len(N), h), extra)
    A[h, hub_neighbors] <- 1
    A[hub_neighbors, h] <- 1
  }
  net$A <- A; net$k <- rowSums(A); net$hubs <- hubs
  net$label <- sprintf("%s + %d hubs", net$label, nhub)
  return(net)
}

#' ---------------------------------------------------------------------------
#' 1.2 Double Edge Swap (the null model).
#'
#' It destroys every correlation in the graph EXCEPT the degree sequence.  
#' Whatever survives in the previous spectrum but not here is structure; 
#' whatever is present in both is "bulk" - uninformative noise.

config_null <- function(A, nswap_per_edge = 20L, seed = 11L) {
  set.seed(seed)
  N <- nrow(A)
  el <- which(upper.tri(A) & A > 0, arr.ind = TRUE)   # edge list
  E  <- nrow(el)
  Ad <- A
  nswap <- nswap_per_edge * E
  for (s in seq_len(nswap)) {
    e1 <- sample.int(E, 1L); e2 <- sample.int(E, 1L)  # randomly select 2 edges
    if (e1 == e2) next                                # edges must be different
    a <- el[e1, 1]; b <- el[e1, 2]                    # first edge
    c <- el[e2, 1]; d <- el[e2, 2]                    # second edge
    if (length(unique(c(a, b, c, d))) < 4L) next      # no shared endpoints
    if (Ad[a, d] > 0 || Ad[c, b] > 0) next            # no multi-edges
    # Swap edges
    Ad[a, b] <- 0; Ad[b, a] <- 0                      # remove old edges
    Ad[c, d] <- 0; Ad[d, c] <- 0
    Ad[a, d] <- 1; Ad[d, a] <- 1                      # add new edges
    Ad[c, b] <- 1; Ad[b, c] <- 1
    el[e1, ] <- c(a, d)                               # update edge list
    el[e2, ] <- c(c, b)
  }
  return(Ad)  # degree sequence preserved, but local structure destroyed
}

# ---------------------------------------------------------------------------
# 2. LAPLACIAN AND SPECTRUM
# ---------------------------------------------------------------------------

#' Laplacian matrix:
#'   type = "comb" : L    = D - A            (standard Laplacian)
#'   type = "rw"   : L_rw = I - D^{-1} A     (normalised - asymmetric)
#'
#' L_rw is not symmetric, but it is similar to the symmetric quantum Laplacian
#' L_sym = I - D^{-1/2} A D^{-1/2}; diagonalise L_sym and transform back (faster and safer).

laplacian <- function(A, type = c("comb", "rw")) {
  type <- match.arg(type)                               # if no type, default to "comb"
  k <- rowSums(A)
  if (type == "comb") return(diag(k) - A)
  ks <- 1 / sqrt(pmax(k, 1e-12))
  diag(nrow(A)) - (ks * A) * rep(ks, each = nrow(A))     # = L_sym
}

#' Eigendecomposition, eigenvalues ASCENDING.

#' For type = "rw" the returned vectors v are those of L_sym; the corresponding
#' L_rw right-eigenvectors are D^{-1/2} v.

spec <- function(L) {
  e <- eigen(L, symmetric = TRUE)
  ord <- order(e$values)
  sp <- list(lambda = pmax(e$values[ord], 0), V = e$vectors[, ord, drop = FALSE])
  return(sp)  # sp$lambda: N; sp$V: N x N
}

# ---------------------------------------------------------------------------
#'' 2.1 Inverse participation ratio of each eigenvector: sum_i v_i^4.
#'' Diagnostic tool to separate the effect of communities from the effect of hubs.

#'' Communities imply modes spread out over all nodes in the community
#'' Hubs correspond to modes localized on themselves and nearest neighbours.

#'' IPR ~ 1/N  => the mode is spread over the whole network (delocalised) - community detection.
#'' IPR ~ 1    => the mode lives on one node (localised) - hub detection.
ipr <- function(V) colSums(V^4)

# ---------------------------------------------------------------------------
# 3. DIFFUSION GEOMETRY
# ---------------------------------------------------------------------------

#' Diffusion map:  Psi_t(i) = sigma * ( e^{-K lambda_a t} v_a(i) )_a
#' where a is the eigenvalue index, from 1 to N.

dmap <- function(sp, t, K = 1, sigma = 1) {
  sweep(sp$V, MARGIN = 2, FUN = "*", STATS = sigma * exp(-K * sp$lambda * t))
}

dmap_rw <- function(sp, k, t, K = 1, sigma = 1) {
  V <- sp$V / sqrt(pmax(k, 1e-12)) # L_sym eigvec -> L_rw eigvec
  sweep(V, MARGIN = 2, FUN = "*", STATS = sigma * exp(-K * sp$lambda * t))
}

#' Diffusion distance matrix:  d_t(i,j) = || Psi_t(i) - Psi_t(j) ||_2
#' ||Psi(i) - Psi(j)||^2 = ||Psi(i)||^2 + ||Psi(j)||^2 - 2 <Psi(i), Psi(j)>
#' is computed through the Gram matrix expansion:
#'   d_t(i,j)^2 = g_i + g_j - 2 <Psi_t(i), Psi_t(j) >.
#' This is O(N^3) once instead of O(N^2 M).

diff_dist <- function(Psi) {
  G  <- tcrossprod(Psi)
  g  <- diag(G)
  d2 <- outer(g, g, "+") - 2 * G
  return(sqrt(pmax(d2, 0)))
}

#' The local order parameter:  rho_ij(t) = exp( -1/2 d_t(i,j)^2 )
rho_theory <- function(dmat) exp(-0.5 * dmat^2)

#' Inverse map: distance implied by a measured rho.  Clipped for safety.
rho_to_dist <- function(rho) sqrt(pmax(-2 * log(pmin(pmax(rho, 1e-300), 1)), 0))

#' Threshold T on rho  <->  ball radius eps on d.
eps_of_T <- function(T) sqrt(-2 * log(T))
T_of_eps <- function(eps) exp(-0.5 * eps^2)

# ---------------------------------------------------------------------------
# 4. THE DYNAMIC CONNECTIVITY MATRIX
# ---------------------------------------------------------------------------

#' Sorted single-linkage hierarchical clustering algorithm
#' on the diffusion distance matrix, recording the distance (eps) at which nodes are merged.
#'
#' It provides a faster method to create a (t, eps) heatmap.
#' The number of connected components of the eps-neighbourhood graph  1[d_ij < eps]
#' is equivalent to  N - #{ merge heights < eps }.

slink_heights <- function(dmat) {
  hl <- sort(hclust(as.dist(dmat), method = "single")$height)
  return(hl)
}

#' Component count of D_t(T) at radius eps, from the merge heights.
#' findInterval() counts the merge heights < eps.

ncomp_at <- function(heights, eps) {
  length(heights) + 1L - findInterval(eps, heights)
}

#' Effective embedding dimension:
#' Number of degrees of freedom at time t.
#' Calculated as the exponential of the spectral entropy of the mode weights.
#' Continuous proxy for the number of communities.

d_eff <- function(lambda, t, K = 1) {
  w <- exp(-2 * K * lambda[-1] * t) # drop trivial lambda_1 = 0
  s <- sum(w)
  if (s <= 0) return(1)
  p <- w / s
  p <- p[p > 0]                     # safety
  exp(-sum(p * log(p)))             # exp(Shannon entropy) = effective number of states
}

# ---------------------------------------------------------------------------
# 5. KURAMOTO INTEGRATOR
# ---------------------------------------------------------------------------

#' d(theta_i)/dt = K sum_j A_ij sin(theta_j - theta_i)
#' Vectorized over M replicas.
#'
#' Th is M x N.  Expanding sin(theta_j - theta_i) into
#' sin(t_j)cos(t_i) - cos(t_j)sin(t_i) turns the double sum into two
#' matrix products
#' d(theta_i)/dt = K * [ (cos(theta_i) * sum_j A_ij * sin(theta_j)) - (sin(theta_i) * sum_j A_ij * cos(theta_j)) ]

kur_deriv <- function(Th, A, K) {
  S <- sin(Th); C <- cos(Th)                                       # replica by row, node by column
  Th_dot <- K * (C * (S %*% A) - S * (C %*% A))                    # A is symmetric
  return(Th_dot)
}

#' Order parameter:
#' rho(t) = <cos(theta_i(t) - theta_j(t))>
#' Computed as
#' rho(t) = [ sum_i <cos theta_i(t)><cos theta_j(t)> + <sin theta_i(t)><sin theta_j(t)> ] / M
rho_from_phases <- function(Th) {
  M <- nrow(Th)
  (crossprod(cos(Th)) + crossprod(sin(Th))) / M
}

#' Integrate NONLINEAR Kuramoto model with RK4
#'
#' Adaptive substepping: between consecutive output times
#' take ceiling(dt / dtmax) equal steps, so early (closely spaced, logarithmic)
#' output times are cheap.
#'
#' Stability of RK4 for the fastest linear mode requires K*lambda_max*dtmax < 2.7;
#' with lambda_max ~ 20 and dtmax = 0.01 this algorithm is two orders of magnitude inside it.

simulate_rho <- function(A, K, Theta0, times, dtmax = 0.01, verbose = TRUE) {
  Th <- Theta0; cur <- 0
  out <- vector("list", length(times))
  for (i in seq_along(times)) {
    nsub <- max(1L, as.integer(ceiling((times[i] - cur) / dtmax))) # number of substeps from previous time to current time
    h <- (times[i] - cur) / nsub                                   # size of substep
    for (s in seq_len(nsub)) {
      # RK4 method
      k1 <- kur_deriv(Th, A, K)                                  # evaluate derivative at current time
      k2 <- kur_deriv(Th + (h / 2) * k1, A, K)                   # evaluate derivative at t + h/2
      k3 <- kur_deriv(Th + (h / 2) * k2, A, K)                   # evaluate derivative at t + h/2
      k4 <- kur_deriv(Th + h * k3, A, K)                         # evaluate derivative at t + h
      Th <- Th + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)           # weighted average of derivatives to update phase
    }
    cur <- times[i]                                              # update time
    out[[i]] <- rho_from_phases(Th)                              # compute order parameter
    if (verbose && (i %% 5 == 0)) {
      cat(sprintf("    t = %8.4f  (%d/%d)\n", cur, i, length(times)))
    }
  }
  out
}

# ---------------------------------------------------------------------------
# 6. PARTITION COMPARISON
# ---------------------------------------------------------------------------

#' Normalised mutual information, 2I(X;Y) / (H(X) + H(Y)):
#' How much information one learns about partition x when knowing y (ground-truth vs. synchronized clusters).
#' 1 = identical partitions, 0 = independent (share no information).

nmi <- function(x, y) {                   # x,y contain the partition labels of the nodes
  tab <- table(x, y)                      # row=true communities, col=detected clusters, counts of nodes belonging to each pair of partitions
  n   <- sum(tab)                         # total number of nodes
  pxy <- tab / n                          # joint probability matrix
  px  <- rowSums(pxy); py <- colSums(pxy) # marginal probabilities
  pos <- pxy > 0                          # positions where pxy > 0
  I  <- sum(pxy[pos] * log(pxy[pos] / outer(as.numeric(px), as.numeric(py))[pos]))
  Hx <- -sum(px[px > 0] * log(px[px > 0]))
  Hy <- -sum(py[py > 0] * log(py[py > 0]))
  if (Hx + Hy == 0) return(1)
  2 * I / (Hx + Hy)
}

# ---------------------------------------------------------------------------
# 7. PLOTTING HELPERS
# ---------------------------------------------------------------------------

FIGDIR <- Sys.getenv("POCN_FIGDIR", unset = file.path("..", "figures"))
if (FIGDIR == "") FIGDIR <- file.path("..", "figures")
dir.create(FIGDIR, showWarnings = FALSE, recursive = TRUE)

open_png <- function(name, w = 2100, h = 750, res = 300, pointsize = 11) {
  png(file.path(FIGDIR, name), width = w, height = h, res = res, pointsize = pointsize, family = "serif")
}

#' Diverging / sequential color palettes.
pal_seq <- function(n) hcl.colors(n, "YlGnBu", rev = TRUE)                    # 0 to 1 plots
pal_div <- function(n) hcl.colors(n, "RdYlBu", rev = TRUE)                    # -1 to 1 plots

#' image() with the matrix readable: row 1 at the TOP
mat_image <- function(M, col = pal_seq(64), zlim = range(M, finite = TRUE), ...) {
  # image() otherwise puts row 1 at the bottom, transposing the matrix image
  image(seq_len(ncol(M)), seq_len(nrow(M)), t(M[nrow(M):1, , drop = FALSE]),
        col = col, zlim = zlim, xlab = "", ylab = "", axes = FALSE, ...)
  box()
}

#' Horizontal colour bar: create 1xN matrix filled with smooth sequence of colors
#' for the given z-range and palette, add axis
color_bar <- function(zlim, col, label = "") {
  op <- par(mar = c(2.0, 1.2, 0.4, 1.2), family = "serif")
  z <- seq(zlim[1], zlim[2], length.out = length(col))
  image(z, 1, matrix(z, ncol = 1), col = col, axes = FALSE, xlab = "", ylab = "")
  axis(1, cex.axis = 1.05, mgp = c(2, 0.5, 0))
  box()
  if (!is.null(label) && (is.language(label) || (is.character(label) && nchar(label) > 0))) {
    mtext(label, side = 1, line = 1.4, cex = 1.0)
  }
  par(op)
}

# Debug message
cat("[00_common.R] loaded\n")
