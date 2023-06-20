##########################################
## Experiment on various dgp comparing XBCF-GP to baseline methods
##########################################

n <- 500
simnum <- 100 # seed
a <- 0.1 # a parameter in non-overlap definition#
b <- 7 # b parameter in non-overlap definition#
reps <- 10

# Experiment Set Up --------------------------------------------------------------
library(Hmisc)
library(MCMCpack)
library(MASS)
library(stats)
library(splines)
library(dbarts)

library(XBART)
library(XBCF)
source("functions.R")
filename <- paste("results/bcf", ".RData", sep = "")
set.seed(simnum)

kappa <- 0.5
rho <- 1

g <- function(x) {
  ret <- rep(0, length(x))
  ret[x == 1] <- 2
  ret[x == 2] <- -1
  ret[x == 3] <- -4
  return(ret)
}

if (file.exists(filename)) {
  load(filename)
} else {
  save_param <- list()
  save_true <- list()
  save_trt <- list()
  save_ps <- list()
  save_xbcf <- list()
  save_xbcf_gp <- list()
  save_sbart <- list()
  save_ubart <- list()
  save_bartspl <- list()
}

params <- c("linear homogeneous", "linear heterogeneous", "non-linear homogeneous", "non-linear heterogeneous")

for (i in 1:length(params)) {
  for (j in 1:reps) {
    param <- params[i]

    if (sum(unlist(save_param) == param) >= j) {
      next
    }
    print(paste("param = ", param, ", iter = ", j))

    if (param == "linear homogeneous") {
      # linear
      mu <- function(x) {
        1 + g(x[, 4]) + x[, 1] * x[, 3]
      } # xbcf linear
      h <- function(x) {
        1.1 * pnorm(3 * mu(x) / sd(mu(x)) - 0.5 * x[, 1] - 3) - 0.15 + runif(nrow(x)) / 10
      }
      # homogenerous
      tau <- function(x) {
        rep(3, nrow(x))
      }
    } else if (param == "linear heterogeneous") {
      # linear
      mu <- function(x) {
        1 + g(x[, 4]) + x[, 1] * x[, 3]
      } # xbcf linear
      h <- function(x) {
        1.1 * pnorm(3 * mu(x) / sd(mu(x)) - 0.5 * x[, 1] - 3) - 0.15 + runif(nrow(x)) / 10
      }
      # heterogeneous
      tau <- function(x) {
        1 + 2 * x[, 2] * x[, 5]
      }
    } else if (param == "non-linear homogeneous") {
      # non-linear
      mu <- function(x) {
        -6 + g(x[, 4]) + 6 * abs(x[, 3] - 1)
      }
      h <- function(x) {
        1.1 * pnorm(3 * mu(x) / sd(mu(x)) - 0.5 * x[, 1]) - 0.15 + runif(nrow(x)) / 10
      }

      # homogenerous
      tau <- function(x) {
        rep(3, nrow(x))
      }
    } else if (param == "non-linear heterogeneous") {
      # non-linear
      mu <- function(x) {
        -6 + g(x[, 4]) + 6 * abs(x[, 3] - 1)
      }
      h <- function(x) {
        1.1 * pnorm(3 * mu(x) / sd(mu(x)) - 0.5 * x[, 1]) - 0.15 + runif(nrow(x)) / 10
      }
      # heterogeneous
      tau <- function(x) {
        1 + 2 * x[, 2] * x[, 5]
      }
    }


    pi <- function(x) {
      res <- h(x)
      res[res > 1] <- 1
      res[res < 0] <- 0
      return(res)
    }

    x <- matrix(0, n, 5)
    x[, 1] <- rnorm(n)
    x[, 2] <- rnorm(n)
    x[, 3] <- rnorm(n)
    x[, 4] <- rbinom(n, 1, 0.5)
    x[, 5] <- sample(1:3, n, replace = TRUE)

    ce_ps <- pi(x)
    z <- rbinom(n, 1, ce_ps)

    # draw outcome
    f_xz <- mu(x) + rho * tau(x) * z
    sigma <- kappa * sd(rho * f_xz)
    y1 <- mu(x) + rho * tau(x)
    y0 <- mu(x)
    y <- f_xz + sigma * rnorm(n)

    # calculate the true average treatment effect (ATE)
    print(mean(tau(x)))

    ce_true <- y1 - y0
    ce_trt <- z

    sink("/dev/null")
    fitz <- nnet::nnet(z ~ ., data = cbind(z, x), size = 3, rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
    sink() # close the stream
    ps <- fitz$fitted.values

    ## find RO and RN ##
    RO_cor <- pw_overlap(ps = ps, E = z, a = a, b = b)
    order_ps_cor <- ps[order(ps)]
    order_RO_cor <- RO_cor[order(ps)]
    ll_cor <- max(which(order_RO_cor == 1))

    ## ignore small non-overlap in the left tail (add all left tail to the RO) ##
    temp1 <- order_ps_cor[1:ll_cor]
    ps_addRO <- temp1[which(order_RO_cor[1:ll_cor] == 0)]
    RO_cor[which((ps %in% ps_addRO) == 1)] <- 1

    ## create trimmed and untrimmed datasets ##
    datall_cor <- data.frame(y, z, ps, x)
    dattr_cor <- datall_cor[which(RO_cor == 1), ]

    #######################
    ## 1. XBCF ##
    #######################
    t <- proc.time()
    xbcf.fit <- XBCF(datall_cor$y, datall_cor$z, as.matrix(x), as.matrix(x),
      pihat = datall_cor$ps, pcat_con = 2, pcat_mod = 2,
      n_trees_mod = 20, num_sweeps = 100, Nmin = 20
    )
    # individual causal effect
    ce_xbcf <- list()
    xbcf.tau <- xbcf.fit$tauhats.adjusted
    ce_xbcf$ite <- rowMeans(xbcf.tau)
    ce_xbcf$itu <- apply(xbcf.tau, 1, quantile, 0.975, na.rm = TRUE)
    ce_xbcf$itl <- apply(xbcf.tau, 1, quantile, 0.025, na.rm = TRUE)
    ce_xbcf$ate <- apply(xbcf.tau, 2, aceBB)
    t <- proc.time() - t
    ce_xbcf$time <- t
    print("finish xbcf")

    #######################
    ## 2. XBCF-GP##
    #######################
    t <- proc.time()
    tau_gp <- mean(xbcf.fit$sigma1_draws)^2 / xbcf.fit$model_params$num_trees_trt
    xbcf.gp <- predictGP(xbcf.fit, datall_cor$y, datall_cor$z, as.matrix(x), as.matrix(x), as.matrix(x), as.matrix(x),
      pihat_tr = datall_cor$ps, pihat_te = datall_cor$ps, theta = sqrt(10), tau = tau_gp, verbose = FALSE
    )
    ce_xbcf_gp <- list()

    xbcf.gp.tau <- xbcf.gp$tau.adjusted
    ce_xbcf_gp$ite <- rowMeans(xbcf.gp.tau)
    ce_xbcf_gp$itu <- apply(xbcf.gp.tau, 1, quantile, 0.975, na.rm = TRUE)
    ce_xbcf_gp$itl <- apply(xbcf.gp.tau, 1, quantile, 0.025, na.rm = TRUE)

    ce_xbcf_gp$ate <- apply(xbcf.gp.tau, 2, aceBB)
    t <- proc.time() - t
    ce_xbcf_gp$time <- ce_xbcf$time + t
    print("finish xbcf-gp")

    #######################
    ## 3. BART stratified ##
    #######################
    ce_sbart <- sbart(x = as.matrix(x), y = datall_cor$y, z = datall_cor$z)

    #######################
    ## 4. untrimmed BART ##
    #######################

    testdat_cor <- datall_cor
    testdat_cor$z <- 1 - datall_cor$z
    ce_ubart <- bartalone(xtr = datall_cor[, 2:ncol(datall_cor)], ytr = datall_cor[, 1], xte = testdat_cor[, 2:ncol(datall_cor)])

    #################
    ## 5. BART+SPL ##
    ##################
    ce_bartspl <- bartspl(datall = datall_cor, RO = RO_cor)
    print("finish bartspl")

    #################
    ## SAVE OUTPUT ##
    #################

    save_param <- c(save_param, param)
    save_true <- c(save_true, list(ce_true))
    save_trt <- c(save_trt, list(ce_trt))
    save_ps <- c(save_ps, list(ce_ps))
    save_xbcf <- c(save_xbcf, list(ce_xbcf))
    save_xbcf_gp <- c(save_xbcf_gp, list(ce_xbcf_gp))
    save_sbart <- c(save_sbart, list(ce_sbart))
    save_ubart <- c(save_ubart, list(ce_ubart))
    save_bartspl <- c(save_bartspl, list(ce_bartspl))

    save(save_param, save_true, save_trt, save_ps,
      save_xbcf, save_xbcf_gp, save_sbart, save_ubart, save_bartspl,
      file = filename
    )
  }
}



# Organize output ----------------------------------------------------------------
load("results/bcf.RData")
source("functions.R")

save_param <- unlist(save_param)
params <- unique(save_param)

df <- c()

for (param in params) {
  print(paste("param = ", param))
  idx <- save_param == param
  xbcf <- getStatsAll(save_xbcf[idx], save_true[idx])
  xbcf_gp <- getStatsAll(save_xbcf_gp[idx], save_true[idx])
  sbart <- getStatsAll(save_sbart[idx], save_true[idx])
  ubart <- getStatsAll(save_ubart[idx], save_true[idx])
  bartspl <- getStatsAll(save_bartspl[idx], save_true[idx])

  temp <- data.frame(
    xbcf = unlist(xbcf), xbcf_gp = unlist(xbcf_gp),
    sbart = unlist(sbart), ubart = unlist(ubart), bartspl = unlist(bartspl)
  )

  rownames(temp) <- names(xbcf)
  temp <- data.frame(t(temp))
  temp$method <- rownames(temp)
  temp$dgp <- rep(param, nrow(temp))
  temp <- temp[, c(9, 8, 1:7)]
  rownames(temp) <- NULL
  df <- rbind(df, temp)
}

df[, 3:9] <- round(df[, 3:9], 3)
write.csv(df, "results/bcf.csv", row.names = FALSE)
