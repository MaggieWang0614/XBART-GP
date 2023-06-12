##########################################
## XBCF-GP comparison demo
##########################################
n <- 500
simnum <- 100
a <- 0.1 # a parameter in non-overlap definition#
b <- 7 # b parameter in non-overlap definition#

# Experiment set up --------------------------------------------------------------
library(Hmisc)
library(MCMCpack)
library(MASS)
library(stats)
library(splines)
library(BayesTree)
library(dbarts)
library(XBART)
library(XBCF)
source("functions.R")
set.seed(simnum)


# DGP --------------------------------------------------------------------
# generate a 1-dim dgp with non-overlap area
n <- 1000
x <- seq(-10, 10, length.out = n)
mu <- sin(x)
tau <- 0.25 * x
pi <- 0.08 * x + 0.5
pi[pi > 1] <- 1
pi[pi < 0] <- 0


z <- rbinom(n, 1, pi)
f <- mu + tau * z
y <- f + 0.2 * sd(f) * rnorm(n)

# Overlap area
v1 <- max(x[which(pi == 0)])
v2 <- min(x[which(pi == 1)])

# Estimate propensity score
sink("/dev/null")
fitz <- nnet::nnet(z ~ ., data = cbind(z, x), size = 3, rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
sink() # close the stream
ps <- fitz$fitted.values

RO_cor <- pw_overlap(ps = ps, E = z, a = a, b = b)
order_ps_cor <- ps[order(ps)]
order_RO_cor <- RO_cor[order(ps)]
ll_cor <- max(which(order_RO_cor == 1))

# ignore small non-overlap in the left tail (add all left tail to the RO) ##
temp1 <- order_ps_cor[1:ll_cor]
ps_addRO <- temp1[which(order_RO_cor[1:ll_cor] == 0)]
RO_cor[which((ps %in% ps_addRO) == 1)] <- 1

#######################
## 1. XBCF ##
#######################
xbcf.fit <- XBCF(y, z, as.matrix(x), as.matrix(x),
  n_trees_mod = 20, num_sweeps = 100,
  pihat = ps, pcat_con = 0, pcat_mod = 0, Nmin = 20
)

ce_xbcf <- list()
xbcf.tau <- xbcf.fit$tauhats.adjusted
ce_xbcf$ite <- rowMeans(xbcf.tau)
ce_xbcf$itu <- apply(xbcf.tau, 1, quantile, 0.975, na.rm = TRUE)
ce_xbcf$itl <- apply(xbcf.tau, 1, quantile, 0.025, na.rm = TRUE)

#######################
## 1. XBCF-GP ##
#######################
tau_gp <- mean(xbcf.fit$sigma1_draws)^2 / (xbcf.fit$model_params$num_trees_trt)
xbcf.gp <- predictGP(xbcf.fit, y, z, as.matrix(x), as.matrix(x), as.matrix(x), as.matrix(x),
  pihat_tr = ps, pihat_te = ps, theta = sqrt(10), tau = tau_gp, verbose = FALSE
)
ce_xbcf_gp <- list()
xbcf.gp.tau <- xbcf.gp$tau.adjusted
ce_xbcf_gp$ite <- rowMeans(xbcf.gp.tau)
ce_xbcf_gp$itu <- apply(xbcf.gp.tau, 1, quantile, 0.975, na.rm = TRUE)
ce_xbcf_gp$itl <- apply(xbcf.gp.tau, 1, quantile, 0.025, na.rm = TRUE)

#######################
## 2. BART stratified ##
#######################
ce_sbart <- sbart(x = as.matrix(x), y = y, z = z)

#######################
## 3. untrimmed BART ##
#######################
## create trimmed and untrimmed datasets ##
datall_cor <- data.frame(y, z, ps, x)
testdat_cor <- datall_cor
testdat_cor$z <- 1 - datall_cor$z
ce_ubart <- bartalone(xtr = datall_cor[, 2:ncol(datall_cor)], ytr = datall_cor[, 1], xte = testdat_cor[, 2:ncol(datall_cor)])

# #################
# ## 4. BART+SPL ##
# # #################

## find RO and RN ##
RO_cor <- pw_overlap(ps = ps, E = z, a = a, b = b)
order_ps_cor <- ps[order(ps)]
order_RO_cor <- RO_cor[order(ps)]
ll_cor <- max(which(order_RO_cor == 1))

## ignore small non-overlap in the left tail (add all left tail to the RO) ##
temp1 <- order_ps_cor[1:ll_cor]
ps_addRO <- temp1[which(order_RO_cor[1:ll_cor] == 0)]
RO_cor[which((ps %in% ps_addRO) == 1)] <- 1

ce_bartspl <- bartspl(datall = datall_cor, RO = RO_cor)

# demo ggplot ---------------------------------------------------------------
library(ggplot2)
df <- data.frame(X = x, True = tau, Group = as.factor(z), XBCF)
ggplot(data = df) +
  geom_point(aes(x = X, y = True, color = Group)) +
  scale_color_manual(values = c(""))

df <- data.frame(X = x, True = tau, XBCF = ce_xbcf$ite, Upper = ce_xbcf$itu, Lower = ce_xbcf$itl)

# Demo plot ---------------------------------------------------------------
cex_size <- 1.2
lab_size <- 2
tick_size <- 2
point_size <- 1
line_size <- 2
color <- c("#bd0026", "#fd8d3c", rgb(254/255, 178/255, 76/255, 0.5))

# Ground Truth
pdf(file = "demo_true.pdf", width = 6, height = 6)
par(mar = c(5, 6, 4, 1) + .1)
plot(x, tau,
     col = color[4 - z*2 - 1], ylim = range(tau, ce_xbcf$itu, ce_xbcf$itl),
     ylab = "Treatment Effect", xlab = "", cex.lab = lab_size, cex.axis = tick_size, cex = 1, pch = 20
)
abline(v = v1, col = 1, lty = 3, cex = 2)
abline(v = v2, col = 1, lty = 3, cex = 2)
legend("topleft",
       legend = c("Treated", "Control"), col = color[c(3, 1)],
       pch = c(20, 20), cex = cex_size, font_size
)
dev.off()

# XBCF
pdf(file = "xbcf_demo.pdf", width = 6, height = 6)
par(mar = c(5, 6, 4, 1) + .1)
plot(x, tau,
  col = color[1], ylim = range(tau, ce_xbcf$itu, ce_xbcf$itl),
  ylab = "Treatment Effect", xlab = "", cex.lab = lab_size, cex.axis = tick_size, lty = 1, lwd = 0.5, pch = 0, cex = 0.3
)
polygon(c(rev(x), x), c(rev(ce_xbcf$itu), ce_xbcf$itl), border = NA, col = color[3])
abline(v = v1, col = 1, lty = 3, cex = 2)
abline(v = v2, col = 1, lty = 3, cex = 2)
lines(x, tau, col = color[1], lwd = 2, cex = line_size)
lines(x, ce_xbcf$ite, col = color[2], lwd = 2, cex = line_size)
lines(x, ce_xbcf$itu, col = color[3], lwd = 2, lty = 2, cex = line_size)
lines(x, ce_xbcf$itl, col = color[3], lwd = 2, lty = 2, cex = line_size)
legend("topleft",
  legend = c("True", "XBCF", "95% CI"), col = color,
  lty = c(1, 1, 1), pch = c(NA, NA, NA), cex = cex_size, font_size
)
dev.off()


# XBCF-GP
pdf(file = "xbcf_gp_demo.pdf", width = 6, height = 6)
par(mar = c(5, 6, 4, 1) + .1)
plot(x, tau,
  col = color[1], ylim = range(tau, ce_xbcf_gp$ite, ce_xbcf_gp$itl),
  ylab = "Treatment Effect", xlab = "", cex.lab = lab_size, cex.axis = tick_size, lty = 1, lwd = 0.5, pch = 0, cex = 0.3
)
polygon(c(rev(x), x), c(rev(ce_xbcf_gp$itu), ce_xbcf_gp$itl), border = NA, col = color[3])
abline(v = v1, col = 1, lty = 3, cex = 2)
abline(v = v2, col = 1, lty = 3, cex = 2)
lines(x, tau, col = color[1], lwd = 2, cex = line_size)
lines(x, ce_xbcf_gp$ite, col = color[2], lwd = 2, cex = line_size)
lines(x, ce_xbcf_gp$itu, col = color[3], lwd = 2, lty = 2, cex = line_size)
lines(x, ce_xbcf_gp$itl, col = color[3], lwd = 2, lty = 2, cex = line_size)
legend("topleft",
  legend = c("True", "XBCF-GP", "95% CI"), col = color,
  lty = c(1, 1, 1), pch = c(NA, NA, NA), cex = cex_size
)
dev.off()

# SBART
pdf(file = "sbart_demo.pdf", width = 6, height = 6)
par(mar = c(5, 6, 4, 1) + .1)
plot(x, tau,
  col = color[1], ylim = range(tau, ce_sbart$itu, ce_sbart$itl),
  ylab = "Treatment Effect", xlab = "", cex.lab = lab_size, cex.axis = tick_size, lty = 1, lwd = 0.5, pch = 0, cex = 0.3
)
polygon(c(rev(x), x), c(rev(ce_sbart$itu), ce_sbart$itl), border = NA, col = color[3])
abline(v = v1, col = 1, lty = 3, cex = 2)
abline(v = v2, col = 1, lty = 3, cex = 2)
lines(x, tau, col = color[1], lwd = 2, cex = line_size)
lines(x, ce_sbart$ite, col = color[2], lwd = 2, cex = line_size)
lines(x, ce_sbart$itu, col = color[3], lwd = 2, lty = 2, cex = line_size)
lines(x, ce_sbart$itl, col = color[3], lwd = 2, lty = 2, cex = line_size)
legend("topleft",
  legend = c("True", "SBART", "95% CI"), col = color,
  lty = c(1, 1, 1), pch = c(NA, NA, NA), cex = cex_size
)
dev.off()

# UBART
pdf(file = "ubart_demo.pdf", width = 6, height = 6)
par(mar = c(5, 6, 4, 1) + .1)
plot(x, tau,
  col = color[1], ylim = range(tau, ce_ubart$itu, ce_ubart$itl),
  ylab = "Treatment Effect", xlab = "", cex.lab = lab_size, cex.axis = tick_size, lty = 1, lwd = 0.5, pch = 0, cex = 0.3
)
polygon(c(rev(x), x), c(rev(ce_ubart$itu), ce_ubart$itl), border = NA, col = color[3])
abline(v = v1, col = 1, lty = 3, cex = 2)
abline(v = v2, col = 1, lty = 3, cex = 2)
lines(x, ce_ubart$ite, col = color[2], lwd = 2, cex = line_size)
lines(x, ce_ubart$itu, col = color[3], lwd = 2, lty = 2, cex = line_size)
lines(x, ce_ubart$itl, col = color[3], lwd = 2, lty = 2, cex = line_size)
lines(x, tau, col = color[1], lwd = 2, cex = line_size)
legend("topleft",
  legend = c("True", "UBART", "95% CI"), col = color,
  lty = c(1, 1, 1), pch = c(NA, NA, NA), cex = cex_size
)
dev.off()

# BART+SPL
pdf(file = "bartspl_demo.pdf", width = 6, height = 6)
par(mar = c(5, 6, 4, 1) + .1)
plot(x, tau,
     col = color[1], ylim = range(tau, ce_bartspl$itu, ce_bartspl$itl),
     ylab = "Treatment Effect", xlab = "", cex.lab = lab_size, cex.axis = tick_size, lty = 1, lwd = 0.5, pch = 0, cex = 0.3
)
polygon(c(rev(x), x), c(rev(ce_bartspl$itu), ce_bartspl$itl), border = NA, col = color[3])
abline(v = v1, col = 1, lty = 3, cex = 2)
abline(v = v2, col = 1, lty = 3, cex = 2)
lines(x, ce_bartspl$ite, col = color[2], lwd = 2, cex = line_size)
lines(x, ce_bartspl$itu, col = color[3], lwd = 2, lty = 2, cex = line_size)
lines(x, ce_bartspl$itl, col = color[3], lwd = 2, lty = 2, cex = line_size)
lines(x, tau, col = color[1], lwd = 2, cex = line_size)
legend("topleft",
       legend = c("True", "BART+SPL", "95% CI"), col = color,
       lty = c(1, 1, 1), pch = c(NA, NA, NA), cex = cex_size
)
dev.off()
