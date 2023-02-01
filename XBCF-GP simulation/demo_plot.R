##########################################
## XBCF-GP comparison demo
##########################################
# setwd('~/Dropbox (ASU)/xbart_gp/causal/')
# setwd('~/coverage/demo/')
n <- 500
simnum<-100
a <- 0.1 #a parameter in non-overlap definition#
b <- 7 #b parameter in non-overlap definition#

# exp set up --------------------------------------------------------------
## run simulation ##
# setwd(wd)
# library(Hmisc)
library(MCMCpack)
library(MASS)
library(stats)
library(splines)
library(BayesTree)
library(dbarts)
library(XBART)
library(XBCF)
source('functions.R')
set.seed(simnum)


# DGP --------------------------------------------------------------------

# a 1-dim dgp that has non-overlap area
n = 1000
x = seq(-10, 10, length.out=n)
mu = sin(x)
tau = 0.25*x
pi = 0.08*x + 0.5
pi[pi > 1] = 1
pi[pi < 0] = 0
# plot(x,pi)
# pi = rep(0.5, n)
# pi[(x<4.5)&(x>-4.5)] = 0
# pi[(x>8)|(x< -8)]=0


z = rbinom(n, 1, pi)
f = mu + tau*z
y = f + 0.2*sd(f)*rnorm(n)

# overlap area
v1 = max(x[which(pi==0)])
v2 = min(x[which(pi==1)])


# fitz = XBART.multinomial(y = z, num_class = 2, X = x, p_cateogrical = 0)
# predz = predict(fitz, as.matrix(x))
# ps = predz$prob

sink("/dev/null")
fitz = nnet::nnet(z ~ .,data = cbind(z, x), size = 3,rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
sink() # close the stream
ps = fitz$fitted.values

RO_cor<-pw_overlap(ps=ps,E=z,a=a,b=b)
order_ps_cor<-ps[order(ps)]
order_RO_cor<-RO_cor[order(ps)]
ll_cor<-max(which(order_RO_cor==1))

## ignore small non-overlap in the left tail (add all left tail to the RO) ##
temp1<-order_ps_cor[1:ll_cor]
ps_addRO<-temp1[which(order_RO_cor[1:ll_cor]==0)]
RO_cor[which((ps %in% ps_addRO)==1)] <- 1


# plot demo --------------------------------------------------------------
library(ggplot2)
library(dplyr)
plt_df <- data.frame(x = x, z = z, y = y)
plt_df$Treatment <- factor(z, levels=c(0, 1), labels=c("Controled", "Treated"))
plt_df %>% ggplot(aes(x = x, fill = Treatment, color = Treatment)) +
  geom_histogram(aes(y = ..density..),alpha=0.6, position = 'identity') +
  theme(
    panel.background = element_rect(fill='transparent'), #transparent panel bg
    plot.background = element_rect(fill='transparent', color=NA), #transparent plot bg
    panel.grid.major = element_blank(), #remove major gridlines
    panel.grid.minor = element_blank() #remove minor gridlines
  )

plt_df %>% 
  ggplot(aes(x, y, color = Treatment)) +
  geom_point() +
  ylab(label = "Response") + 
  theme(
    panel.background = element_rect(fill='transparent'), #transparent panel bg
    plot.background = element_rect(fill='transparent', color=NA), #transparent plot bg
    panel.grid.major = element_blank(), #remove major gridlines
    panel.grid.minor = element_blank() #remove minor gridlines
  )

p1 <- plt_df %>% 
  ggplot(aes(x, y, color = Treatment)) +
  geom_point() +
  ylab(label = "Response") + 
  theme(
    legend.position = "none",
    panel.background = element_rect(fill='transparent'), #transparent panel bg
    plot.background = element_rect(fill='transparent', color=NA), #transparent plot bg
    panel.grid.major = element_blank(), #remove major gridlines
    panel.grid.minor = element_blank() #remove minor gridlines
  )

plt_df$te <- tau
p2 <- plt_df %>% 
  ggplot(aes(x, te, color = Treatment)) +
  geom_point() +
  ylab(label = "Treatement Effect") + 
  theme(
    panel.background = element_rect(fill='transparent'), #transparent panel bg
    plot.background = element_rect(fill='transparent', color=NA), #transparent plot bg
    panel.grid.major = element_blank(), #remove major gridlines
    panel.grid.minor = element_blank() #remove minor gridlines
  )

require(gridExtra)
grid.arrange(p1, p2, ncol=2)
#######################
## 1. XBCF ##
#######################
xbcf.fit = XBCF(y, z, as.matrix(x), as.matrix(x), n_trees_mod = 20, num_sweeps=100,
                pihat = ps, pcat_con = 0,  pcat_mod = 0, Nmin = 20)

ce_xbcf = list()
xbcf.tau = xbcf.fit$tauhats.adjusted
ce_xbcf$ite = rowMeans(xbcf.tau)
ce_xbcf$itu = apply(xbcf.tau, 1, quantile, 0.975, na.rm = TRUE)
ce_xbcf$itl = apply(xbcf.tau, 1, quantile, 0.025, na.rm = TRUE)

#######################
## 1. XBCF-GP ##
#######################
tau_gp = mean(xbcf.fit$sigma1_draws)^2/ (xbcf.fit$model_params$num_trees_trt)
xbcf.gp = predictGP(xbcf.fit, y, z, as.matrix(x),as.matrix(x), as.matrix(x), as.matrix(x),
                    pihat_tr = ps, pihat_te = ps, theta = 0.1, tau = tau_gp, verbose = FALSE)
ce_xbcf_gp = list()
xbcf.gp.tau <- xbcf.gp$tau.adjusted
ce_xbcf_gp$ite = rowMeans(xbcf.gp.tau)
ce_xbcf_gp$itu = apply(xbcf.gp.tau, 1, quantile, 0.975, na.rm = TRUE)
ce_xbcf_gp$itl = apply(xbcf.gp.tau, 1, quantile, 0.025, na.rm = TRUE)

#######################
## 2. BART stratified ##
#######################
ce_sbart <- sbart(x = as.matrix(x), y = y, z = z)

#######################
## 3. untrimmed BART ##
#######################
## create trimmed and untrimmed datasets ##
datall_cor<-data.frame(y,z,ps,x)

testdat_cor<-datall_cor
testdat_cor$z<- 1 - datall_cor$z
ce_ubart <- bartalone(xtr=datall_cor[,2:ncol(datall_cor)],ytr=datall_cor[,1],xte=testdat_cor[,2:ncol(datall_cor)])

# #################
# ## 4. BART+SPL ##
# # #################

## find RO and RN ## 
RO_cor<-pw_overlap(ps=ps,E=z,a=a,b=b)
order_ps_cor<-ps[order(ps)]
order_RO_cor<-RO_cor[order(ps)]
ll_cor<-max(which(order_RO_cor==1))

## ignore small non-overlap in the left tail (add all left tail to the RO) ##
temp1<-order_ps_cor[1:ll_cor]
ps_addRO<-temp1[which(order_RO_cor[1:ll_cor]==0)]
RO_cor[which((ps %in% ps_addRO)==1)] <- 1

library(Hmisc)
library(MCMCpack)
print("start bartspl")
ce_bartspl<-bartspl(datall=datall_cor,RO=RO_cor)
print("finish bartspl")

# # Show plot ---------------------------------------------------------------
# layout(matrix(c(1,2,3,4), 2, 2,byrow=TRUE))
# cex_size = 0.5
# font_size = 1
# # xbcf
# plot(x, tau, col = z + 1, ylim = range(tau, ce_xbcf$ite),
#        ylab = 'Treatment Effect', xlab = 'XBCF', cex.lab = font_size)
# abline(v = v1, col = 1, lty = 3)
# abline(v = v2, col = 1, lty = 3)
# lines(x,ce_xbcf$ite, col = 4, lwd=2)
# lines(x,ce_xbcf$itu, col = 3, lwd=2)
# lines(x,ce_xbcf$itl, col = 3, lwd=2)
# legend('topleft', legend = c('Treated', 'Control', 'XBCF', '95% CI'), col = c(2, 1, 4, 3),
#        lty = 1, cex = cex_size, font_size)
# 
# # # xbcf_gp
# plot(x, tau, col = z + 1, ylim = range(tau, ce_xbcf_gp), ylab = 'Treatment Effect', xlab = 'XBCF-GP' , cex.lab = font_size)
# abline(v = v1, col = 1, lty = 3)
# abline(v = v2, col = 1, lty = 3)
# lines(x,ce_xbcf_gp$ite , col = 4, lwd=2)
# lines(x,ce_xbcf_gp$itu, col = 3, lwd=2)
# lines(x,ce_xbcf_gp$itl, col = 3, lwd=2)
# legend('topleft', legend = c('Treated', 'Control', 'XBCF-GP', '95% CI'), col = c(2, 1, 4, 3),lty = 1, cex = cex_size)
# 
# # sbart
# plot(x, tau, col = z + 1, ylim = range(tau, ce_sbart$ite),
#      ylab = 'Treatment Effect', xlab = 'SBART', cex.lab = font_size)
# abline(v = v1, col = 1, lty = 3)
# abline(v = v2, col = 1, lty = 3)
# lines(x,ce_sbart$ite, col = 4, lwd=2)
# lines(x,ce_sbart$itu, col = 3, lwd=2)
# lines(x,ce_sbart$itl, col = 3, lwd=2)
# legend('topleft', legend = c('Treated', 'Control', 'SBART', '95% CI'), col = c(2, 1, 4, 3),lty = 1, cex = cex_size)
# 
# # ubart
# plot(x, tau, col = z + 1, ylim = range(tau, ce_ubart$ite),
#      ylab = 'Treatment Effect', xlab = 'UBART', cex.lab = font_size)
# abline(v = v1, col = 1, lty = 3)
# abline(v = v2, col = 1, lty = 3)
# lines(x,ce_ubart$ite, col = 4, lwd=2)
# lines(x,ce_ubart$itu, col = 3, lwd=2)
# lines(x,ce_ubart$itl, col = 3, lwd=2)
# legend('topleft', legend = c('Treated', 'Control', 'UBART', '95% CI'), col = c(2, 1, 4, 3),lty = 1, cex = cex_size)
# # 
# # bart+spl
# # plot(x, tau, col = z + 1, ylim = range(tau, ce_bartspl$ite),
# #      ylab = 'Treatment Effect', xlab = 'BART+SPL', cex.lab = font_size)
# # abline(v = v1, col = 1, lty = 3)
# # abline(v = v2, col = 1, lty = 3)
# # lines(x,ce_bartspl$ite, col = 4, lwd=2)
# # lines(x,ce_bartspl$itu, col = 3, lwd=2)
# # lines(x,ce_bartspl$itl, col = 3, lwd=2)
# # legend('topleft', legend = c('Treated', 'Control', 'BART+SPL', '95% CI'), col = c(2, 1, 4, 3),lty = 1, cex = cex_size)
# 
# 

# Demo plot ---------------------------------------------------------------
cex_size = 1.2
lab_size = 2
tick_size = 2
line_size = 1.5

# xbcf
pdf(file="xbcf_demo.pdf", width = 6, height = 6)
par(mar=c(5,6,4,1)+.1)
plot(x, tau, col = z + 1, ylim = range(tau, ce_xbcf$itu, ce_xbcf$itl),
     ylab = 'Treatment Effect', xlab = '', cex.lab = lab_size, cex.axis = tick_size, cex = line_size, pch = 20)
abline(v = v1, col = 1, lty = 3, cex = 2)
abline(v = v2, col = 1, lty = 3, cex = 2)
lines(x,ce_xbcf$ite, col = 4, lwd=2, cex = line_size)
lines(x,ce_xbcf$itu, col = 3, lwd=2, cex = line_size)
lines(x,ce_xbcf$itl, col = 3, lwd=2, cex = line_size)
legend('topleft', legend = c('Treated', 'Control', 'XBCF', '95% CI'), col = c(2, 1, 4, 3),
       lty = c(NA, NA, 1, 1), pch = c(20, 20, NA, NA), cex = cex_size, font_size)
dev.off()

# xbcf_gp
pdf(file="xbcf_gp_demo.pdf", width = 6, height = 6 )
par(mar=c(5,6,4,1)+.1)
plot(x, tau, col = z + 1, ylim = range(tau, ce_xbcf_gp$ite, ce_xbcf_gp$itl), 
     ylab = 'Treatment Effect', xlab = '' , cex.lab = lab_size, cex.axis = tick_size, cex = line_size, pch = 20)
abline(v = v1, col = 1, lty = 3, cex = 2)
abline(v = v2, col = 1, lty = 3, cex = 2)
lines(x,ce_xbcf_gp$ite , col = 4, lwd=2, cex = line_size)
lines(x,ce_xbcf_gp$itu, col = 3, lwd=2, cex = line_size)
lines(x,ce_xbcf_gp$itl, col = 3, lwd=2, cex = line_size)
legend('topleft', legend = c('Treated', 'Control', 'XBCF-GP', '95% CI'), col = c(2, 1, 4, 3),
       lty = c(NA, NA, 1, 1), pch = c(20, 20, NA, NA), cex = cex_size)
dev.off()

# sbart
pdf(file="sbart_demo.pdf", width = 6, height = 6)
par(mar=c(5,6,4,1)+.1)
plot(x, tau, col = z + 1, ylim = range(tau, ce_sbart$itu, ce_sbart$itl),
     ylab = 'Treatment Effect', xlab = '', cex.lab = lab_size, cex.axis = tick_size, cex = line_size, pch = 20)
abline(v = v1, col = 1, lty = 3, cex = 2)
abline(v = v2, col = 1, lty = 3, cex = 2)
lines(x,ce_sbart$ite, col = 4, lwd=2, cex = line_size)
lines(x,ce_sbart$itu, col = 3, lwd=2, cex = line_size)
lines(x,ce_sbart$itl, col = 3, lwd=2, cex = line_size)
legend('topleft', legend = c('Treated', 'Control', 'SBART', '95% CI'), col = c(2, 1, 4, 3),
       lty = c(NA, NA, 1, 1), pch = c(20, 20, NA, NA), cex = cex_size)
dev.off()

# ubart
pdf(file="ubart_demo.pdf", width = 6, height = 6)
par(mar=c(5,6,4,1)+.1)
plot(x, tau, col = z + 1, ylim = range(tau, ce_ubart$itu, ce_ubart$itl),
     ylab = 'Treatment Effect', xlab = '', cex.lab = lab_size, cex.axis = tick_size, cex = line_size, pch = 20)
abline(v = v1, col = 1, lty = 3, cex = 2)
abline(v = v2, col = 1, lty = 3, cex = 2)
points(x,ce_ubart$ite, col = 4, pch = 20, cex = 0.7)
points(x,ce_ubart$itu, col = 3, pch = 20, cex = 0.7)
points(x,ce_ubart$itl, col = 3, pch = 20, cex = 0.7)
legend('topleft', legend = c('Treated', 'Control', 'UBART', '95% CI'), col = c(2, 1, 4, 3),
       pch = c(20, 20, 20, 20), cex = cex_size)
dev.off()

# bart+spl
pdf(file="bartspl_demo.pdf", width = 6, height = 6)
par(mar=c(5,6,4,1)+.1)
plot(x, tau, col = z + 1, ylim = range(tau, ce_bartspl$itu, ce_bartspl$itl),
     ylab = 'Treatment Effect', xlab = '', cex.lab = lab_size, cex.axis = tick_size, cex = line_size, pch = 20)
abline(v = v1, col = 1, lty = 3, cex = 2)
abline(v = v2, col = 1, lty = 3, cex = 2)
points(x,ce_bartspl$ite, col = 4, pch = 20)
points(x,ce_bartspl$itu, col = 3, pch = 20)
points(x,ce_bartspl$itl, col = 3, pch = 20)
legend('topleft', legend = c('Treated', 'Control', 'BART+SPL', '95% CI'), col = c(2, 1, 4, 3),
      pch = c(20, 20, 20, 20), cex = cex_size)
dev.off()

# mu plots ----------------------------------------------------------------
# par(mfrow=c(1,2))
# plot(x,  mu, col =  1, ylim = range(mu+tau,y, xbcf.yhats),
#      ylab = 'Y', xlab = 'XBCF')
# points(x, mu+tau,col=2)
# abline(v = v1, col = 1, lty = 3)
# abline(v = v2, col = 2, lty = 3)
# lines(x, rowMeans(xbcf.fit$mu), col = 3, lwd=2)
# lines(x, rowMeans(xbcf.fit$mu + xbcf.fit$tau.b0), col = 4, lwd=2)
# lines(x, rowMeans(xbcf.fit$mu + xbcf.fit$tau.b1), col = 7, lwd=2)
# legend('topleft', legend = c('mu', 'mu+b0*tau', 'mu+b1*tau'), col = c(3, 4, 7), lty = 1, cex = 0.5)
# 
# plot(x, mu+ tau *rbinom(n,1,0.5), col = 1, ylim = range(mu+ tau,y, xbcf.yhats),
#      ylab = 'Y', xlab = 'XBCF-GP')
# points(x,mu+tau, col=2)
# abline(v = v1, col = 1, lty = 3)
# abline(v = v2, col = 2, lty = 3)
# lines(x, rowMeans(xbcf.gp$mu), col = 3, lwd=2)
# lines(x, rowMeans(xbcf.gp$mu + xbcf.gp$tau.b0), col = 4, lwd=2)
# lines(x, rowMeans(xbcf.gp$mu + xbcf.gp$tau.b1), col = 7, lwd=2)
# legend('topleft', legend = c('mu', 'mu+b0*tau', 'mu+b1*tau'), col = c(3, 4, 7), lty = 1, cex = 0.5)
# 
