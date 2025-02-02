library(XBCF)

n = 500
nt = 200
x = as.matrix(rnorm(n+nt, 0, 5), n+nt,1)
# tau = 5 + cos(0.5*x +1)
tau = -0.1*x #sin(0.2*x)
bound = 5
A = rbinom(n+nt, 1, 0*(x>bound) + 0.5*(abs(x)<=bound) + 1*(x< -bound))
# A = rbinom(n+nt, 1, 0*(x< -5) + 0.5*(abs(x)<=5) + 1*(x>5))
y1 = cos(0.2*x) + tau
y0 = cos(0.2*x)
Ey = A*y1 + (1-A)*y0
sig = 0.25*sd(Ey)
y = Ey + sig*rnorm(n+nt)
# plot(x, y, col = A + 1, cex = 0.8)

# propensity score?
# pihat = NULL
# make sure pihat_tr is consistent
sink("/dev/null")
fitz = nnet::nnet(A[1:n] ~.,data = as.matrix(x[1:n], n, 1), size = 3,rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
sink() # close the stream
pihat_tr = fitz$fitted.values
pihat_te = predict(fitz, as.matrix(x[(n+1):(n+nt)], nt, 1))

ytrain = as.matrix(y[1:n]); ytest = as.matrix(y[(n+1):(n+nt)])
ztrain = as.matrix(A[1:n]); ztest = as.matrix(A[(n+1):(n+nt)])
# pihat_tr = pihat[1:n]; pihat_te = pihat[(n+1):(n+nt)]
xtrain = as.matrix(x[1:n,]); xtest = as.matrix(x[(n+1):(n+nt),])
tautr = tau[1:n]; taute = tau[(n+1):(n+nt)]

# test on train
ytest = ytrain; xtest = xtrain; ztest = ztrain; taute = tautr; pihat_te = pihat_tr
# run XBCF
t1 = proc.time()
# burnin = 5; num_sweeps = 30; num_trees_trt = 10; num_trees_pr = 10
burnin = 20; num_sweeps = 60; num_trees_trt = 10; num_trees_pr = 10
xbcf.fit = XBCF(as.matrix(ytrain), as.matrix(ztrain), xtrain, xtrain, 
                pihat = pihat_tr, pcat_con = 0,  pcat_mod = 0 ,
                num_sweeps = num_sweeps,  burnin = burnin,
                n_trees_mod = num_trees_trt, n_trees_con = num_trees_pr)
# tau_gp = diff(range(xbcf.fit$tauhats.adjusted))/ xbcf.fit$model_params$num_trees_trt
tau_gp = mean(xbcf.fit$sigma1_draws)^2/ xbcf.fit$model_params$num_trees_trt
pred.gp = predictGP(xbcf.fit, as.matrix(ytrain), as.matrix(ztrain), xtrain, xtrain, xtest, xtest, 
                    pihat_tr = pihat_tr, pihat_te = pihat_tr, theta = 0.1, tau = tau_gp, verbose = FALSE)
# pred = predict.XBCF(xbcf.fit, xt, xt, pihat = pihat)
t1 = proc.time() - t1

pred = predict.XBCF(xbcf.fit, xtest, xtest, pihat = NULL)

# sigmahat <- sqrt(xbcf.fit$sigma0_draws[,60]^2 + xbcf.fit$sigma1_draws[,60]^2)
# tauhats.pred = t(apply(xbcf.fit$tauhats.adjusted, 1, function(x) rnorm(length(x), x, sigmahat)))
# tauhats.gp <- t(apply(pred.gp$tau.adjusted, 1, function(x) rnorm(length(x), x, sigmahat)))
tauhats.pred <- xbcf.fit$tauhats.adjusted
tauhats.gp <- pred.gp$tau.adjusted

# true tau?
cat('True ATE:, ', round(mean(taute), 3), ', GP tau: ', round(mean(tauhats.gp), 3), 
    ', XBCF tau: ', round(mean(tauhats.pred), 3), '\n')

gp.upper <- apply(tauhats.gp, 1, quantile, 0.975, na.rm = TRUE)
gp.lower <- apply(tauhats.gp, 1, quantile, 0.025, na.rm = TRUE)
xbcf.upper <- apply(tauhats.pred, 1, quantile, 0.975, na.rm = TRUE)
xbcf.lower <- apply(tauhats.pred, 1, quantile, 0.025, na.rm = TRUE)

# evaluate coverage
cat('Coverage:', '\n')
cat('GP = ', round(mean((gp.upper >= taute) & (gp.lower <= taute)), 3), '\n')
cat('XBCF = ', round(mean((xbcf.upper >= taute) & (xbcf.lower <= taute)), 3), '\n')

par(mfrow=c(1,1))
plot(xtest, y1[1:n], col = 2, cex = 0.5, ylim = range(y1, y0))
points(xtest, y0[1:n], col = 1, cex = 0.5)
points(xtest, rowMeans(pred$mudraws), cex = 0.5, col = 4)
points(xtest, rowMeans(pred$taudraws + tauhats.gp),cex = 0.5, col = 5)
points(xtest, rowMeans(pred.gp$mu.adjusted), cex = 0.5, col = 6)
points(xtest, rowMeans(pred.gp$mu.adjusted +tauhats.gp), cex = 0.5, col = 7)
legend('topright', cex = 0.5, pch = 1, col = c(2, 1, 4, 5, 6, 7), 
       legend = c('y1', 'y0', 'mu.xbcf', 'mu.xbcf+tau', 'mu.gp', 'mu.gp+tau.gp'))

par(mfrow=c(1, 1))
plot(xtest, y[1:n], col = ztest + 1, cex = 0.5, ylab = 'Observed outcomes')
points(xtest[order(xtest)], y1[1:n][order(xtest)], col = 2, cex = 0.2)
points(xtest[order(xtest)], y0[1:n][order(xtest)], col = 1, cex = 0.2)
points(xtest, rowMeans(pred.gp$mu.adjusted), col = 6, cex = 0.5)
points(xtest, rowMeans(pred.gp$mu.adjusted + tauhats.gp), col = 7, cex = 0.5)
abline(v = -5, col = 4)
abline(v = 5, col = 4)
legend('topright', cex = 0.5, pch = 1, col = c(1, 2, 6, 7, 3), 
       legend = c('Treated', 'Control', 'Pred Treated', 'Pred Control', '95% C.I'))




par(mfrow=c(1,1))
plot(xtest, y1[1:n] - y0[1:n], col = ztest + 1, cex = 0.5, ylim = range(rowMeans(tauhats.gp), gp.upper, gp.lower, y1- y0))
points(xtest, rowMeans(tauhats.pred), col = 4, cex = 0.5)
# points(xtest, rowMeans(pred.gp$tau.adjusted), col = 6, cex = 0.5)
points(xtest, rowMeans(tauhats.gp), col = 7, cex = 0.5)
points(xtest, gp.upper, col = 3, cex = 0.5)
points(xtest, gp.lower, col = 3, cex = 0.5)
abline(v = -5, col = 4)
abline(v = 5, col = 4)
legend('topleft', cex = 0.5, pch = 1, col = c(1, 4, 7, 3), 
       legend = c('y1 - y0', 'tau.xbcf','tau.gp', 'gp C.I'))

