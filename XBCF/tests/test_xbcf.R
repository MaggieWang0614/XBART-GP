# simple demonstration of XBCF with default parameters
library(XBCF)

#### 1. DATA GENERATION PROCESS
n = 5000 # number of observations
# set seed here
# set.seed(1)

# generate dcovariates
x1 = rnorm(n)
x2 = rnorm(n)
x3 = rbinom(n,1,0.2)
x4 = rbinom(n,1,0.7)
x5 = data.frame(x3 = factor(sample(1:3,n,replace=TRUE,prob = c(0.1,0.6,0.3))))
x5 <- model.matrix(~ . + 0, data=x3, contrasts.arg = lapply(x3, contrasts, contrasts=FALSE))
x = cbind(x1,x2,x3,x4,x5)

# define treatment effects
tau = 2 + 0.5*x[,4]*(2*x[,5]-1)

## define prognostic function (RIC)
mu = function(x){
  lev = c(-0.5,0.75,0)
  result = 1 + x[,1]*(2*x[,2] - 2*(1-x[,2])) + lev[x3]
  return(result)
}

# compute propensity scores and treatment assignment
pi = pnorm(-0.5 + mu(x) - x[,2] + 0.*x[,4],0,3)
#hist(pi,100)
z = rbinom(n,1,pi)

# generate outcome variable
Ey = mu(x) + tau*z
sig = 0.25*sd(Ey)
y = Ey + sig*rnorm(n)

# If you didn't know pi, you would estimate it here
pihat = pi

# matrix prep
x <- data.frame(x)

# add pihat to the prognostic term matrix
x1 <- cbind(pihat,x)


#### 2. XBCF

# run XBCF
t1 = proc.time()
xbcf.fit = XBCF(y, z, x1, x, pcat_mod = 5,  pcat_con = 5)
t1 = proc.time() - t1

# get treatment individual-level estimates
tauhats <- getTaus(xbcf.fit)

# main model parameters can be retrieved below
#print(xbcf.fit$model_params)

# compare results to inference
plot(tau, tauhats); abline(0,1)
print(paste0("xbcf RMSE: ", sqrt(mean((tauhats - tau)^2))))
print(paste0("xbcf runtime: ", round(as.list(t1)$elapsed,2)," seconds"))
