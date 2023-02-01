aceBB<-function(x,nboot=250){
  
  diffpotmat<-matrix(rep(x,nboot),nrow=nboot,byrow=T)
  
  dirichlet_sample <- matrix( rexp(length(x) * nboot, 1) , nrow=nboot, byrow = TRUE)
  dirichlet_sample <- dirichlet_sample / rowSums(dirichlet_sample)
  
  posteriorace<-rowSums(dirichlet_sample*diffpotmat)
  
  ## sample from the posterior ##
  MCMCace<-sample(posteriorace,size=1)
  
  return(MCMCace)
}

bartalone<-function(xtr,ytr,xte){
  t = proc.time()
  ce_bart <- list()
  bartps<-bart(x.train=xtr,y.train=ytr,x.test=xte)
  ppd_test<- apply(bartps$yhat.test,1,function(x) rnorm(n=length(x),mean=x,sd=bartps$sigma))
  ppd_test_mean<-rowMeans(ppd_test)
  
  cate <- matrix(NA, nrow = nrow(ppd_test), ncol = ncol(ppd_test))
  cate[which(xtr[,1]==1),] <-ytr[which(xtr[,1]==1)] - ppd_test[which(xtr[,1]==1),]
  cate[which(xtr[,1]==0),] <- ppd_test[which(xtr[,1]==0),] - ytr[which(xtr[,1]==0)]
  
  ## individual causal effects ##
  ce_bart$ite<-rep(NA,length(ytr))
  ce_bart$ite[which(xtr[,1]==1)]<-ytr[which(xtr[,1]==1)]-ppd_test_mean[which(xtr[,1]==1)]
  ce_bart$ite[which(xtr[,1]==0)]<-ppd_test_mean[which(xtr[,1]==0)]-ytr[which(xtr[,1]==0)]
  
  ce_bart$itu<-apply(cate,1,quantile,probs=0.975)
  ce_bart$itl<-apply(cate,1,quantile,probs=0.025)
  
  ## average causal effects ##
  ppd_ice<-matrix(NA,nrow=nrow(ppd_test),ncol=length(ytr))
  for (j in 1:length(ytr)){
    if (xtr[j,1]==1) ppd_ice[j,]<-ytr[j]-ppd_test[j,]
    else ppd_ice[j,]<-ppd_test[j,]-ytr[j]
  }
  ## get ACE posterior using the Bayesian bootstrap ##
  ce_bart$ate<-apply(ppd_ice,1,aceBB)
  
  t = proc.time() - t
  ce_bart$time = t
  return(ce_bart)
}

bartspl<-function(datall,RO,nburn=10000,nsamp=5000){
  
  ## this function implements BART+SPL method (for continuous outcomes)
  ## first column in datall should be the observed outcome variable
  ## second column should be the binary exposure indicator
  ## third variable should be the PS or confounder on which to base the non-overlap
  ## any other variables to be included in the model are in the fourth column and beyond
  t = proc.time()
  ce_bartspl <- list()
  names(datall)[1:3]<-c('Yobs','x','ps')
  if (ncol(datall)>3) names(datall)[4:ncol(datall)]<-paste('u',1:(ncol(datall)-3),sep='')
  
  ## create an overlap dataset ##
  datov<-datall[which(RO==1),]
  ## create a test overlap dataset used to predict counterfactuals with BART ##
  datovtest<-datov
  datovtest$x<-1-datovtest$x
  ## create an overlap dataset for plugging into the spline model ##
  sp_ind<-which(datov$ps>quantile(datov$ps,.05) & datov$ps<quantile(datov$ps,.95))
  datov_sp<-datov[sp_ind,]
  ## create a non-overlap dataset ##
  datno<-datall[which(RO==0),]
  ## dataset containing units in the RN with E=0 (untreated) ##
  datno0<-datall[which(RO==0 & datall$x==0),]
  ## dataset containing units in the RN with E=1 (treated) ##
  datno1<-datall[which(RO==0 & datall$x==1),]
  
  ## for everyone in the RN, find the distance from their PS to the nearest PS in the RO ##
  ROdist<-NULL
  for (i in 1:nrow(datno)){
    psi<-datno$ps[i]
    ROdist<-c(ROdist,min(abs(datov$ps-psi)))
  }
  
  ## initialize the dbarts sampler to implement BART in the range of overlap ##
  dbfit<-dbarts(formula=Yobs~.,data=datov,test=datovtest[,2:ncol(datovtest)])

  
  ## initialize parameters etc ##
  delta<-matrix(0,nrow=nrow(datov),ncol=1)
  Y1s<-datov_sp$Yobs
  Y0s<-datov_sp$Yobs
  
  ## hyperparameters for the spline ##
  ncol_spl1 <- 1 + length(unique(quantile(datov_sp$ps,probs=c(.1,.25,.5,.75,.9)))) - 1 + length(unique(quantile(Y1s,probs=c(.2,.4,.6,.8)))) - 1
  ncol_spl0 <- 1 + length(unique(quantile(datov_sp$ps,probs=c(.1,.25,.5,.75,.9)))) - 1 + length(unique(quantile(Y0s,probs=c(.2,.4,.6,.8)))) - 1
  p_spl1<-ncol_spl1+(ncol(datall)-3)
  p_spl0<-ncol_spl0+(ncol(datall)-3)
  mu0<-matrix(0,nrow=p_spl1,ncol=1)
  Sigma1<-10000*diag(p_spl1)
  Sigma1_inv<-solve(Sigma1)
  Sigma0<-10000*diag(p_spl0)
  Sigma0_inv<-solve(Sigma0)
  a0<-1
  b0<-1
  beta1<-matrix(0,nrow=p_spl1,ncol=1)
  sigma_spl1<-1
  beta0<-matrix(0,nrow=p_spl0,ncol=1)
  sigma_spl0<-1
  
  ## matrices to store posterior samples ##
  # delta_save<-matrix(NA,nrow=nsamp,ncol=nrow(datov))
  delta_star_save<-matrix(NA,nrow=nsamp,ncol=nrow(datall))
  # sigma_bart_save<-rep(NA,nsamp)
  # beta_save<-matrix(NA,nrow=nsamp,ncol=p_spl)
  # sigma_spl_save<-rep(NA,nsamp)
  
  ## run the sampler ##
  for (i in 1:(nburn+nsamp)){
    
    ########################################################################
    ## 1. run the dbarts sampler once and take a sample from the BART ppd ##
    ########################################################################
    
    temp<-dbfit$run(numBurnIn=0,numSamples=1)
    ppd_test<-rnorm(n=length(temp$test),mean=temp$test,sd=temp$sigma)
    
    ## form the predicted individual causal effects in the RO from the BART predictions ##
    delta[which(datov$x==1),]<-datov$Yobs[which(datov$x==1)]-ppd_test[which(datov$x==1)]
    delta[which(datov$x==0),]<-ppd_test[which(datov$x==0)]-datov$Yobs[which(datov$x==0)]
    delta_sp<-delta[sp_ind,]
    
    ## update Y1s ##
    foosp<-temp$test[sp_ind]
    Y1s[which(datov_sp$x==0)]<-foosp[which(datov_sp$x==0)]
    Y0s[which(datov_sp$x==1)]<-foosp[which(datov_sp$x==1)]
    
    ## hyperparameters for the spline ##
    ncol_spl1 <- 1 + length(unique(quantile(datov_sp$ps,probs=c(.1,.25,.5,.75,.9)))) - 1 + length(unique(quantile(Y1s,probs=c(.2,.4,.6,.8)))) - 1
    ncol_spl0 <- 1 + length(unique(quantile(datov_sp$ps,probs=c(.1,.25,.5,.75,.9)))) - 1 + length(unique(quantile(Y0s,probs=c(.2,.4,.6,.8)))) - 1
    p_spl1<-ncol_spl1+(ncol(datall)-3)
    p_spl0<-ncol_spl0+(ncol(datall)-3)
    mu0<-matrix(0,nrow=p_spl0,ncol=1)
    mu1<-matrix(0,nrow=p_spl1,ncol=1)
    Sigma1<-10000*diag(p_spl1)
    Sigma1_inv<-solve(Sigma1)
    Sigma0<-10000*diag(p_spl0)
    Sigma0_inv<-solve(Sigma0)
    beta1<-matrix(0,nrow=p_spl1,ncol=1)
    sigma_spl1<-1
    beta0<-matrix(0,nrow=p_spl0,ncol=1)
    sigma_spl0<-1
    
    ## fit spline to the estimated causal effects in the RO from BART ##
    delta_star<-rep(NA,sum(RO==0))
    bs_ps<-rcspline.eval(datov_sp$ps,knots=quantile(datov_sp$ps,probs=c(.1,.25,.5,.75,.9)),inclx = T)
    
    ## spline for treated units (E=1) in the RN ##
    if (nrow(datno1)>0){
      
      ####################################
      ## 2. run the spline sampler once ##
      ####################################
      
      bs_y1s<-rcspline.eval(Y1s,knots=quantile(Y1s,probs=c(.2,.4,.6,.8)),inclx = T)
      if (ncol(datov_sp)<=3){
        X_spl<-as.matrix(cbind(1,bs_ps,bs_y1s))
      }
      else{
        X_spl<-as.matrix(cbind(1,bs_ps,bs_y1s,datov_sp[,4:ncol(datov_sp)]))
      }
      
      ## sample the betas ##
      Vbeta<-solve(Sigma1_inv+((1/sigma_spl1)*t(X_spl)%*%X_spl))
      Ebeta<-Vbeta%*%(Sigma1_inv%*%mu1+((1/sigma_spl1)*t(X_spl)%*%delta_sp))
      beta1<-matrix(mvrnorm(n=1,mu=Ebeta,Sigma=Vbeta),nrow=p_spl1,ncol=1)
      
      ## sample the sigma_spls ##
      a<-a0+(nrow(datov_sp)/2)
      b<-b0+((1/2)*sum((delta_sp-(X_spl%*%beta1))^2))
      sigma_spl1<-rinvgamma(n=1,shape=a,scale=b)
      
      ######################################################################
      ## 3. draw from the posterior predictive for the non-overlap region ##
      ######################################################################
      bs_ps_star<-rcspline.eval(datno1$ps,knots=quantile(datov_sp$ps,probs=c(.1,.25,.5,.75,.9)),inclx = T)
      bs_y1s_star<-rcspline.eval(datno1$Yobs,knots=quantile(Y1s,probs=c(.2,.4,.6,.8)),inclx=T)
      if (ncol(datno)<=3){
        X_spl_star<-as.matrix(cbind(1,bs_ps_star,bs_y1s_star))
      }
      else{
        X_spl_star<-as.matrix(cbind(1,bs_ps_star,bs_y1s_star,datno1[,4:ncol(datno1)]))
      }
      Eppd<-X_spl_star%*%beta1
      delta_star[which(datno$x==1)]<-rnorm(n=nrow(datno1),mean=Eppd,sd=sqrt(sigma_spl1+ROdist[which(datno$x==1)]*10*(max(delta)-min(delta))))
    }
    
    ## spline for untreated units (E=0) in the RN ##
    if (nrow(datno0)>0){
      
      ####################################
      ## 2. run the spline sampler once ##
      ####################################
      
      bs_y0s<-rcspline.eval(Y0s,knots=quantile(Y0s,probs=c(.2,.4,.6,.8)),inclx=T)
      if (ncol(datov_sp)<=3){
        X_spl<-as.matrix(cbind(1,bs_ps,bs_y0s))
      }
      else{
        X_spl<-as.matrix(cbind(1,bs_ps,bs_y0s,datov_sp[,4:ncol(datov_sp)]))
      }
      
      ## sample the betas ##
      Vbeta<-solve(Sigma0_inv+((1/sigma_spl0)*t(X_spl)%*%X_spl))
      Ebeta<-Vbeta%*%(Sigma0_inv%*%mu0+((1/sigma_spl0)*t(X_spl)%*%delta_sp))
      beta0<-matrix(mvrnorm(n=1,mu=Ebeta,Sigma=Vbeta),nrow=p_spl0,ncol=1)
      
      ## sample the sigma_spls ##
      a<-a0+(nrow(datov_sp)/2)
      b<-b0+((1/2)*sum((delta_sp-(X_spl%*%beta0))^2))
      sigma_spl0<-rinvgamma(n=1,shape=a,scale=b)
      
      ######################################################################
      ## 3. draw from the posterior predictive for the non-overlap region ##
      ######################################################################
      bs_ps_star<-rcspline.eval(datno0$ps,knots=quantile(datov_sp$ps,probs=c(.1,.25,.5,.75,.9)),inclx = T)
      bs_y0s_star<-rcspline.eval(datno0$Yobs,knots=quantile(Y0s,probs=c(.2,.4,.6,.8)),inclx=T)
      if (ncol(datno)<=3){
        X_spl_star<-as.matrix(cbind(1,bs_ps_star,bs_y0s_star))
      }
      else{
        X_spl_star<-as.matrix(cbind(1,bs_ps_star,bs_y0s_star,datno0[,4:ncol(datno0)]))
      }
      Eppd<-X_spl_star%*%beta0
      delta_star[which(datno$x==0)]<-rnorm(n=nrow(datno0),mean=Eppd,sd=sqrt(sigma_spl0+ROdist[which(datno$x==0)]*10*(max(delta)-min(delta))))
    }
    
    
    ###############################################
    ## 4. if we're past burn-in, save the output ##
    ###############################################
    if (i>nburn){
      delta_star_save[(i-nburn),which(RO==1)]<-c(delta)
      delta_star_save[(i-nburn),which(RO==0)]<-delta_star
    }
    
  }
  ce_bartspl$ite <- apply(delta_star_save,2,mean)
  ce_bartspl$itu <- apply(delta_star_save,2,quantile,probs=.975)
  ce_bartspl$itl <- apply(delta_star_save,2,quantile,probs=.025)
  ## use the Bayesian bootstrap to get ACE posterior ##
  ce_bartspl$ate<-apply(delta_star_save,1,aceBB)
  t = proc.time() - t
  ce_bartspl$time <- t
  return(ce_bartspl)
}


# a function that take in save_true and save_method
# return all metrics
getStatsAll <- function(ce, true){
  rep <- length(true)
  # ate.abs.bias <- rep(NA, rep)
  # ate.bias.pct <- rep(NA, rep)
  # ate.sd <- rep(NA, rep)
  # ate.se <- rep(NA, rep)
  ate.rmse <- rep(NA, rep)
  ate.coverage <- rep(NA, rep)
  ate.i.l <- rep(NA, rep)
  
  # ite.abs.bias <- rep(NA, rep)
  # ite.bias.pct <- rep(NA, rep)
  cate.rmse <- rep(NA, rep)
  cate.coverage <- rep(NA, rep)
  cate.i.l <- rep(NA, rep)
  time <- rep(NA, rep)
  
  for(iter in 1:rep){
    tau <- true[[iter]]
    ite <- ce[[iter]]$ite
    itu <- ce[[iter]]$itu
    itl <- ce[[iter]]$itl
    ate <- ce[[iter]]$ate
    time[iter] <- as.numeric(ce[[iter]]$time[3])
    
    
    
    # ate.abs.bias[iter] <- abs(mean(ate) - mean(tau))
    # ate.bias.pct[iter] <- abs((mean(ate) - mean(tau)) / mean(tau)) * 100
    # 
    # ate.sd[iter] <- sd(ate)
    # ate.se[iter] <- (mean(ate) - mean(tau))^2
    ate.rmse[iter] <-  (mean(ate) - mean(tau))^2
    ate.coverage[iter] <- (mean(tau) <= quantile(ate, 0.975) ) & (mean(tau) >= quantile(ate, 0.025))
    ate.i.l[iter] <- quantile(ate, 0.975) - quantile(ate, 0.025)
    
    # ite.abs.bias[iter] <- mean(abs(ite - tau))
    # ite.bias.pct[iter] <- mean(abs((ite - tau) / tau)) * 100
    cate.rmse[iter] <-  sqrt(mean((ite - tau)^2))
    cate.coverage[iter] <- mean((tau <= itu) & (tau >= itl))
    cate.i.l[iter] <- mean(itu - itl)
  }
  ret = list()
  # ret$ate.abs.bias <- mean(ate.abs.bias)
  # ret$ate.bias.pct <- mean(ate.bias.pct)
  # ret$ate.coverage <- mean(ate.coverage)
  # ret$sd <- mean(ate.sd)
  # ret$se <- sqrt(sum(ate.se)/(rep-1))
  ret$ate.rmse <- sqrt(mean(ate.rmse))
  ret$ate.coverage <- mean(ate.coverage)
  ret$ate.i.l <- mean(ate.i.l)
  
  # ret$cate.abs.bias <- mean(ite.abs.bias)
  # ret$cate.bias.pct <- mean(cate.bias.pct)
  ret$cate.rmse <- mean(cate.rmse)
  ret$cate.coverage <- mean(cate.coverage)
  ret$cate.i.l <- mean(cate.i.l)
  ret$time <- mean(time)
  return(ret)
}

# Return ATE / ITE / Coverage / I.L on RO / RN separately
getStatsSep <- function(ce, true, ce_ps){
  rep <- length(true)
  # ate.abs.bias <- rep(NA, rep)
  # ate.bias.pct <- rep(NA, rep)
  ate.rmse <- rep(NA, rep)
  ate.coverage <- rep(NA, rep)
  ate.i.l <- rep(NA, rep)
  
  # ite.abs.bias.overlap <- rep(NA, rep)
  # ite.abs.bias.trt <- rep(NA, rep)
  # ite.abs.bias.ctrl <- rep(NA, rep)
  
  # ite.bias.pct.overlap <- rep(NA, rep)
  # ite.bias.pct.trt <- rep(NA, rep)
  # ite.bias.pct.ctrl <- rep(NA, rep)
  
  ite.rmse.overlap <- rep(NA, rep)
  ite.rmse.trt <- rep(NA, rep)
  ite.rmse.ctrl <- rep(NA, rep)
  
  ite.coverage.overlap <- rep(NA, rep)
  ite.coverage.trt <- rep(NA, rep)
  ite.coverage.ctrl <- rep(NA, rep)
  
  ite.i.l.overlap <- rep(NA, rep)
  ite.i.l.trt <- rep(NA, rep)
  ite.i.l.ctrl <- rep(NA, rep)
  
  time <- rep(NA, rep)
  
  for(iter in 1:rep){
    tau <- true[[iter]]
    ite <- ce[[iter]]$ite
    itu <- ce[[iter]]$itu
    itl <- ce[[iter]]$itl
    ate <- ce[[iter]]$ate
    ps <- ce_ps[[iter]]
    time[iter] <- as.numeric(ce[[iter]]$time[3])
    
    # ate.abs.bias[iter] <- abs(mean(ate) - mean(tau))
    # ate.bias.pct[iter] <- abs((mean(ate) - mean(tau)) / mean(tau)) * 100
    ate.rmse[iter] <- (mean(ate) - mean(tau))^2
    ate.coverage[iter] <- (mean(tau) <= quantile(ate, 0.975) ) & (mean(tau) >= quantile(ate, 0.025))
    ate.i.l[iter] <- quantile(ate, 0.975) - quantile(ate, 0.025)
    
    overlap <- (ps < 1) & (ps > 0)
    trt <- ps == 1
    ctrl <- ps == 0
    
    # ite.abs.bias.overlap[iter] <- mean(abs(ite[overlap] - tau[overlap]))
    # ite.abs.bias.trt[iter] <- mean(abs(ite[trt] - tau[trt]))
    # ite.abs.bias.ctrl[iter] <- mean(abs(ite[ctrl] - tau[ctrl]))
    # 
    # ite.bias.pct.overlap[iter] <- mean(abs((ite[overlap] - tau[overlap]) / tau[overlap])) * 100
    # ite.bias.pct.trt[iter] <- mean(abs((ite[trt] - tau[trt]) / tau[trt])) * 100
    # ite.bias.pct.ctrl[iter] <- mean(abs((ite[ctrl] - tau[ctrl]) / tau[ctrl])) * 100
    
    ite.rmse.overlap[iter] <- sqrt(mean((ite[overlap] - tau[overlap])^2))
    ite.rmse.trt[iter] <- sqrt(mean((ite[trt] - tau[trt])^2))
    ite.rmse.ctrl[iter] <- sqrt(mean((ite[ctrl] - tau[ctrl])^2))
    
    ite.coverage.overlap[iter] <- mean((tau[overlap] <= itu[overlap]) & (tau[overlap] >= itl[overlap]))
    ite.coverage.trt[iter] <- mean((tau[trt] <= itu[trt]) & (tau[trt] >= itl[trt]))
    ite.coverage.ctrl[iter] <- mean((tau[ctrl] <= itu[ctrl]) & (tau[ctrl] >= itl[ctrl]))
    
    ite.i.l.overlap[iter] <- mean(itu[overlap] - itl[overlap])
    ite.i.l.trt[iter] <- mean(itu[trt] - itl[trt])
    ite.i.l.ctrl[iter] <- mean(itu[ctrl] - itl[ctrl])
  }
  ret = list()
  # ret$ate.abs.bias <- mean(ate.abs.bias)
  # ret$ate.bias.pct <- mean(ate.bias.pct)
  ret$ate.rmse <- sqrt(mean(ate.rmse))
  ret$ate.coverage <- mean(ate.coverage)
  ret$ate.i.l <- mean(ate.i.l)
  
  # ret$ite.abs.bias.overlap <- mean(ite.abs.bias.overlap)
  # ret$ite.abs.bias.trt <- mean(ite.abs.bias.trt)
  # ret$ite.abs.bias.ctrl <- mean(ite.abs.bias.ctrl)
  # 
  # ret$ite.bias.pct.overlap <- mean(ite.bias.pct.overlap)
  # ret$ite.bias.pct.trt <- mean(ite.bias.pct.trt)
  # ret$ite.bias.pct.ctrl <- mean(ite.bias.pct.ctrl)
  
  ret$ite.rmse.overlap <- mean(ite.rmse.overlap)
  ret$ite.coverage.overlap <- mean(ite.coverage.overlap)
  ret$ite.i.l.overlap <- mean(ite.i.l.overlap)
  
  ret$ite.rmse.trt <- mean(ite.rmse.trt)
  ret$ite.coverage.trt <- mean(ite.coverage.trt)
  ret$ite.i.l.trt <- mean(ite.i.l.trt)
  
  ret$ite.rmse.ctrl <- mean(ite.rmse.ctrl)
  ret$ite.coverage.ctrl <- mean(ite.coverage.ctrl)
  ret$ite.i.l.ctrl <- mean(ite.i.l.ctrl)
  
  ret$time <- mean(time)
  return(ret)
}


## gutman and rubin's method ##
gr<-function(Y,trt,ps,X,M,qps){
  ## step 1: create subclasses based on the PS ##
  ps.<-as.numeric(cut(ps,qps,include.lowest = T,right=F))
  
  use<-1
  for (i in 1:length(unique(ps.))){
    if (length(which(ps.==i & trt==0))<3 | length(which(ps.==i & trt==1))<3) use<-0
  }
  
  if (use==0){
    return(list(use))
  }
  else{
    ## steps 2 & 3: estimate splines separately for treated and controls & sample M times from the posterior ##
    pstrans<-log(ps/(1-ps))
    psspline<-ns(pstrans,knots = qps[-c(1,length(qps))])
    xort<-NULL
    for (i in 1:ncol(X)){
      xort<-rbind(xort,lm(pstrans~X[,i])$resid)
    }
    xort<-t(xort)
    
    Ytrt<-Y[which(trt==1)]
    pssplinetrt<-psspline[which(trt==1)]
    xorttrt<-xort[which(trt==1)]
    Yctl<-Y[which(trt==0)]
    pssplinectl<-psspline[which(trt==0)]
    xortctl<-xort[which(trt==0)]
    
    keep<-sample(1:10000,M)
    
    outtrt<-as.matrix(MCMCregress(Ytrt~pssplinetrt+xorttrt,burnin = 1000,mcmc=10000,verbose=F))[keep,]
    outctl<-as.matrix(MCMCregress(Yctl~pssplinectl+xortctl,burnin = 1000,mcmc=10000,verbose=F))[keep,]
    
    ## step 4: impute the missing potential outcomes for each sample from the posterior##
    imptrt<-list()
    impctl<-list()
    for (i in 1:M){
      imptrt<-c(imptrt,list(rowSums(matrix(outctl[i,(-ncol(outctl))],nrow=length(Ytrt),ncol=ncol(outctl)-1,byrow=T)*cbind(1,pssplinetrt,xorttrt))))
      impctl<-c(impctl,list(rowSums(matrix(outtrt[i,(-ncol(outtrt))],nrow=length(Yctl),ncol=ncol(outtrt)-1,byrow=T)*cbind(1,pssplinectl,xortctl))))
    }
    imptrt<-as.matrix(as.data.frame(imptrt))
    impctl<-as.matrix(as.data.frame(impctl))
    
    ## step 6: estimate the treatment effect and its sample variance for each dataset ##
    indtrteff<-t(rbind(matrix(Ytrt,nrow=length(Ytrt),ncol=M,byrow=F)-imptrt,
                       impctl-matrix(Yctl,nrow=length(Yctl),ncol=M,byrow=F)))
    
    iceavg<-apply(indtrteff,2,mean)
    icelw<-apply(indtrteff,2,quantile,probs=0.025)
    icehi<-apply(indtrteff,2,quantile,probs=0.975)
    ace<-rowMeans(indtrteff)
    
    #gammahat<-mean(avgtrteff)
    #gammaci<-quantile(avgtrteff,probs=c(.025,.975))
    
    return(list(use,iceavg,icelw,icehi,ace))
  }
}



pw_overlap<-function(ps,E,a,b){
  ps1<-ps[which(E==1)]
  ps0<-ps[which(E==0)]
  ps1<-ps1[order(ps1)]
  ps0<-ps0[order(ps0)]
  
  RO<-rep(0,length(ps))
  for (k in ps){
    cte<-0
    for (e in 0:1){
      cth<-0
      temp<-get(paste('ps',e,sep=''))
      fooless<-temp[which(temp<k)]
      foomore<-temp[which(temp>k)]
      if (k %in% temp){
        if (length(fooless)>=b & length(foomore)>=b){
          allcheck<-c(fooless[(length(fooless)-(b-1)):length(fooless)],k,foomore[1:b])
        } else if (length(fooless)>=b & length(foomore)<b){
          allcheck<-c(fooless[(length(fooless)-(b-1)):length(fooless)],k,foomore,rep(Inf,b-length(foomore)))
        } else if (length(fooless)<b & length(foomore)>=b){
          allcheck<-c(rep(-Inf,b-length(fooless)),fooless,k,foomore[1:b])
        } else{
          allcheck<-c(rep(-Inf,b-length(fooless)),fooless,k,foomore,rep(Inf,b-length(foomore)))
        }
        
        for (h in 1:(b+1)){
          if (abs(allcheck[h]-allcheck[h+b])<a) cth<-cth+1
        }
      } else{
        if (length(fooless)>=(b+1) & length(foomore)>=(b+1)){
          allcheck<-c(fooless[(length(fooless)-b):length(fooless)],k,foomore[1:(b+1)])
        } else if (length(fooless)>=(b+1) & length(foomore)<(b+1)){
          allcheck<-c(fooless[(length(fooless)-b):length(fooless)],k,foomore,rep(Inf,(b+1)-length(foomore)))
        } else if (length(fooless)<(b+1) & length(foomore)>=(b+1)){
          allcheck<-c(rep(-Inf,(b+1)-length(fooless)),fooless,k,foomore[1:(b+1)])
        } else{
          allcheck<-c(rep(-Inf,(b+1)-length(fooless)),fooless,k,foomore,rep(Inf,(b+1)-length(foomore)))
        }
        
        for (h in 1:(b+2)){
          if (abs(allcheck[h]-allcheck[h+b+1])<a) cth<-cth+1
        }
      }
      if (cth>0) cte<-cte+1
    }
    if (cte==2) RO[which(ps==k)]<-1
  }
  
  return(RO)
}



sbart <- function(x, y, z){
  t <- proc.time()
  ce_bart_strat = list()
  x = as.matrix(x)
  xtr = as.matrix(x[z==1, ])
  ytr = y[z==1]
  xctrl = as.matrix(x[z==0, ])
  yctrl = y[z==0]
  bart.tr <- bart(x.train = xtr,y.train = ytr, x.test = x)
  bart.ctrl <- bart(x.train = xctrl, y.train = yctrl, x)
  #expected tau
  tau <- bart.tr$yhat.test - bart.ctrl$yhat.test
  # sampled tau
  # tauhat <- t(apply(tau, 2, function(x) rnorm(length(x), x, sqrt(bart.tr$sigma^2 + bart.ctrl$sigma^2))))
  tauhat <- t(tau)
  ce_bart_strat$ite = rowMeans(tauhat)
  ce_bart_strat$itu = apply(tauhat, 1, quantile, 0.975, na.rm = TRUE)
  ce_bart_strat$itl = apply(tauhat, 1, quantile, 0.025, na.rm = TRUE)
  #sate per draw
  ## get ACE posterior using the Bayesian bootstrap ##
  ce_bart_strat$ate<-apply(tauhat,2,aceBB)
  t = proc.time() - t
  ce_bart_strat$time = t
  return(ce_bart_strat)
}