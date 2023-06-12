import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xbart import XBART
from sklearn.ensemble import RandomForestRegressor

import time
from tqdm import tqdm
from os.path import exists
import csv

seed = 98765
np.random.seed(seed)

######################################
# Time efficiency experiments on interior and exterior points
# with test scale 1.5
######################################  
note = 'Time efficiency experiments on update_gp branch with test scale 1.5, theta = sqrt(10)'

######################################
# Define regression algorithm (for all other methods)
######################################

def leastsq_ridge(X,Y,X1,ridge_mult=0.001):
    lam = ridge_mult * np.linalg.svd(X,compute_uv=False).max()**2
    betahat = np.linalg.solve(\
            np.c_[np.ones(X.shape[0]),X].T.dot(np.c_[np.ones(X.shape[0]),X]) \
                              + lam * np.diag(np.r_[0,np.ones(X.shape[1])]),
            np.c_[np.ones(X.shape[0]),X].T.dot(Y))
    return betahat[0] + X1.dot(betahat[1:])

def random_forest(X,Y,X1,ntree=20):
    rf = RandomForestRegressor(n_estimators=ntree,criterion='mae').fit(X,Y)
    return rf.predict(X1)

def neural_net(X,Y,X1):
    nnet = MLPRegressor(solver='lbfgs',activation='logistic').fit(X,Y)
    return nnet.predict(X1)

def check_out_of_range(x1, x_min, x_max):
    for i in range(len(x1)):
        if x1[i] < x_min[i] or x1[i] > x_max[i]:
            return True
    return False


######################################
# Define dgp
######################################

def jacknife_linear(x):
    beta = np.random.normal(size=x.shape[1])
    beta = beta/np.sqrt((beta**2).sum()) * np.sqrt(SNR)
    return x.dot(beta)

def linear(x):
    d = x.shape[1]
    beta = [-2 + 4*(i - 1) / (d-1) for i in range(1, d+1)]
    return x.dot(beta)
    
def single_index(x):
    d = x.shape[1]
    gamma = [-1.5 + i/3 for i in list(range(0, d))]
    a =  np.apply_along_axis(lambda x: sum((x-gamma)**2), 1, x)
    f = 10 * np.sqrt(a) + np.sin(5*a)
    return f

def trig_poly(x):
    f = np.apply_along_axis(lambda x: 5 * np.sin(3*x[0]) + 2 * x[1]**2 + 3 * x[2] * x[3], 1, x)
    return f


def xbart_max(x):
    return np.apply_along_axis(lambda x: max(x[0:2]), 1, x)

def generate_data(x, dgp):
    if dgp == 'linear':
        return linear(x)
    if dgp == 'single_index':
        return single_index(x)
    if dgp == 'trig_poly':
        return trig_poly(x)
    if dgp == 'max':
        return xbart_max(x)




def compute_PIs(X,Y,X1,alpha,fit_muh_fun):
    n = len(Y)
    n1 = X1.shape[0]
    times = {}
    
    ###############################
    # XBART-GP
    ###############################
    start = time.time()
    num_trees = 20
    num_sweeps = 100
    burnin = 20
    tau = np.var(Y) / num_trees
    theta = sqrt(10)
    n_min = 20
    xbart = XBART(num_trees = num_trees, num_sweeps = num_sweeps + burnin, burnin = burnin, tau = tau, sampling_tau = True, n_min = n_min)
    xbart.fit(X,Y,0)
    mu_pred = xbart.predict(X1, return_mean = False)
    mu_pred = mu_pred[:, burnin:] # discard burnin
    xbart_train = time.time() - start
    
    y_pred = pd.DataFrame(mu_pred).transpose().apply(
        lambda x: x + xbart.sigma_draws[burnin:,num_trees - 1] * np.random.normal(size=num_sweeps), 0).transpose() 
    xbart_PI =  pd.DataFrame(y_pred).transpose().apply(
                    lambda x: [np.quantile(x, alpha/2), np.quantile(x, 1 - alpha/2), x.mean()], 0).transpose()
    xbart_PI.rename(columns = {0: 'lower', 1: 'upper', 2:'pred'}, inplace = True)
    times['XBART'] = time.time() - start
    
    mu_pred_gp = xbart.predict_gp(X, Y, X1, p_cat = 0, theta = theta, tau = tau, return_mean=False)
    mu_pred_gp = mu_pred_gp[:,burnin:]
    y_pred_gp = pd.DataFrame(mu_pred_gp).transpose().apply(
        lambda x: x + xbart.sigma_draws[burnin:,num_trees - 1] * np.random.normal(size=num_sweeps), 0).transpose() 
    xbart_PI_gp =  pd.DataFrame(y_pred_gp).transpose().apply(
                    lambda x: [np.quantile(x, alpha/2), np.quantile(x, 1 - alpha/2), x.mean()], 0).transpose()
    xbart_PI_gp.rename(columns = {0: 'lower', 1: 'upper', 2:'pred'}, inplace = True)
    times['XBART-GP'] = time.time() - start - (times['XBART'] - xbart_train)

    ###############################
    # Jackknife+ XBART
    ###############################

    start = time.time()
    xbart = XBART(num_trees = num_trees, num_sweeps = num_sweeps + burnin, burnin = burnin, tau = tau, sampling_tau = True)
    resids_LOO_xbart = np.zeros(n)
    muh_LOO_vals_testpoint_xbart = np.zeros((n,n1))
    for i in range(n):
        xbart.fit(np.delete(X,i,0),np.delete(Y,i), 0)
        muh_vals_LOO = xbart.predict(np.r_[X[i].reshape((1,-1)),X1])
        resids_LOO_xbart[i] = np.abs(Y[i] - muh_vals_LOO[0])
        muh_LOO_vals_testpoint_xbart[i] = muh_vals_LOO[1:]
    ind_q = (np.ceil((1-alpha)*(n+1))).astype(int)
    times['jackknife+ XBART'] = time.time() - start
    
    ###############################
    # CV+ XBART
    ###############################
    
    start = time.time()
    K = 10
    n_K = np.floor(n/K).astype(int)
    base_inds_to_delete = np.arange(n_K).astype(int)
    resids_LKO_xbart = np.zeros(n)
    muh_LKO_vals_testpoint_xbart = np.zeros((n,n1))
    for i in range(K):
        inds_to_delete = (base_inds_to_delete + n_K*i).astype(int)
        # muh_vals_LKO = fit_muh_fun(np.delete(X,inds_to_delete,0),np.delete(Y,inds_to_delete),\
        #                            np.r_[X[inds_to_delete],X1])
        xbart.fit(np.delete(X,inds_to_delete,0),np.delete(Y,inds_to_delete),0)
        muh_vals_LKO = xbart.predict(np.r_[X[inds_to_delete],X1])
        resids_LKO_xbart[inds_to_delete] = np.abs(Y[inds_to_delete] - muh_vals_LKO[:n_K])
        for inner_K in range(n_K):
            muh_LKO_vals_testpoint_xbart[inds_to_delete[inner_K]] = muh_vals_LKO[n_K:]
    ind_Kq = (np.ceil((1-alpha)*(n+1))).astype(int)
    times['CV+ XBART'] = time.time() - start

     ###############################
    # Jackknife+ RF
    ###############################
    
    start = time.time()
    resids_LOO = np.zeros(n)
    muh_LOO_vals_testpoint = np.zeros((n,n1))
    for i in range(n):
        muh_vals_LOO = fit_muh_fun(np.delete(X,i,0),np.delete(Y,i),\
                                   np.r_[X[i].reshape((1,-1)),X1])
        resids_LOO[i] = np.abs(Y[i] - muh_vals_LOO[0])
        muh_LOO_vals_testpoint[i] = muh_vals_LOO[1:]
    ind_q = (np.ceil((1-alpha)*(n+1))).astype(int)
    times['jackknife+ RF'] = time.time() - start


    ###############################
    # CV+ RF
    ###############################

    start = time.time()
    K = 10
    n_K = np.floor(n/K).astype(int)
    base_inds_to_delete = np.arange(n_K).astype(int)
    resids_LKO = np.zeros(n)
    muh_LKO_vals_testpoint = np.zeros((n,n1))
    for i in range(K):
        inds_to_delete = (base_inds_to_delete + n_K*i).astype(int)
        muh_vals_LKO = fit_muh_fun(np.delete(X,inds_to_delete,0),np.delete(Y,inds_to_delete),\
                                   np.r_[X[inds_to_delete],X1])
        resids_LKO[inds_to_delete] = np.abs(Y[inds_to_delete] - muh_vals_LKO[:n_K])
        for inner_K in range(n_K):
            muh_LKO_vals_testpoint[inds_to_delete[inner_K]] = muh_vals_LKO[n_K:]
    ind_Kq = (np.ceil((1-alpha)*(n+1))).astype(int)
    times['CV+ RF'] = time.time() - start
    
    ###############################
    # construct prediction intervals
    ###############################

    PIs_dict = {'XBART' : xbart_PI,\
                'XBART-GP': xbart_PI_gp,\
                'jackknife+ XBART' : pd.DataFrame(\
                    np.c_[np.sort(muh_LOO_vals_testpoint_xbart.T - resids_LOO_xbart,axis=1).T[-ind_q], \
                        np.sort(muh_LOO_vals_testpoint_xbart.T + resids_LOO_xbart,axis=1).T[ind_q-1],
                         muh_LOO_vals_testpoint_xbart.T.mean(axis = 1)],\
                           columns = ['lower','upper', 'pred']),\
                 'jackknife+ RF' : pd.DataFrame(\
                    np.c_[np.sort(muh_LOO_vals_testpoint.T - resids_LOO,axis=1).T[-ind_q], \
                        np.sort(muh_LOO_vals_testpoint.T + resids_LOO,axis=1).T[ind_q-1], 
                         muh_LOO_vals_testpoint.T.mean(axis = 1)],\
                           columns = ['lower','upper', 'pred']),\
                'CV+ XBART' : pd.DataFrame(\
                    np.c_[np.sort(muh_LKO_vals_testpoint_xbart.T - resids_LKO_xbart,axis=1).T[-ind_Kq], \
                        np.sort(muh_LKO_vals_testpoint_xbart.T + resids_LKO_xbart,axis=1).T[ind_Kq-1],
                         muh_LKO_vals_testpoint_xbart.T.mean(axis = 1)],\
                           columns = ['lower','upper', 'pred']),\
                'CV+ RF' : pd.DataFrame(\
                    np.c_[np.sort(muh_LKO_vals_testpoint.T - resids_LKO,axis=1).T[-ind_Kq], \
                        np.sort(muh_LKO_vals_testpoint.T + resids_LKO,axis=1).T[ind_Kq-1],
                         muh_LKO_vals_testpoint.T.mean(axis = 1)],\
                           columns = ['lower','upper', 'pred']),\
                
               }

    return [pd.concat(PIs_dict.values(), axis=1, keys=PIs_dict.keys()), times]


# simulation
n_list = [50, 100, 150, 200, 300, 500]
ntrial = 10
alpha = 0.1
test_scale = 1.5
d = 10
dgp_list = ['linear']  
method_names = ['XBART','XBART-GP','jackknife+ XBART','jackknife+ RF','CV+ XBART', 'CV+ RF']


results = pd.DataFrame(columns = ['itrial','n', 'dgp','method','rmse','coverage','width','coverage_type','time','num_outliers'])


filename = 'results/xbart_gp_time.csv'
if not exists(filename):
    print("create csv file: " + filename)
    with open(filename, 'w',encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(list(results.columns))

print('test_scale: '+ str(test_scale))
print('method_names: ' + str(method_names))
print('Experiment: ' + note)

for dgp in dgp_list:
    print("dgp = " + str(dgp))
    for itrial in range(ntrial):
        
        results = pd.read_csv(filename)
        if sum((results['dgp']==dgp) & (results['itrial'] == itrial)) > 0:
            continue
       
        X = np.random.normal(size=(n_list[-1],d))
        Y = generate_data(X, dgp) + np.random.normal(size=n_list[-1])

        # standard normal
        X1 = np.random.normal(size=(n_list[-1],d), scale = test_scale)
        Y1 = generate_data(X1, dgp) + np.random.normal(size=n_list[-1])
        
        for n in tqdm(n_list):

            X_min = np.apply_along_axis(lambda x: x.min(), 0, X[:n, :])
            X_max = np.apply_along_axis(lambda x: x.max(), 0, X[:n, :])
            outliers = np.apply_along_axis(check_out_of_range, 1, X1[:n, :], x_min = X_min, x_max = X_max)
            num_outliers = sum(outliers)
            
            PIs,times = compute_PIs(X[:n,:],Y[:n], X1[:n,:], alpha,random_forest)
            
            with open(filename, 'a',encoding='UTF8') as f:
                writer = csv.writer(f)

                for method in method_names:
            
                    rmse = np.sqrt(np.mean((PIs[method]['pred']-Y1[:n])**2))
                    coverage = ((PIs[method]['lower'] <= Y1[:n])&(PIs[method]['upper']>= Y1[:n])).mean()
                    width = (PIs[method]['upper'] - PIs[method]['lower']).mean()
                    writer.writerow([itrial, n, dgp, method, rmse, coverage, width, 'Overall', times[method], num_outliers])
                    
                    rmse = np.sqrt(np.mean(( (PIs[method]['pred']-Y1[:n])**2)[np.invert(outliers)] ))
                    coverage = ((PIs[method]['lower'] <= Y1[:n])&(PIs[method]['upper']>= Y1[:n]))[np.invert(outliers)].mean()
                    width = (PIs[method]['upper'] - PIs[method]['lower'])[np.invert(outliers)].mean()
                    writer.writerow([itrial, n, dgp, method, rmse, coverage, width, 'Interior', times[method], num_outliers])

                    rmse = np.sqrt(np.mean(( (PIs[method]['pred']-Y1[:n])**2)[outliers] ))
                    coverage = ((PIs[method]['lower'] <= Y1[:n])&(PIs[method]['upper']>= Y1[:n]))[outliers].mean()
                    width = (PIs[method]['upper'] - PIs[method]['lower'])[outliers].mean()
                    writer.writerow([itrial, n, dgp, method, rmse, coverage, width, 'Exterior', times[method], num_outliers])


                    


