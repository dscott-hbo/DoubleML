#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:43:57 2021

@author: danielscott
"""

import numpy as np
from sklearn.datasets import make_spd_matrix
import math
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from doubleml import DoubleMLData
from doubleml import DoubleMLPLR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


seed = 123
np.random.seed(seed)

N = 500 # No. obs 

k=100 # = No. variables in x_i 

theta=0.5 #real causal value

# Generate some fake weights for X used in generation of g and m functions
b= [1/k for k in range(1,((k+1)))]

#generate covariance matrix to be used in data generation
sigma = make_spd_matrix(k,random_state=seed) # 

MC_no = 250 # Number of simulations 

#generate predefined G as sine squared func
def g(x):
    return np.power(np.sin(x),2)

#generate predefined m as wrapped cauchy distribution with
# (nu)v=0 and gamma = 1#
def m(x,nu=0.,gamma=1.):
    return 0.5/math.pi*(np.sinh(gamma))/(np.cosh(gamma)-np.cos(x-nu))

# Create empty array to store estimated thetas     
theta_est = np.zeros(shape=[MC_no,6])

#for each simulation
for i in range(MC_no):
    
    # Generate data: 
    #no. obs x no. variables in x_i     
    X = np.random.multivariate_normal(np.ones(k),sigma,size=[N,])
    
    #generate function g of data X * bias
    G_of_X = g(np.dot(X,b)) 

    #generate function m of data X * bias
    M_of_X = m(np.dot(X,b)) 
    
    error_d = np.random.standard_normal(size=[N,]) 
    #generate D treatment
    D = M_of_X+error_d
    
    error_y = np.random.standard_normal(size=[N,])
    
    #generate Y
    Y = np.dot(theta,D)+ G_of_X + error_y 
    
    #
    # Now run the different methods     # 
    ######################################################################    
    # OLS --------------------------------------------------     
    ######################################################################
    OLS = sm.OLS(Y,D)
    results = OLS.fit()
    #save OLS estimate into theta_estimate array
    theta_est[i][0] = results.params[0]
    

    ######################################################################
    # DML package                      -----------------------------------     
    ######################################################################
    # DML DML2 algo                    -----------------------------------     
    ######################################################################
    dml_data = DoubleMLData.from_arrays(X, Y, D)

    ml_g_rf = RandomForestRegressor(max_depth=2)
    ml_m_rf = RandomForestRegressor(max_depth=2)
    
    dml_plr_tree = DoubleMLPLR(dml_data,
                             ml_g = ml_g_rf,
                             ml_m = ml_m_rf,
                             n_folds = 2,
                             n_rep = 1,
                             score = 'partialling out',
                             dml_procedure = 'dml1')
    #estimation
    dml_plr_tree.fit()
    
    #coefficient estimate
    theta_est[i][1] = dml_plr_tree.coef
    
    
    ######################################################################
    # Naive double machine Learning 
    # With orthogonal (for regularisation)
    # No Cross fitting (for bias)
    ######################################################################
    
    naiveDMLg =RandomForestRegressor(max_depth=4)

    # Compute ghat     
    naiveDMLg.fit(X,Y)
    Ghat = naiveDMLg.predict(X)
    
    # Compute ghat     
    naiveDMLm =RandomForestRegressor(max_depth=4)
    naiveDMLm.fit(X,D)
    Mhat = naiveDMLm.predict(X)
    
    # vhat as residual, orthogonalized regressors     
    Vhat = D-Mhat
    
    #save Naive ML estimate into theta_estimate array
    theta_est[i][2] = np.mean(np.dot(Vhat,Y-Ghat))/np.mean(np.dot(Vhat,D))


    ######################################################################
    # Cross-fitting DML -----------------------------------     
    ######################################################################
    # Split the sample   
    kf = KFold(n_splits = 2)
    kf.get_n_splits(X)
    
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        D_train, D_test = D[train_index], D[test_index]
    
    # Ghat for both     
    Ghat_1 = RandomForestRegressor(max_depth=2).fit(X_train,y_train).predict(X_test)
    Ghat_2 = RandomForestRegressor(max_depth=2).fit(X_test,y_test).predict(X_train)
    # Mhat and vhat for both     
    Mhat_1 = RandomForestRegressor(max_depth=2).fit(X_train,D_train).predict(X_test)
    Mhat_2 = RandomForestRegressor(max_depth=2).fit(X_test,D_test).predict(X_train)    
    
    #orthogonalized regressors
    Vhat_1 = D_test -  Mhat_1
    Vhat_2 = D_train - Mhat_2
    
    theta_1 = np.mean(np.dot(Vhat_1,(y_test-Ghat_1)))/np.mean(np.dot(Vhat_1,D_test))
    theta_2 = np.mean(np.dot(Vhat_2,(y_train-Ghat_2)))/np.mean(np.dot(Vhat_2,D_train))
    
    #get avg of each theta estimate and store
    theta_est[i][3] = np.mean([[theta_1, theta_2]])
        

#generate density plot
g = sns.displot(theta_est, kind="kde", legend=True).set(title='distribution of estimating theta with {} runs'.format(MC_no))
# title
new_title = 'Method'
g._legend.set_title(new_title)
# replace labels
new_labels = ['OLS','DML package' ,'Naive DML','Cross fit DML']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
ols_mean = theta_est[:,0].mean()
plt.axvline(ols_mean, c = 'navy', ls = '--', alpha = 0.3)
naive_mean = theta_est[:,1].mean()
plt.axvline(naive_mean, c='orange',ls = '--', alpha = 0.7)
crossfit_dml_mean = theta_est[:,2].mean()
plt.axvline(crossfit_dml_mean, c = 'green',ls = '--', alpha = 0.3)
plt.axvline(theta, c = 'red',ls = '-', alpha = 0.8)