#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:16:11 2021

@author: danielscott
"""

import numpy as np
from scipy.stats import bernoulli
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from doubleml import DoubleMLData
from doubleml import DoubleMLPLR

def gen_features(n_samples,n_features, c, noise):
    X = np.random.randn(n_samples, n_features)
    D =  np.random.binomial(1,0.5,size=n_samples) # treatment var
    weights = np.random.rand(n_features)
    c = c
    Y = np.dot(X,weights) + c*D
    if noise >0.0:
        Y += np.random.normal(scale=noise, size = Y.shape)   
    return X, Y, D

seed = 123
np.random.seed(seed)



DML1 = pd.DataFrame(columns=['coef', 'std err', 'P>|t|','2.5%','97.5%'])

#X,y,D = gen_features(10,5,0.6,0)
df = pd.DataFrame(columns=['Run', 'OLS', 'DML1', 'No_feats'])

# Create empty array to store estimated thetas     
#theta_est = np.zeros(shape=[50,3])
MC_no = 100
max_no_feats = 100
c = 0.5

for s in range(MC_no):
    for i in range(1,max_no_feats):
        #generate data
        X,Y,D = gen_features(n_samples=500,n_features=i,c=c,noise=0)
            
        # Now run the different methods     # 
        ######################################################################    
        # OLS --------------------------------------------------     
        ######################################################################
        OLS = sm.OLS(Y,D)
        results = OLS.fit()        
        
        #save OLS estimate into theta_estimate array
        #theta_est[i][0] = results.params[0]
        
     
        ######################################################################
        # DML package                      -----------------------------------     
        ######################################################################
        # DML DML1 algo                    -----------------------------------     
        ######################################################################
        dml_data = DoubleMLData.from_arrays(X, Y, D)
    
        ml_g_rf = RandomForestRegressor(n_estimators=100,max_depth=10, n_jobs = -1)
        ml_m_rf = RandomForestRegressor(n_estimators=100,max_depth=10, n_jobs = -1)
        
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
        #theta_est[i][1] = dml_plr_tree.coef
        DML1 = DML1.append(dml_plr_tree.summary)
        ######################################################################
 
        row = [s,results.params[0],dml_plr_tree.coef,i]
        df.loc[len(df)] = row
        

df['DML1'] = df['DML1'].str[0]


mean_df = df.groupby(['No_feats'], axis=0).mean()


#produce to graphs
for i in range(1,max_no_feats):
    temp_df = df.query('No_feats== {}'.format(i))
    temp_df = temp_df.filter(items=['OLS', 'DML1','DML2'])

    #generate density plot
    g = sns.displot(temp_df, kind="kde", legend=True).set(title='distribution of estimating theta with {} features'.format(i))
    # title
    new_title = 'Method'
    g._legend.set_title(new_title)
    # replace labels
    new_labels = ['OLS','DML1 package']
    for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
    ols_mean = temp_df['OLS'].mean()
    plt.axvline(ols_mean, c = 'navy', ls = '--', alpha = 0.6)
    plt.text(0, .5, "OLS:"+str(round(ols_mean,2)), horizontalalignment='left', size='medium', color='black')
    naive_mean = temp_df['DML1'].mean()
    plt.axvline(naive_mean, c='orange',ls = '--', alpha = 0.7)
    plt.text(0, 0.35, "DML1: "+ str(round(naive_mean,2)), horizontalalignment='left', size='medium', color='black')

    plt.axvline(0.5, c = 'red',ls = '-', alpha = 0.3)

