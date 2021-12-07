#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 16:16:11 2021

Script to visualise the effect of adding more features to a synthetic causal
dataset. 

Inputs required:
    generate features parameters
    maximum number of simulations (MC_no)
    maximum number of features to add to the system
    The causal inference factor (c)


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

seed = 123
np.random.seed(seed)

def gen_features(n_samples,n_features, c, distribution, noise):
    if distribution == 'gaussian':
        X = np.random.randn(n_samples, n_features)
        D =  np.random.binomial(1,0.5,size=n_samples) # treatment var
        weights = np.random.rand(n_features)
        c = c
        Y = np.dot(X,weights) + c*D
        if noise >0.0:
            Y += np.random.normal(scale=noise, size = Y.shape)    
    
    elif distribution == 'bernoulli':
        X = bernoulli.rvs(0.9, size=[n_samples,n_features], random_state = seed)
        D =  np.random.binomial(1,0.5,size=n_samples) # treatment var
        weights = np.random.rand(n_features)
        c = c
        Y = np.dot(X,weights) + c*D
        if noise >0.0:
            Y += np.random.normal(scale=noise, size = Y.shape)  
    else:
        print("Not a valid distribution")
    
    return X, Y, D


#create empty df to hold 
df = pd.DataFrame(columns=['Run', 'OLS', 'DML1', 'No_feats', 'No_samples','coef', 'std err','t', 'P>|t|','2.5 %','97.5 %'])

#define simulation parameters
c = 0.5
num_runs = [1,2,3,4,5]
data_distribution = "bernoulli" #gaussian/bernoulli
num_samples = [1000,100000,1000000]
num_feats = [10,30]
for run in num_runs:
    for feature_num in num_feats:
        for samples in num_samples:
            #generate data
            X,Y,D = gen_features(n_samples=samples,n_features=feature_num,c=c,distribution=data_distribution,noise=0)
                
            # Now run the different methods     # 
            ######################################################################    
            # OLS --------------------------------------------------     
            ######################################################################
            OLS = sm.OLS(Y,D)
            results = OLS.fit()        
            
            ####
            ##Add confidence interval for OSL
            #LB = results.conf_int(alpha=0.05, cols=None)[0][0]
            #UP = results.conf_int(alpha=0.05, cols=None)[0][2]            
         
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
                                     ml_m = ml_m_rf)
            #estimation
            dml_plr_tree.fit()
            
            #append coefficient to coefficient df
            row = [run,results.params[0],dml_plr_tree.coef,feature_num,samples,dml_plr_tree.coef,dml_plr_tree.se,dml_plr_tree.t_stat,dml_plr_tree.pval,dml_plr_tree.confint().iloc[0][0],dml_plr_tree.confint().iloc[0][1]]
            df.loc[len(df)] = row
        
        
#clean values in dataframe
df[['DML1', 'coef','std err','t','P>|t|']] = df[['DML1', 'coef','std err','t','P>|t|']].apply(lambda x: x.str[0])


mean_df = df.groupby(['No_feats'], axis=0).mean()

#write data to machine
df.to_pickle()


temp_df = df.query('No_feats== {} & No_samples == {}'.format(11,100000))
temp_df = temp_df.filter(items=['OLS', 'DML1'])
g = sns.displot(temp_df, kind="kde", legend=True).set(title='distribution of estimating theta with {} features with {} samples'.format(1,1000))

#set visualisation settings
sns.set()
colors = sns.color_palette()

plt.rcParams['figure.figsize'] = 10., 7.5
sns.set(font_scale=1.5)
sns.set_style('whitegrid', {'axes.spines.top': False,
                            'axes.spines.bottom': False,
                            'axes.spines.left': False,
                            'axes.spines.right': False})
#function to plot CI of the runs
def generate_CI_graph(df,feats,samples):
    temp_df2 = df.query('No_feats== {} & No_samples == {}'.format(feats,samples))
    temp_df = np.full((2, temp_df2.shape[0]), np.nan)
    temp_df[0, :] = temp_df2['coef'] - temp_df2['2.5 %']
    temp_df[1, :] = temp_df2['97.5 %'] - temp_df2['coef']
    plt.errorbar([1,2,3], temp_df2.coef, fmt='o', yerr=temp_df)
    plt.axhline(y=c, color='r', linestyle='-')
    plt.title('No_feats== {} & No_samples == {}'.format(feats,samples))
    plt.xlabel('Run')
    plt.ylim(0.3,0.7) #standardise visualisation
    plt.xticks(np.arange(1, 4, 1))  # Set label locations for three runs. 
    _ =  plt.ylabel('Coefficients and 95%-CI')
    plt.show()

num_feats = [1,11,21,31,41,51,61,71,81,91]    
for feature_num in num_feats:
        for samples in num_samples:
            generate_CI_graph(df,feature_num,samples)
            

def generate_CI_graph(df,feats,samples):
    temp_df2 = df.query('No_feats== {} & No_samples == {}'.format(feats,samples))
    temp_df = np.full((2, temp_df2.shape[0]), np.nan)
    temp_df[0, :] = temp_df2['coef'] - temp_df2['2.5 %']
    temp_df[1, :] = temp_df2['97.5 %'] - temp_df2['coef']
    plt.errorbar(temp_df2.Run, temp_df2.coef, fmt='o', yerr=temp_df)
    plt.axhline(y=c, color='r', linestyle='-')
    plt.title('No_feats== {} & No_samples == {}'.format(feats,samples))
    plt.xlabel('Run')
    _ =  plt.ylabel('Coefficients and 95%-CI')
    plt.show()

            
def check_vals_in_CI(df,c):
    


def generate_distribution_graphs(df,feature_num)
    temp_df = df.query('No_feats== {} & No_samples == {}'.format(i,samples))
    temp_df = temp_df.filter(items=['OLS', 'DML1','DML2'])
    g = sns.displot(temp_df, kind="kde", legend=True).set(title='distribution of estimating theta with {} features with {} samples'.format(i,samples))


try:
    
    #generate distribution per number of features used
    for i in range(1,max_runs,steps):
        for samples in num_samples:
            #iterate through each 
            temp_df = df.query('No_feats== {} & No_samples == {}'.format(i,samples))
            temp_df = temp_df.filter(items=['OLS', 'DML1','DML2'])
            #generate density plot
            g = sns.displot(temp_df, kind="kde", legend=True).set(title='distribution of estimating theta with {} features with {} samples'.format(i,samples))
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
except:
    pass
