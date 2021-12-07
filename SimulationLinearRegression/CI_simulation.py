#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 10:27:22 2021

Simulation for DML vs OLS with linear data. Generates data and iterates through different scenarios. 
Final functions generate CI graphs for each scenario.

@author: danielscott
"""

##Import libraries
import numpy as np
from scipy import stats
from scipy.stats import bernoulli
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from doubleml import DoubleMLData
from doubleml import DoubleMLPLR

########################
#### Generate data #####
########################
def gen_features(n_samples,n_features, c,treatment_prob, distribution, noise,seed):
    """
    Parameters
    ----------
    n_samples : int
        The number of rows of data to be generated
    n_features : int
        The number of features to be generated
    c : float
        The predefined treatment effect
    treatment_prob : float
        The probability of having treatment
    distribution : string
        The distribution of the covariates to be generated
    noise : float
        A noise factor
    seed : int
        seed

    Returns
    -------

    X : Array of float64
        The synthetic data acting as the covariates for the causal inference
    D: Array of int64
        A list of 1's or 0's representing being in a treatment or not
    dx: Array of float64
        Treatment variable stacked with the covariate data
    weights: Array of float64
        randomly generated weights for the linear 
    Y: Array of float64
        The outcome 
    """

    #Generate gaussian covariates
    if distribution == 'gaussian':
        X = np.random.randn(n_samples, n_features)
        D =  bernoulli.rvs(p = treatment_prob,size=n_samples) # treatment var
        dx = np.column_stack((D,X))
        trt_effect = c
        weights = np.append(trt_effect, stats.norm.rvs(0, 1, size = n_features))#add the constant then random variabel
        Y = np.dot(dx,weights.T)
        if noise >0.0:
            Y += stats.norm.rvs(loc = 0, scale=noise,size=Y.shape)     

    #Generate bivariate covariates    
    elif distribution == 'bernoulli':
        # generate X features
        p_vec = np.random.uniform(size = n_features)
        X = stats.bernoulli.rvs(p = p_vec, size = [n_samples,n_features])
        #add the treatment var into X data        
        D = bernoulli.rvs(p = treatment_prob,size=n_samples) # treatment var
        dx = np.column_stack((D,X))
        
        #generate weights
        trt_effect = c
        weights = np.append(trt_effect, stats.norm.rvs(0, 1, size = n_features))#add the constant then random variabel
        Y = np.dot(dx,weights.T)
        #implement noise for bernoulli with noise = p(flip bit)
        if noise >0.0:
            #Y_multiplier = [-1 if random.random() < noise else 1 for i in Y]
            #Y = Y * Y_multiplier
            Y += stats.norm.rvs(loc = 0, scale=noise,size=Y.shape)  
            
    else:
        print("Not a valid distribution")
    
    return X,dx,weights, Y, D


#create empty df to hold simulation results
df = pd.DataFrame(columns=['Run', 'OLS','OLS_LB','OLS_UB','std_err','DML','DML_err', 'DML_LB','DML_UB','DML_t_stat','DML_reg','DML_reg_err', 'DML_reg_LB','DML_reg_UB','DML_reg_t_stat','num_samples','distribution','num_features','noise'])

#####################################
#### Define simulation parameters ###
#####################################

c = 0.5 
num_runs = [1,2,3] #array of integers
data_distribution = ["bernoulli","gaussian"] #gaussian/bernoulli
num_samples = [10000,100000,500000,1000000] #array of integers
num_feats = [30] #array of integers
noises = [0,0.5,2] #array of floats. normally distributed noise centered around 0. This value changes the std
trt_prob = [0.9]
#set some seed
seed = 123
np.random.seed(seed)

############################
#### Execute simulation ####
############################
for run in num_runs:
    for feature_num in num_feats:
        for samples in num_samples:
            for dist in data_distribution:
                for noise in noises:
                    
                    #generate data
                    X,dx, weights,Y,D = gen_features(n_samples=samples,n_features=feature_num,c=c,treatment_prob=trt_prob,distribution=dist,noise=noise, seed=seed)
                    
                    ######################################################################    
                    # OLS --------------------------------------------------     
                    ######################################################################
                    OLS = sm.OLS(Y,dx)
                    results = OLS.fit()        
                    
                    ##Add confidence interval for OLS
                    ols_LB = results.conf_int(alpha=0.05, cols=None)[0][0]
                    ols_UB = results.conf_int(alpha=0.05, cols=None)[0][1]            
                    std_err = results.bse[0]
                    
                    ######################################################################
                    # DML package                      -----------------------------------     
                    ######################################################################
                    # DML ml_m_rf as classifier        -----------------------------------     
                    ######################################################################
                    dml_data = DoubleMLData.from_arrays(X, Y, D)
                    
                    ml_g_rf = RandomForestRegressor(n_estimators=100,max_depth=10, n_jobs = -1)
                    ml_m_rf = RandomForestClassifier(n_estimators=100,max_depth=10, n_jobs = -1)
                    
                    dml_plr_tree = DoubleMLPLR(dml_data, ml_g = ml_g_rf, ml_m = ml_m_rf)
                    #estimation
                    dml_plr_tree.fit()
                
                    ######################################################################
                    # DML package                      -----------------------------------     
                    ######################################################################
                    # DML ml_m_rf as regressor        -----------------------------------     
                    ######################################################################
                    
                    ml_m_rfreg = RandomForestRegressor(n_estimators=100,max_depth=10, n_jobs = -1)
                    
                    dml_plr_tree_reg = DoubleMLPLR(dml_data, ml_g = ml_g_rf, ml_m = ml_m_rfreg)
                    #estimation
                    dml_plr_tree_reg.fit()
                    
                    row = [run,results.params[0],ols_LB,ols_UB,std_err, dml_plr_tree.coef,dml_plr_tree.se, dml_plr_tree.confint().iloc[0][0],dml_plr_tree.confint().iloc[0][1],dml_plr_tree.t_stat,dml_plr_tree_reg.coef,dml_plr_tree_reg.se, dml_plr_tree_reg.confint().iloc[0][0],dml_plr_tree_reg.confint().iloc[0][1],dml_plr_tree_reg.t_stat,samples,dist, feature_num, noise]
                    df.loc[len(df)] = row


#clean values in dataframe
df[['DML','DML_err','DML_t_stat','DML_reg','DML_reg_err','DML_reg_t_stat']] = df[['DML','DML_err','DML_t_stat','DML_reg','DML_reg_err','DML_reg_t_stat']].apply(lambda x: x.str[0])

#Generate df with average values
mean_df = df.groupby(['distribution','num_samples','noise']).mean()

#set visualisation settings
sns.set()
colors = sns.color_palette()
plt.rcParams['figure.figsize'] = 10., 7.5
sns.set(font_scale=1.5)
sns.set_style('whitegrid', {'axes.spines.top': False,
                            'axes.spines.bottom': False,
                            'axes.spines.left': False,
                            'axes.spines.right': False})

####################
# Visualisation    #
####################

########################################
#VISUALISATION FUNCTIONS               #
########################################
#function to 
def extract_table_text(table):
    """
    Helper function which extracts data to be visualised in a table the 
    CI graph function
    """
    cell_text=[]
    for row in range(len(table)):
        cell_text.append(table.iloc[row].round(3))
    return cell_text    

def generate_CI_graph(df,feats,samples,num_runs,distribution,noise, method):
    """
    Generate a graph showing coefficients and CI's. The inputs are used as to 
    query the results dataframe.
    
    Parameters
    ----------
    df : df
        A df containing the results of the simulation
    feats : 
        The number of features to filter the df to
    samples : int
        The number of rows of data to be generated
    num_runs : int
        The number of rows of data to be generated
    distribution : string
        The distribution of the covariates to be generated
    noise : float
        A noise factor
    method : string
        valid inputs are either 'OlS' or 'DML'

    Returns
    -------
    Plot containing the filters provided above. Ideally should be run by 
    iterating over the same combinations used in data generation
    
    i.e. for feature_num in num_feats:
            for samples in num_samples:
                generate_CI_graph

    """
    meth = method
    meth_ub = method+"_UB"
    meth_lb = method + "_LB"
    #query dataset
    temp_df2 = df.query('num_features== {0} & num_samples=={1} & distribution=="{2}" & noise=={3}'.format(feats,samples,distribution, noise))
    temp_df = np.full((2, temp_df2.shape[0]), np.nan)
    temp_df[0, :] = temp_df2[method] - temp_df2[meth_lb]
    temp_df[1, :] = temp_df2[meth_ub] - temp_df2[meth]
    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)
    
    plt.errorbar(num_runs, temp_df2[meth], fmt='o', yerr=temp_df)
    
    #plot tables
    data_table = temp_df2.filter(items=[method, meth_lb,meth_ub])
    cell_text = extract_table_text(data_table)
    plt.table(cellText=cell_text, colLabels=data_table.columns, cellLoc = 'center', 
              rowLoc = 'center',loc='bottom',bbox=[0.0, -0.4, 0.9, 0.2])
    #plot real inference value as red horizontal line
    plt.axhline(y=c, color='r', linestyle='-')
    plt.title('{} coefficients with CI. No_feats {} & No_samples {}, {} distributed data, noise {}'.format(method,feats,samples,dist,noise))
    plt.xlabel('Run')
    plt.ylim(0.3,0.7) #standardise visualisation
    plt.xticks(np.arange(1, max(num_runs)+1, 1))  # Set label locations for three runs. 
    _ =  plt.ylabel('Coefficients and 95%-CI')   
    plt.show()
    
####################
#generate plots    #
####################
for feature_num in num_feats:
        for samples in num_samples:
            for dist in ['bernoulli','gaussian']:
                for noise in noises:
                    for meth in ["OLS","DML"]:
                        generate_CI_graph(df,feature_num,samples,num_runs,dist,noise,meth)
                        
                    
                    
