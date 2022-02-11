


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
df = pd.DataFrame(columns=['Run', 'OLS','OLS_LB_95','OLS_UB_95', 'DML1', 'No_feats', 'No_samples','coef', 'std err','t', 'P>|t|','2.5 %','97.5 %'])

#define simulation parameters
c = 0.5
num_runs = [1,2,3,4,5]
data_distribution = "bernoulli" #gaussian/bernoulli
num_samples = [1000,10000,500000,1000000]
num_feats = [10,30,60]


for run in num_runs:
    for feature_num in num_feats:
        for samples in num_samples:
            for dist in ['bernoulli','gaussian']:
                #generate data
                X,Y,D = gen_features(n_samples=samples,n_features=feature_num,c=c,distribution=dist,noise=0)
                
                # Now run the different methods     # 
                ######################################################################    
                # OLS --------------------------------------------------     
                ######################################################################
                OLS = sm.OLS(Y,D)
                results = OLS.fit()        
                
                ####
                ##Add confidence interval for OSL
                ols_LB = results.conf_int(alpha=0.05, cols=None)[0][0]
                ols_UB = results.conf_int(alpha=0.05, cols=None)[0][1]            
                
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
                row = [run,results.params[0],ols_LB,ols_UB,dml_plr_tree.coef,feature_num,samples,dml_plr_tree.coef,dml_plr_tree.se,dml_plr_tree.t_stat,dml_plr_tree.pval,dml_plr_tree.confint().iloc[0][0],dml_plr_tree.confint().iloc[0][1]]
                df.loc[len(df)] = row


#clean values in dataframe
df[['DML1', 'coef','std err','t','P>|t|']] = df[['DML1', 'coef','std err','t','P>|t|']].apply(lambda x: x.str[0])


#set visualisation settings
sns.set()
colors = sns.color_palette()

plt.rcParams['figure.figsize'] = 10., 7.5
sns.set(font_scale=1.5)
sns.set_style('whitegrid', {'axes.spines.top': False,
                            'axes.spines.bottom': False,
                            'axes.spines.left': False,
                            'axes.spines.right': False})

#function to extract data to be visualised in a table the CI graph function
def extract_table_text(table):
    cell_text=[]
    for row in range(len(table)):
        cell_text.append(table.iloc[row].round(3))
    return cell_text    
    
def generate_CI_graph(df,feats,samples,num_runs,distribution):
    #query dataset
    temp_df2 = df.query('No_feats== {0} & No_samples=={1} & distribution=="{2}"'.format(feats,samples,distribution))
    temp_df = np.full((2, temp_df2.shape[0]), np.nan)
    temp_df[0, :] = temp_df2['coef'] - temp_df2['2.5 %']
    temp_df[1, :] = temp_df2['97.5 %'] - temp_df2['coef']
    
    #plt.axis('off')
    
    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)
 
    plt.errorbar(num_runs, temp_df2.coef, fmt='o', yerr=temp_df) 


    data_table = temp_df2.filter(items=['OLS', 'OLS_LB_95','OLS_LB_95','DML1','2.5 %','97.5 %'])
    cell_text = extract_table_text(data_table)
    plt.table(cellText=cell_text, colLabels=data_table.columns, cellLoc = 'center', 
              rowLoc = 'center',loc='bottom',bbox=[0.0, -0.4, 0.9, 0.2])

    
    plt.axhline(y=c, color='r', linestyle='-')
    plt.title('No_feats {} & No_samples {}, {}'.format(feats,samples,dist))
    plt.xlabel('Run')
    plt.ylim(0.3,0.7) #standardise visualisation
    plt.xticks(np.arange(1, max(num_runs)+1, 1))  # Set label locations for three runs. 
    _ =  plt.ylabel('Coefficients and 95%-CI')
    

   
    plt.show()


for feature_num in num_feats:
        for samples in num_samples:
            for dist in ['bernoulli','gaussian']:
                generate_CI_graph(df,feature_num,samples,num_runs,dist)