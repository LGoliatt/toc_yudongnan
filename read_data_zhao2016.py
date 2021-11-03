# -*- coding: utf-8 -*
import numpy as np
import pandas as pd
import seaborn as sns
import pylab as pl    
from sklearn.model_selection import train_test_split

pl.rc('text', usetex=False)
pl.rc('font',**{'family':'serif','serif':['Palatino']})
#%%
def read_zhao2016():
    #%%         
    X = pd.read_csv('./data/data_zhao2016/zhao2016.csv', sep=';', header=0)
    X.to_latex(buf='zhao_dataset.tex', index=False)
    X.drop(['Sample', 'Depth'], axis=1, inplace=True)
    target_names=['TOC',]
    variable_names = list(X.columns.drop(target_names))
    
    df = X[variable_names+target_names].copy()
    df.columns = [x.replace('(wt%)','') for x in df.columns]
    sns.set_context("paper")

    pl.figure(figsize=(5, 4))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    heatmap = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    #heatmap = sns.heatmap(corr, mask=None, vmin=-1, vmax=1, annot=True, cmap='BrBG',)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=0);
    pl.savefig('zhao_heatmap_correlation.png',  bbox_inches='tight', dpi=300)
    pl.show()
    
    #sns.pairplot(df)
    #pl.show()
    #%%
    X_train=X[variable_names]
    y_train=X[target_names]
   
    #X_train, X_test, y_train, y_test = train_test_split(
    #                                  X_train, y_train, test_size=0.3, random_state=None)
    
    n=len(y_train);     
    n_samples, n_features = X_train.shape 
         
    regression_data =  {
      'task'            : 'regression',
      'name'            : 'YuDongNan',
      'feature_names'   : np.array(variable_names),
      'target_names'    : target_names,
      'n_samples'       : n_samples, 
      'n_features'      : n_features,
      'X_train'         : X_train.values,
      'y_train'         : y_train.values.T,
       #'X_test'          : X_test.values,
       #'y_test'          : y_test.values.T,
      'X_test'          : [],
      'y_test'          : [[],[]],
      'targets'         : target_names,
      'true_labels'     : None,
      'predicted_labels': None,
      'descriptions'    : 'None',
      'reference'       : "https://bit.ly/31MSe2S",
      'items'           : None,
      'normalize'       : None,
      }
    #%%
    return regression_data

#%%

