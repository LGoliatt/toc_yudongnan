#!/usr/bin/python
# -*- coding: utf-8 -*-    
import numpy as np
import pandas as pd
#import pygmo as pg
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_predict, 
                                     TimeSeriesSplit, cross_val_score, 
                                     LeaveOneOut, KFold, StratifiedKFold,
                                     cross_val_predict,train_test_split)
from sklearn.metrics import r2_score, mean_squared_error, max_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, MaxAbsScaler, Normalizer, StandardScaler, MaxAbsScaler, FunctionTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline

from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, Ridge, PassiveAggressiveRegressor, LogisticRegression, BayesianRidge, LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor#, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
#from xgboost import  XGBRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)


import re
#from sklearn.gaussian_process import GaussianProcess
#from catboost import Pool, CatBoostRegressor
#from pyearth import Earth as MARS
#from sklearn.ensemble import StackingRegressor
#from sklearn.experimental import enable_hist_gradient_boosting
#from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_approximation import RBFSampler,SkewedChi2Sampler

from util.ELM import  ELMRegressor, ELMRegressor
#from util.MLP import MLPRegressor as MLPR
#from util.RBFNN import RBFNNRegressor, RBFNN
#from util.LSSVR import LSSVR
#from gmdhpy.gmdh import MultilayerGMDH

from scipy import stats
from hydroeval import kge, nse

#from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

#%%----------------------------------------------------------------------------
#pd.options.display.float_format = '{:20,.3f}'.format
pd.options.display.float_format = '{:.3f}'.format

import warnings
warnings.filterwarnings('ignore')

import sys, getopt
program_name = sys.argv[0]
arguments = sys.argv[1:]
count = len(arguments)

if len(arguments)>0:
    if arguments[0]=='-r':
        run0 = int(arguments[1])
        n_runs = run0
else:
    run0, n_runs = 0,100
#%%----------------------------------------------------------------------------   
def accuracy_log(y_true, y_pred):
    y_true=np.abs(np.array(y_true))
    y_pred=np.abs(np.array(y_pred))
    return (np.abs(np.log10(y_true/y_pred))<0.3).sum()/len(y_true)*100

def rms(y_true, y_pred):
    y_true=np.abs(np.array(y_true))
    y_pred=np.abs(np.array(y_pred))
    return ( (np.log10(y_pred/y_true)**2).sum()/len(y_true) )**0.5
#------------------------------------------------------------------------------   
def lhsu(xmin,xmax,nsample):
   nvar=len(xmin); ran=np.random.rand(nsample,nvar); s=np.zeros((nsample,nvar));
   for j in range(nvar):
       idx=np.random.permutation(nsample)
       P =(idx.T-ran[:,j])/nsample
       s[:,j] = xmin[j] + P*(xmax[j]-xmin[j]);
       
   return s
#------------------------------------------------------------------------------   
def RMSE(y, y_pred):
    y, y_pred = np.array(y).ravel(), np.array(y_pred).ravel()
    error = y -  y_pred    
    return np.sqrt(np.mean(np.power(error, 2)))
#------------------------------------------------------------------------------   
def RRMSE(y, y_pred):
    y, y_pred = np.array(y).ravel(), np.array(y_pred).ravel()
    return RMSE(y, y_pred)*100/np.mean(y)
#------------------------------------------------------------------------------   
def MAPE(y_true, y_pred):    
  y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
  return np.mean(np.abs(y_pred - y_true)/np.abs(y_true))*100
  #return RMSE(y, y_pred)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------                
import glob as gl
import pylab as pl
import os
basename='iccsa2021_'
#%%
from read_data_zhao2016 import *
datasets = [
            read_zhao2016(),
            ]     
#%%----------------------------------------------------------------------------   
pl.show()

pd.options.display.float_format = '{:.3f}'.format
n_splits    = 10
scoring     = 'neg_root_mean_squared_error'
for run in range(run0, n_runs):
    random_seed=run*10
    
    for dataset in datasets:#[:1]:
        dr=dataset['name'].replace(' ','_').replace("'","").lower()
        path='./pkl_'+dr+'/'
        os.system('mkdir  '+path)
        
        for tk, tn in enumerate(dataset['target_names']):
            print (tk, tn)
            dataset_name = dataset['name']+'-'+tn
            target                          = dataset['target_names'][tk]
            y_train, y_test                 = dataset['y_train'][tk], dataset['y_test'][tk]
            X_train, X_test                 = dataset['X_train'], dataset['X_test']
            n_samples_train, n_features     = dataset['n_samples'], dataset['n_features']
            task, normalize                 = dataset['task'], dataset['normalize']
            n_samples_test                  = len(y_test)
            
            s=''+'\n'
            s+='='*80+'\n'
            s+='Dataset                    : '+dataset_name+' -- '+target+'\n'
            s+='Output                     : '+tn+'\n'
            s+='Number of training samples : '+str(n_samples_train) +'\n'
            s+='Number of testing  samples : '+str(n_samples_test) +'\n'
            s+='Number of features         : '+str(n_features)+'\n'
            s+='Normalization              : '+str(normalize)+'\n'
            s+='Task                       : '+str(dataset['task'])+'\n'
            #s+='Reference                  : '+str(dataset['reference'])+'\n'
            s+='='*80
            s+='\n'            
            feature_names = dataset['feature_names']
            print(s)    
            #------------------------------------------------------------------
            args = (X_train, y_train, X_test, y_test, 'eval', task,  n_splits, 
                    int(random_seed), scoring, target, 
                    n_samples_train, n_samples_test, n_features)
            #------------------------------------------------------------------         
            params_gmdh = {
                'ref_functions': ('linear_cov', 'quadratic', 'cubic', 'linear'),
                #'criterion_type': 'test_bias',
                'seq_type': 'random',
                'feature_names': feature_names,
                'min_best_neurons_count':5, 
                'criterion_minimum_width': 5,
                'admix_features': False,
                'max_layer_count':20,
                'stop_train_epsilon_condition': 0.000001,
                'layer_err_criterion': 'top',
                #'alpha': 0.5,
                'normalize':False,
                'n_jobs': 1
                }   
            
            lr = LinearRegression()
            svr_lin = SVR(kernel='linear')

            cv=3
            ridgecv = GridSearchCV(estimator=Ridge(random_state=random_seed), 
                param_grid={
                    'alpha':[0.  , 0.1 ,  0.3 ,  0.5, 1]+[2,5,10,50,100,500,100]
                    }, 
                cv=cv, scoring='neg_root_mean_squared_error', n_jobs=6,
                refit=True)
                                     
            mlpcv = GridSearchCV(estimator=MLPRegressor(random_state=random_seed), 
                param_grid={
                    'hidden_layer_sizes':[(20),(30),(50), (100), (20,20), (50,50), (10,10,10)],
                    'activation':['identity', 'logistic', 'tanh', 'relu'],
                    },
                cv=cv, scoring='neg_root_mean_squared_error', n_jobs=6,
                refit=True)

            elmcv = GridSearchCV(ELMRegressor(alpha=1, random_state=random_seed), 
                param_grid={
                    'n_hidden':[20,30,40,50,100,150,200,300,500],
                    'activation_func':['identity', 'logistic', 'tanh', 'relu',
                                  'swish', 'gaussian', 'multiquadric'],
                    },
                cv=cv, scoring='neg_root_mean_squared_error', n_jobs=6,
                refit=True)
                       
            svr = GridSearchCV(SVR(max_iter=1000), 
                param_grid = {'C':[0.001, 0.01, 0.1, 10, 50, 100, 500, 1000, 10000],
                              'kernel':['rbf','poly','sigmoid','linear'],
                              'epsilon':[0.001,0.001,0.01,0.1, 0.5, 1, 10, 50, 100]
                              }  ,           
                cv=cv, scoring='neg_root_mean_squared_error',
                refit=True)
            
            optimizers=[             
                ('RR'  ,  args, random_seed, ridgecv),
                ('SVR' ,  args, random_seed, svr),
                ('ELM'  ,  args, random_seed, elmcv),
            ]
                           
            for (clf_name, args, random_seed, clf) in optimizers:
                np.random.seed(random_seed)
                list_results=[]
                #--------------------------------------------------------------
                s=''
                s='-'*80+'\n'
                s+='Estimator                  : '+clf_name+'\n'
                #s+='Function                   : '+str(fun)+'\n'
                s+='Dataset                    : '+dataset_name+' -- '+target+'\n'
                s+='Output                     : '+tn+'\n'
                s+='Run                        : '+str(run)+'\n'
                s+='Random seed                : '+str(random_seed)+'\n'
                                
                #s+='Optimizer                  : '+algo.get_name()+'\n'                
                s+='-'*80+'\n'
                print(s)
               
                
                if len(X_test)>1:
                    pass
                else:
                    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
                    y_train_pred=y_train.copy()
                    for train_index, test_index in kf.split(X_train):
                         xx_train, xx_test = X_train[train_index], X_train[test_index]
                         yy_train, yy_test = y_train[train_index], y_train[test_index]
                         clf.fit(xx_train, yy_train)
                         y_train_pred[test_index] = clf.predict(xx_test).ravel()

                #%%                
                sim={'Y_TRAIN_TRUE':y_train, 'Y_TRAIN_PRED':y_train_pred,
                     'EST_NAME':clf_name}
                
                sim['ALGO'] = clf_name
                sim['EST_PARAMS']=clf.best_estimator_.get_params()
                sim['OPT_PARAMS']=clf.best_params_
                #print(clf.best_estimator_.get_params())
                
                sim['OUTPUT'] = sim['TARGET'] = target
                sim['SEED']=random_seed
                sim['ACTIVE_VAR_NAMES']=sim['ACTIVE_VAR']=dataset['feature_names']#[sim['ACTIVE_VAR']]
                sim['SCALER']=None
                pl.figure()#(random_seed+0)
                pl.plot(sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_TRUE'].ravel(), 'r-', 
                            sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_PRED'].ravel(), 'b.' )
                r2=r2_score(sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_PRED'].ravel())
                r=stats.pearsonr(sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_PRED'].ravel())[0]
                rmse=RMSE(sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_PRED'].ravel())  
                #rmsl=rms(sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_PRED'].ravel())                  
                #mape=MAPE(sim['Y_TRAIN_TRUE'].ravel(), sim['Y_TRAIN_PRED'].ravel())                  
                pl.ylabel(dataset_name)
                pl.title(sim['EST_NAME']+': (Training) R$^2$='+str('%1.3f' % r2)+'\t RMSE='+str('%1.3f' % rmse)
                                #+'\t MAPE='+str('%1.3f' % mape)
                                +'\n R='+str('%1.3f' % r)
                                +'\t MSE='+str('%1.3f' % rmse**2)
                              #+', '.join(sim['ACTIVE_VAR_NAMES'])
                              )      
                pl.axes().set_aspect('equal', )
                pl.show()
                #%%
                if n_samples_test > 0:    
                    pl.figure()#(random_seed+1)
                    #pl.plot(sim['Y_TEST_TRUE'].ravel(), 'r-', sim['Y_TEST_PRED'].ravel(), 'b-' )
                    pl.plot(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_TRUE'].ravel(), 'r-', 
                            sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel(), 'b.' )
                    r2=r2_score(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())
                    r=stats.pearsonr(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())[0]
                    rmse=RMSE(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())                
                    rmsl=rms(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())     
                    mape=MAPE(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())   
                    acc=accuracy_log(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())  
                    kge_=kge(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())[0][0]
                    nse_=nse(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())
                    pl.ylabel(dataset_name)
                    pl.title(sim['EST_NAME']+': (Testing) R$^2$='+str('%1.3f' % r2)+'\t RMSE='+str('%1.3f' % rmse)
                                    +'\t MAPE='+str('%1.3f' % mape)
                                    +'\n R='+str('%1.3f' % r)
                                    +'\t NSE='+str('%1.3f' % nse_)
                                    +'\t KGE='+str('%1.3f' % kge_)
                                  #+', '.join(sim['ACTIVE_VAR_NAMES'])
                                  )
                    pl.axes().set_aspect('equal', )
                    pl.show()
                    
                    if task=='forecast' or task=='regression':
                        pl.figure(figsize=(12,5)); 
                        #s = y_test.argsort()
                        s = range(len(y_test))
                        pl.plot(sim['Y_TEST_TRUE'][s].ravel(), 'r-o', label='Real data',)
                        pl.plot(sim['Y_TEST_PRED'][s].ravel(), 'b-o', label='Predicted',)
                        #r2=r2_score(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())
                        #r=stats.pearsonr(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())[0]
                        #rmse=RMSE(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())                
                        acc=accuracy_log(sim['Y_TEST_TRUE'].ravel(), sim['Y_TEST_PRED'].ravel())                
                        pl.title(dataset_name+' -- '+target+'\nRMSE = '+str(rmse)+', '+'R$^2$ = '+str(r2)+', '+'R = '+str(r)+'KGE = '+str(kge_))
                        pl.ylabel(dataset_name)
                        pl.title(sim['EST_NAME']+': (Testing) R$^2$='+str('%1.3f' % r2)+'\t RMSE='+str('%1.3f' % rmse)
                                    +'\t MAPE ='+str('%1.3f' % mape)
                                    +'\t R ='+str('%1.3f' % r)
                                    +'\t NSE ='+str('%1.3f' % nse_)
                                    +'\t KGE ='+str('%1.3f' % kge_)
                                  #+', '.join(sim['ACTIVE_VAR_NAMES'])
                                  )
                        pl.show()                                                        
                    
                sim['RUN']=run;
                sim['DATASET_NAME']=dataset_name; 
                list_results.append(sim) 
        
                data    = pd.DataFrame(list_results)
                ds_name = dataset_name.replace('/','_').replace("'","").lower()
                tg_name = target.replace('/','_').replace("'","").lower()
                algo    = sim['ALGO'].split(':')[0] 
                pk=(path+#'_'+
                    basename+'_'+
                    '_run_'+str("{:02d}".format(run))+'_'+
                    ("%15s"%ds_name         ).rjust(15).replace(' ','_')+#'_'+
                    ("%9s"%sim['EST_NAME']  ).rjust( 9).replace(' ','_')+#'_'+
                    ("%10s"%algo            ).rjust(10).replace(' ','_')+#'_'+
                    ("%15s"%tg_name         ).rjust(25).replace(' ','_')+#'_'+
                    #("%15s"%os.uname()[1]   ).rjust(25).replace(' ','_')+#'_'+
                    #time.strftime("%Y_%m_%d_") + time.strftime("_%Hh_%Mm_%S")+
                    '.pkl') 
                pk=pk.replace(' ','_').replace("'","").lower()
                pk=pk.replace('(','_').replace(")","_").lower()
                pk=pk.replace('[','_').replace("]","_").lower()
                pk=pk.replace('-','_').replace("_","_").lower()
                #print(pk)
                data.to_pickle(pk)
                
##%%----------------------------------------------------------------------------