#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 05:47:34 2017

@author: dashiell
"""
import pandas as pd
import numpy as np
from scipy.stats import skew, boxcox
from gplearn.genetic import SymbolicTransformer



train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#ss = pd.read_csv('../input/sample_submission.csv')       
#cols_drop = [col for col in X_train.columns if not col.startswith('ps_calc_')]
       
tt = pd.concat((train,test)).reset_index(drop=True)
    
### GP feature creation
    
numeric_feats = tt.dtypes[tt.dtypes == np.float64].index
numeric_feats = numeric_feats.drop('target')
function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv']

gp = SymbolicTransformer(generations=20, population_size=2000,
                         hall_of_fame=100, n_components=10,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=0, n_jobs=6)

gp.fit(train[numeric_feats], train['target'])

gp_feats = gp.transform(tt[numeric_feats])
tt = pd.concat([tt,pd.DataFrame(gp_feats)],axis=1)

    
### box cox transform
'''
#numeric_feats = tt.dtypes[tt.dtypes != 'object'].index 
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    
skewed_feats = skewed_feats[skewed_feats > 0.2]
skewed_feats = skewed_feats.index
for feat in skewed_feats:
    tt[feat] = tt[feat] +10
    (tt[feat], lam) = boxcox(tt[feat])
    
''' 

### Categorical to dummy    
cols_oh = train.columns[train.columns.str.endswith('_cat')]
         
for col in cols_oh.astype(str).tolist():
    oh = pd.get_dummies(tt[col],prefix=col)
    tt = pd.concat((tt,oh), axis=1)
        
    
tt.drop(cols_oh, axis=1, inplace=True)
tt.drop('id', axis=1, inplace=True)
    
### recreate train, test
    
train = tt.iloc[:len(train),:]
test = tt.iloc[len(train):,:]

    ### return X_train, y_train, X_test
    #y_train = train['target']
    #X_train = train.drop(['target'], axis=1)
    #X_test = test.drop(['target'], axis=1)
    
train.drop(['target'],axis=1).to_pickle('../input/X_train.pkl', compression='bz2')
train['target'].to_pickle('../input/y_train.pkl', compression='bz2')
test.drop(['target'],axis=1).to_pickle('../input/X_test.pkl', compression='bz2')
    
    
   

#(X_train, y_train, X_test) = train_test()

