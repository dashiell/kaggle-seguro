#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 03:36:10 2017

@author: dashiell

colsample_bytree=0.5, learning_rate=0.02, max_depth=5, min_child_weight=0.5, reg_alpha=0, 
reg_lambda=0.5, scale_pos_weight=1, subsample=0.7,
"""

import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.metrics import roc_auc_score
#from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

X_train = pd.read_pickle('../input/X_train.pkl', compression='bz2')
y_train = pd.read_pickle('../input/y_train.pkl', compression='bz2')
X_test = pd.read_pickle('../input/X_test.pkl', compression='bz2')

#(X_train, y_train, X_test) = load.train_test()

ker_X_test = pd.read_csv('../output/out-keras.csv')
ker_X_train = pd.read_csv('keras-X_train.csv')

X_train['ker'] = ker_X_train['prediction']
X_test['ker'] = ker_X_test['target']


#X_train = pd.DataFrame(np.load('../input/lh_train.npy'))
#X_test = pd.DataFrame(np.load('../input/lh_test.npy'))

def calc_gini(y_pred, dtrain):   
    y_true = dtrain.get_label()
    gini = (roc_auc_score(y_true, y_pred) * 2) - 1
    return [('gini', -gini)] # negative to minimize CV error

### model ###

k_folds = 5

skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=2)

oof_train_pred = np.zeros(len(y_train),)
full_test_pred = np.zeros( (len(X_test), k_folds))
#binary:logistic
model = XGBClassifier(n_estimators=1500, max_depth=4,objective="binary:logistic",  learning_rate=0.02, 
                      subsample=.8, min_child_weight=.8, colsample_bytree=.8, scale_pos_weight=1, gamma=5, 
                      reg_alpha=2.3, reg_lambda=1.3, tree_method='gpu_hist')

#model = XGBRegressor(n_estimators=1500, max_depth=4,objective="reg:linear",  learning_rate=0.02, 
#                      subsample=.8, min_child_weight=.8, colsample_bytree=.8, scale_pos_weight=1, gamma=5, 
#                      reg_alpha=2.3, reg_lambda=1.3, tree_method='gpu_hist')


for fold_ix, (tr_ix, te_ix) in enumerate(skf.split(X_train, y_train)):
    print(" training fold", fold_ix)

    xgb_fitted = model.fit(X = X_train.iloc[tr_ix], y = y_train.iloc[tr_ix], 
                          eval_set = [(X_train.iloc[te_ix], y_train.iloc[te_ix])],
                          eval_metric = calc_gini, 
                          early_stopping_rounds = 100, 
                          verbose = 100)
    
    # predict oof part of train set
    oof_train_pred[te_ix] = xgb_fitted.predict_proba(X_train.iloc[te_ix])[:,1] 
    
    full_test_pred[:,fold_ix] = xgb_fitted.predict_proba(X_test)[:,1]
    print( "  Best N trees = ", xgb_fitted.best_ntree_limit )
    print( "  Best gini = ", xgb_fitted.best_score )
    
y_pred = np.mean(full_test_pred, axis=1)    
print("gini for full training set", (roc_auc_score(y_train, oof_train_pred) *2)-1 )

ss = pd.read_csv('../input/sample_submission.csv')
ss['target'] = y_pred

ss.to_csv('../output/out-xgb.csv.gz',index=False, compression='gzip')