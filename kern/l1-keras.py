#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 06:10:36 2017

@author: dashiell
"""

import numpy as np
np.random.seed()
import pandas as pd
from datetime import datetime
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras.wrappers.scikit_learn import KerasClassifier

class roc_auc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_proba(self.x, verbose=0)
        roc = roc_auc_score(self.y, y_pred)
        logs['roc_auc'] = roc_auc_score(self.y, y_pred)
        logs['norm_gini'] = ( roc_auc_score(self.y, y_pred) * 2 ) - 1

        y_pred_val = self.model.predict_proba(self.x_val, verbose=0)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        logs['roc_auc_val'] = roc_auc_score(self.y_val, y_pred_val)
        logs['norm_gini_val'] = ( roc_auc_score(self.y_val, y_pred_val) * 2 ) - 1

        print('\rroc_auc: %s - roc_auc_val: %s - norm_gini: %s - norm_gini_val: %s' % (str(round(roc,5)),str(round(roc_val,5)),str(round((roc*2-1),5)),str(round((roc_val*2-1),5))), end=10*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod(
            (datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' %
              (thour, tmin, round(tsec, 2)))
    
    
def scale_data(X, scaler=None):
    if not scaler:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

DATA_TRAIN_PATH = '../input/train.csv'
DATA_TEST_PATH = '../input/test.csv'

def load_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH):
    train_loader = pd.read_csv(path_train, dtype={'target': np.int8, 'id': np.int32})
    train = train_loader.drop(['target', 'id'], axis=1)
    train_labels = train_loader['target'].values
    train_ids = train_loader['id'].values
    print('\n Shape of raw train data:', train.shape)

    test_loader = pd.read_csv(path_test, dtype={'id': np.int32})
    test = test_loader.drop(['id'], axis=1)
    test_ids = test_loader['id'].values
    print(' Shape of raw test data:', test.shape)

    return train, train_labels, test, train_ids, test_ids

folds = 4
runs = 2

cv_LL = 0
cv_AUC = 0
cv_gini = 0
fpred = []
avpred = []
avreal = []
avids = []

# Load data set and target values
train, target, test, tr_ids, te_ids = load_data()
n_train = train.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True)
col_to_drop = train.columns[train.columns.str.endswith('_cat')]
col_to_dummify = train.columns[train.columns.str.endswith('_cat')].astype(str).tolist()

for col in col_to_dummify:
    dummy = pd.get_dummies(train_test[col].astype('category'))
    columns = dummy.columns.astype(str).tolist()
    columns = [col + '_' + w for w in columns]
    dummy.columns = columns
    train_test = pd.concat((train_test, dummy), axis=1)

train_test.drop(col_to_dummify, axis=1, inplace=True)
train_test_scaled, scaler = scale_data(train_test)
train = train_test_scaled[:n_train, :]
test = train_test_scaled[n_train:, :]
print('\n Shape of processed train data:', train.shape)
print(' Shape of processed test data:', test.shape)

patience = 10
batchsize = 128

skf = StratifiedKFold(n_splits=folds, random_state=1001)
starttime = timer(None)
for i, (train_index, test_index) in enumerate(skf.split(train, target)):
    start_time = timer(None)
    X_train, X_val = train[train_index], train[test_index]
    y_train, y_val = target[train_index], target[test_index]
    train_ids, val_ids = tr_ids[train_index], tr_ids[test_index]
    
    def baseline_model():
        model = Sequential()
        model.add(
            Dense(
                200,
                input_dim=X_train.shape[1],
                kernel_initializer='glorot_normal',
                ))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(100, kernel_initializer='glorot_normal'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Dense(50, kernel_initializer='glorot_normal'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.15))
        model.add(Dense(25, kernel_initializer='glorot_normal'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        model.compile(optimizer='adam', metrics = ['accuracy'], loss='binary_crossentropy')

        return model


    for run in range(runs):
        print('\n Fold %d - Run %d\n' % ((i + 1), (run + 1)))
        np.random.seed()

        callbacks = [
            roc_auc_callback(training_data=(X_train, y_train),validation_data=(X_val, y_val)),  # call this before EarlyStopping
            EarlyStopping(monitor='norm_gini_val', patience=patience, mode='max', verbose=1),
            CSVLogger('keras-5fold-run-01-v1-epochs.log', separator=',', append=False),
            ModelCheckpoint(
                    'keras-5fold-run-01-v1-fold-' + str('%02d' % (i + 1)) + '-run-' + str('%02d' % (run + 1)) + '.check',
                    monitor='norm_gini_val', mode='max', # mode must be set to max or Keras will be confused
                    save_best_only=True,
                    verbose=1)
        ]

        nnet = KerasClassifier(
            build_fn=baseline_model,
            epochs=3,
            batch_size=batchsize,
            validation_data=(X_val, y_val),
            verbose=2,
            shuffle=True,
            callbacks=callbacks)

        fit = nnet.fit(X_train, y_train)
        

        del nnet
        nnet = load_model('keras-5fold-run-01-v1-fold-' + str('%02d' % (i + 1)) + '-run-' + str('%02d' % (run + 1)) + '.check')
        scores_val_run = nnet.predict_proba(X_val, verbose=0)
        LL_run = log_loss(y_val, scores_val_run)
        print('\n Fold %d Run %d Log-loss: %.5f' % ((i + 1), (run + 1), LL_run))
        AUC_run = roc_auc_score(y_val, scores_val_run)
        print(' Fold %d Run %d AUC: %.5f' % ((i + 1), (run + 1), AUC_run))
        print(' Fold %d Run %d normalized gini: %.5f' % ((i + 1), (run + 1), AUC_run*2-1))
        y_pred_run = nnet.predict_proba(test, verbose=0)
        if run > 0:
            scores_val = scores_val + scores_val_run
            y_pred = y_pred + y_pred_run
        else:
            scores_val = scores_val_run
            y_pred = y_pred_run
            
    scores_val = scores_val / runs
    y_pred = y_pred / runs
    LL = log_loss(y_val, scores_val)
    print('\n Fold %d Log-loss: %.5f' % ((i + 1), LL))
    AUC = roc_auc_score(y_val, scores_val)
    print(' Fold %d AUC: %.5f' % ((i + 1), AUC))
    print(' Fold %d normalized gini: %.5f' % ((i + 1), AUC*2-1))
    timer(start_time)

    if i > 0:
        fpred = pred + y_pred
        avreal = np.concatenate((avreal, y_val), axis=0)
        avpred = np.concatenate((avpred, scores_val), axis=0)
        avids = np.concatenate((avids, val_ids), axis=0)
    else:
        fpred = y_pred
        avreal = y_val
        avpred = scores_val
        avids = val_ids
    pred = fpred
    cv_LL = cv_LL + LL
    cv_AUC = cv_AUC + AUC
    cv_gini = cv_gini + (AUC*2-1)
    
LL_oof = log_loss(avreal, avpred)
print('\n Average Log-loss: %.5f' % (cv_LL/folds))
print(' Out-of-fold Log-loss: %.5f' % LL_oof)
AUC_oof = roc_auc_score(avreal, avpred)
print('\n Average AUC: %.5f' % (cv_AUC/folds))
print(' Out-of-fold AUC: %.5f' % AUC_oof)
print('\n Average normalized gini: %.5f' % (cv_gini/folds))
print(' Out-of-fold normalized gini: %.5f' % (AUC_oof*2-1))
score = str(round((AUC_oof*2-1), 5))
timer(starttime)
mpred = pred / folds

print('#\n Writing results')
now = datetime.now()
oof_result = pd.DataFrame(avreal, columns=['target'])
oof_result['prediction'] = avpred
oof_result['id'] = avids
oof_result.sort_values('id', ascending=True, inplace=True)
oof_result = oof_result.set_index('id')
sub_file = 'train_5fold-keras-run-01-v1-oof_' + str(score) + '_' + str(now.strftime('%Y-%m-%d-%H-%M')) + '.csv'
print('\n Writing out-of-fold file:  %s' % sub_file)
oof_result.to_csv(sub_file, index=True, index_label='id')

result = pd.DataFrame(mpred, columns=['target'])
result['id'] = te_ids
result = result.set_index('id')
print('\n First 10 lines of your 5-fold average prediction:\n')
print(result.head(10))
sub_file = 'submission_5fold-average-keras-run-01-v1_' + str(score) + '_' + str(now.strftime('%Y-%m-%d-%H-%M')) + '.csv'
print('\n Writing submission:  %s' % sub_file)
result.to_csv(sub_file, index=True, index_label='id')

