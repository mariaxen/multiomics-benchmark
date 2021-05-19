#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:03:50 2020

@author: maria
"""

import pandas as pd 
import numpy as np
from rpy2.robjects import r, pandas2ri
from sklearn.preprocessing import StandardScaler
import rpy2.robjects as ro
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GroupKFold
import groupyr
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, ElasticNetCV, MultiTaskLassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import nnls
from skopt import BayesSearchCV
from group_lasso import GroupLasso
from skopt.space import Real, Categorical, Integer

def get_feature_groups(DF, omic_names, predictor_index):
    
    DFC = DF.copy()
    #Remove the omic dataset you do not want to use 
    unwanted = DFC.columns[DFC.columns.str.startswith(omic_names[predictor_index])]
    DFC.drop(unwanted, axis=1, inplace=True)
    
    #Get the columns 
    DFC_columns = pd.DataFrame(DFC.columns.values)
    DFC_columns.columns = ['cols']

    for i in range(len(omic_names)):
        DFC_columns[omic_names[i]] = np.where(DFC_columns.cols.str.startswith(omic_names[i]), i+1, 0)

    DFC_columns = DFC_columns.drop(['cols'], axis=1)
    DFC_columns['group_cols']= DFC_columns.sum(axis=1)

    return DFC_columns['group_cols']

def get_feature_list(X, omic_names, predictor_index):

    columns = np.asarray(X.columns.values)

    groups_c = []

    new_omic_names = omic_names.copy()
    del new_omic_names[predictor_index]

    for l in range(len(new_omic_names)):
        groups_c.append(sum(w.startswith(new_omic_names[l]) for w in columns))

    blocks_l = np.cumsum(groups_c)  

    block_list = []
    block = np.array(range(0, blocks_l[0]))

    block_list.append(block)

    for i in range(len(blocks_l)-1):
        block_list.append(np.array(range(blocks_l[i], blocks_l[i+1])))
        
    return block_list

def get_cumulative_group_counts(X, omic_names, predictor_index):

    groups_c = []
    
    columns = np.asarray(X.columns.values)

    new_omic_names = omic_names.copy()
    del new_omic_names[predictor_index]

    for l in range(len(new_omic_names)):
        groups_c.append(sum(w.startswith(new_omic_names[l]) for w in columns))
        
    blocks_l = np.cumsum(groups_c)    
        
    return blocks_l

def get_list_omics(X, omic_names):

    X_list = []
    
    for predictor_index in range(len(omic_names)):

        wanted = X.columns[X.columns.str.startswith(omic_names[predictor_index])]
        X_mod = X[wanted].values

        X_list.append(X_mod)

    return X_list

def get_X_y(DF, omic_names, predictor_index, responses):
    
    # Get your X
    DFC = DF.copy()
    unwanted = DFC.columns[DFC.columns.str.startswith(omic_names[predictor_index])]
    DFC.drop(unwanted, axis=1, inplace=True)
    X = DFC
    
    response = responses[predictor_index]
    response_df = DF[response]
    y = np.array(response_df, dtype='float')

    return X, y

def cv_splitting(X, y, groups, longitudinal, folds, repeats):
    
    if longitudinal == False:
        
        kfold = RepeatedKFold(n_splits=folds, n_repeats=repeats)
        splitting = kfold.split(X)
    
    elif longitudinal == True:
        
        group_kfold = GroupKFold(n_splits=folds)        
        splitting = group_kfold.split(X, y, groups)
        
    return splitting

def metalearner_results(type_meta, y_pred_test_intrain, y_obsr_test_intrain, test_data, tries, cv, type_model = 0, positive = False):

    if type_meta == 'regular':
        
        if type_model == 'lasso': 

            meta_model = LassoCV(n_jobs = 2, n_alphas = tries, cv = cv, verbose=0)
            
        elif type_model == 'EN': 

            meta_model = ElasticNetCV(n_jobs = 2, n_alphas = tries, cv = cv, verbose=0)
                        
        elif type_model == 'ridge': 
            
            meta_model = RidgeCV(alphas = np.arange(0,12,0.3), cv = cv)
            
        elif type_model == 'RF':
            
            meta_model = RandomForestRegressor()
        
        if positive == True:
            meta_model.positive = positive
        
        #Fit
        meta_model.fit(y_pred_test_intrain, y_obsr_test_intrain)
        #Predict
        y_pred = pd.DataFrame(meta_model.predict(test_data))
        
    elif type_meta == 'nnls':

        #Fit
        coeff = nnls(y_pred_test_intrain, y_obsr_test_intrain)[0]

        #Multiply with coefficients
        prediction_train = coeff*test_data
        #prediction
        y_pred = pd.DataFrame(prediction_train.sum(axis = 1))
        
    return y_pred

def Ranger(X, y, feat_n, patient_groups, longitudinal, folds, repeats, tries, cv):
    
    pandas2ri.activate()
    
    y = y[:, feat_n]
    
    model ="""
            function(X_trainr, X_testr, y_trainr, tries){
                        
                library('tuneRanger')
                
                X_trainr <- as.data.frame(X_trainr)
                X_testr <- as.data.frame(X_testr)
                
                X_trainr[, 'target'] <- y_trainr
                
                make_task <- makeRegrTask(data = X_trainr, target = 'target')
                
                model <- tuneRanger(make_task, iters = tries - 30, num.threads = 1)
                
                #Get results
                pred_train <- predict(model$model, newdata = X_trainr)$data$response
                pred_test <- predict(model$model, newdata = X_testr)$data$response
                
                results <- list("train" = pred_train, "test" = pred_test) 
                results
                
        }""" 
    
    # Get your splitting
    splitting = cv_splitting(X, y, patient_groups, longitudinal, folds, repeats)
        
    #############    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    #############    
        
    # Run the cross validation
    for train_index, test_index in splitting:
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
                
        # Fit a standard scaler on your training data
        scaler = StandardScaler()
        scaler.fit(np.array(X_train, dtype='float'))

        # Transform your training and test data
        X_train = pd.DataFrame(scaler.transform(np.array(X_train, dtype='float')), columns = X_train.columns)
        X_test = pd.DataFrame(scaler.transform(np.array(X_test, dtype='float')), columns = X_test.columns)

        #Change to R data
        nr, nc = X_train.shape
        X_trainr = ro.r.matrix(X_train.values, nrow=nr, ncol=nc, byrow=True)

        nr, nc = X_test.shape
        X_testr = ro.r.matrix(X_test.values, nrow=nr, ncol=nc, byrow=True)

        y_trainr = ro.FloatVector(y_train)
                                 
        rfunc=ro.r(model)
        
        y_pred = rfunc(X_trainr, X_testr, y_trainr, tries)            
                        
        # Save the results
        y_prediction_train.append(pd.DataFrame(y_pred[0]))
        y_observed_train.append(pd.DataFrame(y_train))

        y_prediction_test.append(pd.DataFrame(y_pred[1]))
        y_observed_test.append(pd.DataFrame(y_test))
    
    return pd.concat(y_prediction_train), pd.concat(y_observed_train), pd.concat(y_prediction_test),pd.concat(y_observed_test)

def blockRF(X, y, feat_n, feature_groups_list, patient_groups, longitudinal, folds, repeats, tries, cv):
    
    pandas2ri.activate()
    
    y = y[:, feat_n]

    model ="""
            function(X_trainr, X_testr, y_trainr, blocksr, tries){
                library('blockForest')

                colnames(X_trainr) <- paste("X", 1:ncol(X_trainr), sep="")
                colnames(X_testr)  <- paste("X", 1:ncol(X_testr ), sep="")

                forest_obj <- blockfor(X_trainr, y_trainr, blocks=blocksr, nsets = tries, num.threads = 5)

                #Get results
                pred_train <- predict(forest_obj$forest, X_trainr)$predictions
                pred_test <- predict(forest_obj$forest, X_testr)$predictions

                results <- list("train" = pred_train, "test" = pred_test) 
                results

            }"""

    # Get your feature groups
    blocksr = ro.ListVector([(str(i), x) for i, x in enumerate(feature_groups_list)])

    # Get your splitting
    splitting = cv_splitting(X, y, patient_groups, longitudinal, folds, repeats)

    #############    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    #############    

    # Run the cross validation
    for train_index, test_index in splitting:

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        # Fit a standard scaler on your training data
        scaler = StandardScaler()
        scaler.fit(np.array(X_train, dtype='float'))

        # Transform your training and test data
        X_train = pd.DataFrame(scaler.transform(np.array(X_train, dtype='float')), columns = X_train.columns)
        X_test = pd.DataFrame(scaler.transform(np.array(X_test, dtype='float')), columns = X_test.columns)

        #Change to R data
        nr, nc = X_train.shape
        X_trainr = ro.r.matrix(X_train.values, nrow=nr, ncol=nc, byrow=True)

        nr, nc = X_test.shape
        X_testr = ro.r.matrix(X_test.values, nrow=nr, ncol=nc, byrow=True)

        y_trainr = ro.FloatVector(y_train)
            
        rfunc=ro.r(model)
            
        y_pred = rfunc(X_trainr, X_testr, y_trainr, blocksr, tries)            
                        
        # Save the results
        y_prediction_train.append(pd.DataFrame(y_pred[0]))
        y_observed_train.append(pd.DataFrame(y_train))

        y_prediction_test.append(pd.DataFrame(y_pred[1]))
        y_observed_test.append(pd.DataFrame(y_test))
    
    return pd.concat(y_prediction_train), pd.concat(y_observed_train), pd.concat(y_prediction_test),pd.concat(y_observed_test)

def pymodels(X, y, type_model, feat_n, patient_groups, longitudinal, folds, repeats, tries, cv):
        
    # Get your y
    y = y[:, feat_n]
        
    # Get your splitting
    splitting = cv_splitting(X, y, patient_groups, longitudinal, folds, repeats)
        
    #############    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    #############    
        
    # Run the cross validation
    for train_index, test_index in splitting:

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #if type_model == 'SGLasso': 
        
         #   model = groupyr.SGLCV(l1_ratio = 1, groups=feature_groups_list, fit_intercept=False, cv=cv, n_jobs=5, tuning_strategy='bayes', n_bayes_iter=tries, n_bayes_points=5)

        if type_model == 'lasso': 

            model = LassoCV(n_jobs = 5, n_alphas = tries, cv = cv, verbose=0)
            
        elif type_model == 'EN': 

            model = ElasticNetCV(n_jobs = 5, n_alphas = tries, cv = cv, verbose=0)
                        
        elif type_model == 'ridge': 
            
            model = RidgeCV(alphas = np.arange(0,12,0.3), cv = cv)
            
        elif type_model == 'RF':
            
            model = RandomForestRegressor()


        model_pipeline = Pipeline([
          ('scaler', StandardScaler()),
          ('regression', model)
                        ])
        
        model_pipeline.fit(np.array(X_train.values, dtype='float'), y_train)

        # Make prediction
        y_pred_train = model_pipeline.predict(np.array(X_train.values, dtype='float'))
        y_pred_test = model_pipeline.predict(np.array(X_test.values, dtype='float'))
                                
        # Save the results
        y_prediction_train.append(pd.DataFrame(y_pred_train))
        y_observed_train.append(pd.DataFrame(y_train))
        y_prediction_test.append(pd.DataFrame(y_pred_test))
        y_observed_test.append(pd.DataFrame(y_test))
    
    return pd.concat(y_prediction_train), pd.concat(y_observed_train), pd.concat(y_prediction_test),pd.concat(y_observed_test)

def pcLasso(X, y, blocks_l, feat_n, patient_groups, longitudinal, folds, repeats, tries):

    pandas2ri.activate()
    
    y = y[:, feat_n]
    
    # Get your splitting
    splitting = cv_splitting(X, y, patient_groups, longitudinal, folds, repeats)

    #############    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    #############    

    # Run the cross validation
    for train_index, test_index in splitting:

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        block_list = []
        block = np.array(range(1, blocks_l[0]+1))

        #pcLasso
        if len(block)>35000:
            block1 = np.array(range(1, 25000))
            block2 = np.array(range(25000, blocks_l[0]+1))
            block_list.append(block1)
            block_list.append(block2)

        elif len(block)<35000: 
            block_list.append(block)

        for i in range(len(blocks_l)-1):
            block_list.append(np.array(range(blocks_l[i]+1, blocks_l[i+1]+1)))

        blocksr = ro.ListVector([(str(i), x) for i, x in enumerate(block_list)])

        ratio = 0.7                 
        
        # Fit a standard scaler on your training data
        scaler = StandardScaler()
        scaler.fit(np.array(X_train, dtype='float'))

        # Transform your training and test data
        X_train = pd.DataFrame(scaler.transform(np.array(X_train, dtype='float')), columns = X_train.columns)
        X_test = pd.DataFrame(scaler.transform(np.array(X_test, dtype='float')), columns = X_test.columns)

        #Change to R data
        nr, nc = X_train.shape
        X_trainr = ro.r.matrix(X_train.values, nrow=nr, ncol=nc, byrow=True)

        nr, nc = X_test.shape
        X_testr = ro.r.matrix(X_test.values, nrow=nr, ncol=nc, byrow=True)

        y_trainr = ro.FloatVector(y_train)
        
        model ="""
            function(X_trainr, X_testr, y_trainr, blocksr, ratio, tries){
                library('pcLasso')         

                model <- pcLasso(X_trainr, y_trainr, ratio = ratio, 
                         groups = blocksr, family = "gaussian", 
                         nlam = tries, standardize = FALSE)

                pred_train <- predict(model, X_trainr)#, s = "lambda.min")
                pred_test <- predict(model, X_testr)#, s = "lambda.min")

                results <- list("train" = pred_train, "test" = pred_test) 

                results
        }"""            

        rfunc=ro.r(model)
        
        y_pred = rfunc(X_trainr, X_testr, y_trainr, blocksr, ratio, tries)  
        
        # Save the results
        y_prediction_train.append(pd.DataFrame(y_pred[0]))
        y_observed_train.append(pd.DataFrame(y_train))

        y_prediction_test.append(pd.DataFrame(y_pred[1]))
        y_observed_test.append(pd.DataFrame(y_test))
        
    return pd.concat(y_prediction_train), pd.concat(y_observed_train), pd.concat(y_prediction_test),pd.concat(y_observed_test)

def inf_Lasso(X, y, block_list, type_model, feat_n, patient_groups, longitudinal, folds, repeats, tries, cv):

    pandas2ri.activate()
    
    y = y[:, feat_n]
    
    # Get your splitting
    splitting = cv_splitting(X, y, patient_groups, longitudinal, folds, repeats)

    #############    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    #############    

    # Run the cross validation
    for train_index, test_index in splitting:

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        blocksr = ro.ListVector([(str(i), x) for i, x in enumerate(block_list)])
                
        # Fit a standard scaler on your training data
        scaler = StandardScaler()
        scaler.fit(np.array(X_train, dtype='float'))

        # Transform your training and test data
        X_train = pd.DataFrame(scaler.transform(np.array(X_train, dtype='float')), columns = X_train.columns)
        X_test = pd.DataFrame(scaler.transform(np.array(X_test, dtype='float')), columns = X_test.columns)

        #Change to R data
        nr, nc = X_train.shape
        X_trainr = ro.r.matrix(X_train.values, nrow=nr, ncol=nc, byrow=True)

        nr, nc = X_test.shape
        X_testr = ro.r.matrix(X_test.values, nrow=nr, ncol=nc, byrow=True)

        y_trainr = ro.FloatVector(y_train)
        
        if type_model == 'ipf_lasso':
        
            model ="""
            function(X_trainr, X_testr, y_trainr, Br, cv){
                library('ipflasso')

                model <- cvr.adaptive.ipflasso(X=X_trainr,Y=y_trainr,family="gaussian", type.measure = 'mse',
                standardize=FALSE, alpha = 0, type.step1 = 'sep',
                blocks=list(block1=1:Br[1], block2=(Br[1]+1):Br[2], block3=(Br[2]+1):Br[3], 
                block4=(Br[3]+1):Br[4], block5=(Br[4]+1):Br[5], block6=(Br[5]+1):Br[6]),
                nfolds = cv, ncv=1)

                #Get results
                pred_train <- ipflasso.predict(object=model, Xtest=X_trainr)$linpredtest
                pred_test <- ipflasso.predict(object=model, Xtest=X_testr)$linpredtest

                results <- list("train" = pred_train, "test" = pred_test) 
                results

            }"""
            
        elif type_model == 'priority_lasso':
            
            model ="""
            function(X_trainr, X_testr, y_trainr, Br, cv){
                library('ipflasso')

                model <- prioritylasso(X=X_trainr,Y=y_trainr,family="gaussian", type.measure = 'mse',
                                    standardize=FALSE, lambda.type = 'lambda.min', blocks=list(block1=1:Br[1], 
                                    block2=(Br[1]+1):Br[2], block3=(Br[2]+1):Br[3], block4=(Br[3]+1):Br[4], 
                                    block5=(Br[4]+1):Br[5], block6=(Br[5]+1):Br[6]), nfolds=cv, 
                                    cvoffset = TRUE, cvoffsetnfolds = cv)

                #Get results
                pred_train <- predict(object=model, newdata=X_trainr, type = 'response')
                pred_test <- predict(object=model, newdata=X_testr, type = 'response')

                results <- list("train" = pred_train, "test" = pred_test) 
                results

            }"""            
        
        rfunc=ro.r(model)
        
        y_pred = rfunc(X_trainr, X_testr, y_trainr, blocksr, cv)  
        
        # Save the results
        y_prediction_train.append(pd.DataFrame(y_pred[0]))
        y_observed_train.append(pd.DataFrame(y_train))

        y_prediction_test.append(pd.DataFrame(y_pred[1]))
        y_observed_test.append(pd.DataFrame(y_test))
        
    return pd.concat(y_prediction_train), pd.concat(y_observed_train), pd.concat(y_prediction_test),pd.concat(y_observed_test)


def MTLasso(X, y, patient_groups, longitudinal, folds, repeats, tries, cv):
                
    # Get your splitting
    splitting = cv_splitting(X, y, patient_groups, longitudinal, folds, repeats)
        
    #############    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    #############    
        
    # Run the cross validation
    for train_index, test_index in splitting:

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = MultiTaskLassoCV(n_jobs = 5, n_alphas = tries, cv = cv)

        model_pipeline = Pipeline([
          ('scaler', StandardScaler()),
          ('regression', model)
                        ])
        
        model_pipeline.fit(np.array(X_train.values, dtype='float'), y_train)

        # Make prediction
        y_pred_train = model_pipeline.predict(np.array(X_train.values, dtype='float'))
        y_pred_test = model_pipeline.predict(np.array(X_test.values, dtype='float'))
                                
        # Save the results
        y_prediction_train.append(pd.DataFrame(y_pred_train))
        y_observed_train.append(pd.DataFrame(y_train))
        y_prediction_test.append(pd.DataFrame(y_pred_test))
        y_observed_test.append(pd.DataFrame(y_test))
    
    return pd.concat(y_prediction_train), pd.concat(y_observed_train), pd.concat(y_prediction_test),pd.concat(y_observed_test)


def Stacked_Lasso(X, y, omic_names, feat_n, responses, patient_groups, longitudinal, folds, repeats, tries, cv):
        
    # Get your y
    y = y[:, feat_n]
    
    # Get your splitting
    splitting = cv_splitting(X, y, patient_groups, longitudinal, folds, repeats)

    #############    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    #############    

    y_prediction_train_meta_all = {} 
    y_prediction_test_meta_all = {}

    metalearner_list = list()
    metalearner_list.append(['regular', 'lasso', False])
    metalearner_list.append(['regular', 'lasso', True])
    metalearner_list.append(['regular', 'EN', False])
    metalearner_list.append(['regular', 'EN', True])
    metalearner_list.append(['regular', 'RF', False])
    metalearner_list.append(['regular', 'ridge', False])
    metalearner_list.append(['nnls', 'nnls', False])

    for i in range(len(metalearner_list)):
        y_prediction_train_meta_all[metalearner_list[i][0]+'_'+metalearner_list[i][1]+'_'+str(metalearner_list[i][2])] = []
        y_prediction_test_meta_all[metalearner_list[i][0]+'_'+metalearner_list[i][1]+'_'+str(metalearner_list[i][2])] = []

    # Run the cross validation
    for train_index, test_index in splitting:        

        y_pred_train_omic = []
        y_pred_test_omic = []

        # Train and test sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        patient_groups = pd.Series(patient_groups)
        patient_groups_train = patient_groups.iloc[train_index]

        #Get your model
        model = LassoCV(n_jobs = 1, n_alphas = tries, cv = cv, verbose=0)
        
        model_pipeline = Pipeline([
          ('scaler', StandardScaler()),
          ('regression', model)
                        ])
        
        # Get a prediction for each omic
        for omic_index in range(len(omic_names)): 

            #Get the names of the columns you want
            wanted = X_train.columns[X_train.columns.str.startswith(omic_names[omic_index])]

            # Subset the X with the selected columns
            X_train_omic = X_train[wanted]
            X_test_omic = X_test[wanted]

            # Change them to float format
            X_train_omic = np.array(X_train_omic.values, dtype='float')
            X_test_omic = np.array(X_test_omic.values, dtype='float')    

            # Train the model for each omic
            model_pipeline.fit(X_train_omic, y_train)

            # Make a prediction for each omic, for the train and test sets
            y_pred_train = model_pipeline.predict(X_train_omic)
            y_pred_test = model_pipeline.predict(X_test_omic)

            # Save the prediction for each omic in a list 
            y_pred_train_omic.append(y_pred_train)
            y_pred_test_omic.append(y_pred_test)

        #Now make a dataframe that contains the predictions from all omics
        #You will use this later to make the final predictions with the metalearner
        y_pred_tr = pd.DataFrame(y_pred_train_omic).transpose() 
        y_pred_tst = pd.DataFrame(y_pred_test_omic).transpose() 

        y_pred_test_allFolds_intrain = []
        y_obsr_test_allFolds_intrain = []

        # Now do a second cross validation. You will use this to train the metalearner

        #Create your splitting 
        splitting = cv_splitting(X_train, y_train, patient_groups_train, longitudinal, cv, repeats)

        #Loop through the folds
        for train_index_train, test_index_train in splitting:        

            y_pred_test_omic_intrain = []
            y_obsr_test_omic_intrain = []

            # Train and test sets
            y_train_intrain, y_test_intrain = y_train[train_index_train], y_train[test_index_train]
            X_train_intrain, X_test_intrain = X_train.iloc[train_index_train], X_train.iloc[test_index_train]

            #Loop through all omics to get a prediction for each one
            for predictor_index in range(len(omic_names)): 

                #Get the column names you want
                wanted_intrain = X_train_intrain.columns[X_train_intrain.columns.str.startswith(omic_names[predictor_index])]

                #Subset with the selected columns
                X_train_omic_intrain = X_train_intrain[wanted_intrain]
                X_test_omic_intrain = X_test_intrain[wanted_intrain]

                #Change data type
                X_train_omic_intrain = np.array(X_train_omic_intrain.values, dtype='float')
                X_test_omic_intrain = np.array(X_test_omic_intrain.values, dtype='float')    

                #Train the model
                model_pipeline.fit(X_train_omic_intrain, y_train_intrain)

                #Make a prediction for the test set
                y_pred_test_omic_intrain.append(model_pipeline.predict(X_test_omic_intrain))
                y_obsr_test_omic_intrain.append(y_test_intrain)

            #Put together all the predictions from the different modalities    
            y_pred_test_intrain = pd.DataFrame(y_pred_test_omic_intrain).transpose() 

            #Save the result for every fold
            y_pred_test_allFolds_intrain.append(y_pred_test_intrain)
            y_obsr_test_allFolds_intrain.append(y_test_intrain)

        #Put together all the predictions for the test sets to train the metalearner
        y_pred_test_intrain = pd.concat(y_pred_test_allFolds_intrain)
        y_obsr_test_intrain = np.concatenate(y_obsr_test_allFolds_intrain, axis=None)

        #Get the results from the metalearner
        #Try different metalearners
        #Loop through different metalearning strategies

        for k in range(len(metalearner_list)):

            y_pred_train = metalearner_results(metalearner_list[k][0], y_pred_test_intrain, y_obsr_test_intrain, y_pred_tr, tries, cv, 
                                               metalearner_list[k][1], metalearner_list[k][2])
            y_pred_test = metalearner_results(metalearner_list[k][0], y_pred_test_intrain, y_obsr_test_intrain, y_pred_tst, tries, cv, 
                                              metalearner_list[k][1], metalearner_list[k][2])

            #Save them in a list, for each fold
            y_prediction_train_meta_all[metalearner_list[k][0]+'_'+metalearner_list[k][1]+'_'+str(metalearner_list[k][2])].append(y_pred_train) 
            y_prediction_test_meta_all[metalearner_list[k][0]+'_'+metalearner_list[k][1]+'_'+str(metalearner_list[k][2])].append(y_pred_test) 

        #Save the observed ones in a list, for each fold
        y_observed_train.append(pd.DataFrame(y_train))
        y_observed_test.append(pd.DataFrame(y_test))

    #Return results for all folds
    return y_prediction_train_meta_all, pd.concat(y_observed_train), y_prediction_test_meta_all, pd.concat(y_observed_test)


def GFA(X, y, omic_names, patient_groups, longitudinal, folds, repeats, tries, cv):
                
    pandas2ri.activate()
    
    #Get your model
    model ="""
            function(X_trainr, X_testr, y_trainr, target_index, tries){
            
                library('CCAGFA')

                type_list = rep(1, length(X_trainr))
                type_list[[target_index + 1]] = 0

                opts <- getDefaultOpts()
                
                opts$verbose = 0

                K = 10
                
                model <- GFAexperiment(X_trainr, K, opts, Nrep=tries)
                
                pred_train <- GFApred(type_list, X_trainr, model)$Y[target_index + 1]
                pred_test <- GFApred(type_list, X_testr, model)$Y[target_index + 1]

                results <- list("train" = pred_train, "test" = pred_test) 

                results
        }"""     
    
    # Get your splitting
    splitting = cv_splitting(X, y, patient_groups, longitudinal, folds, repeats)
        
    #############    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    #############    
        
    # Run the cross validation
    for train_index, test_index in splitting:
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit a standard scaler on your training data
        scaler = StandardScaler()
        scaler.fit(np.array(X_train.values, dtype='float'))

        # Transform your training and test data
        X_train = pd.DataFrame(scaler.transform(np.array(X_train.values, dtype='float')), columns = X_train.columns)
        X_test = pd.DataFrame(scaler.transform(np.array(X_test.values, dtype='float')), columns = X_test.columns)
        
        # Change to list of datasets
        X_train, X_test = get_list_omics(X_train, omic_names), get_list_omics(X_test, omic_names)

        # Replace one dataset with the features you are predicting
        X_train.append(y_train)
        X_test.append(y_test)
        
        target_index = len(omic_names)

        nr, nc = y_train.shape
        y_trainr = ro.r.matrix(y_train, nrow=nr, ncol=nc, byrow=True)

        X_trainr = ro.ListVector([(str(i), x) for i, x in enumerate(X_train)])
        X_testr = ro.ListVector([(str(i), x) for i, x in enumerate(X_test)])
    
        rfunc=ro.r(model)
        
        y_pred = rfunc(X_trainr, X_testr, y_trainr, target_index, tries)  
        
        # Save the results
        y_prediction_train.append(pd.DataFrame(y_pred[0][0]))
        y_observed_train.append(pd.DataFrame(y_train))

        y_prediction_test.append(pd.DataFrame(y_pred[1][0]))
        y_observed_test.append(pd.DataFrame(y_test))
    
    return pd.concat(y_prediction_train), pd.concat(y_observed_train), pd.concat(y_prediction_test),pd.concat(y_observed_test)


def SGLasso(X, y, feat_n, feature_groups,  patient_groups, longitudinal, folds, repeats, tries, cv):
        
    # Get your y
    y = y[:, feat_n]
    y = y.reshape(-1, 1)
        
    # Get your splitting
    splitting = cv_splitting(X, y, patient_groups, longitudinal, folds, repeats)
        
    #############    
    y_prediction_train = []
    y_observed_train = []

    y_prediction_test = []
    y_observed_test = []
    #############    
        
    # Run the cross validation
    for train_index, test_index in splitting:

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = RandomForestRegressor()

        model_pipeline = Pipeline([
          ('scaler', StandardScaler()),
          ('regression', model)
                        ])
                
        model = BayesSearchCV(GroupLasso(groups=feature_groups, 
                                        warm_start = True,
                                        fit_intercept = False), 
                                  {'group_reg': Real(0.01, 0.5, prior='log-uniform'),
                                   'l1_reg': Real(0.01, 0.5, prior='log-uniform')
                                  }, 
                                  n_iter = tries, verbose=0, cv = cv,
                                  n_jobs = 1, n_points = 1, iid = False)
        
        model_pipeline.fit(np.array(X_train.values, dtype='float'), y_train)

        # Make prediction
        y_pred_train = model_pipeline.predict(np.array(X_train.values, dtype='float'))
        y_pred_test = model_pipeline.predict(np.array(X_test.values, dtype='float'))
                                
        # Save the results
        y_prediction_train.append(pd.DataFrame(y_pred_train))
        y_observed_train.append(pd.DataFrame(y_train))
        y_prediction_test.append(pd.DataFrame(y_pred_test))
        y_observed_test.append(pd.DataFrame(y_test))
    
    return pd.concat(y_prediction_train), pd.concat(y_observed_train), pd.concat(y_prediction_test),pd.concat(y_observed_test)



