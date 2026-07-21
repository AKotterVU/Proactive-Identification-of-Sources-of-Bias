# benchmark.py
"""Trains and tests a panel of models, then reports which model 
architecture performed best.
Uses Optuna with stratified k-fold for hyperparameter optimization.
Command-line arguments:
    1. n (int): how many bootstrap samples to use for model performance
    2. k (int): how many splits to use for k-fold cross-validation
    3. n_trials (int): how many Optuna trials to run per model
10-18-24: Updated to drop Support Vector Machine and Gaussian Process 
models due to efficiency constraints on large datasets.
10-21-24: Updated to use F1-score instead of AUROC.
11-15-24: Updated to do cross-validation for hyperparameter 
optimization and bootstrapping for model evaluation.
(Two updates in same version)
11-18-24: Added univariate and multivariate feature selection to cut 
down on computational complexity. Pre-pruned some features known to be 
highly correlated with each other to allow multivarate feature 
selection to do its job without pruning too many features.
11-19-24: Added final statistical analysis to determine if differences 
between models are statistically significant.
11-22-24: Changed metric to average precision (AUPRC but less 
optimistic) instead of F1-score and added final testing set.
https://sanchom.wordpress.com/tag/average-precision/
https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/
12-4-24: Updated to include the correct proportion of positive sepsis 
cases in training for the final one-holdout test.
"""

# Import statements

import sys
import time
import optuna
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
# No naive Bayes because only for discrete features

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from sklearn.model_selection import KFold

from functions import bootstrap
from functions import objective
from functions import test_split
from functions import write_print
from functions import new_classifier
from functions import reproportion_pos
from functions import feature_select_univar
from functions import feature_select_multivar

from scipy.stats import friedmanchisquare as friedman
from scikit_posthocs import posthoc_nemenyi_friedman as nemenyi


# Utility function for running the benchmarking loop


# Run the file


if __name__ == "__main__":
    
    # Command-line arguments
    n = int(sys.argv[1])
    k = int(sys.argv[2])
    n_trials = int(sys.argv[3])
    
    # Hard-coded values
    # Tracking?
    debugging=True
    # Make sure CSN IDs are either in training or testing, not both?
    id_split=True
    # Which prediction window to use?
    out_col = "end_point_6"
    # IDs are kept in 'CSN' or somewhere else?
    id_col = "CSN"
    # Proportion of samples in training and validation that are positive
    train_pos_prop = 0.5
    # Proportion of samples in testing that are positive
    test_pos_prop = 0.05
    # Proportion of positive samples assigned to training instead of testing
    train_split_prop = 0.7
    # Labels to be counted as positive values
    pos_vals = ["SEPSIS"]
    # Positive training set equal in size to the original positive set?
    original_size = True
    # What metric should be used?
    score_metric = lambda x,y: average_precision_score(x,y,average=None)
    
    # Outfile for tracking
    date_str = "{}_{}_{}".format(*time.localtime()[0:3])
    outfile = f"objective_track_BASE_{date_str}.txt"
    write_print(f"Time: {time.ctime()}\n",outfile)
    
    # Get the data
    # X
    X = pd.read_csv("icu_X.csv", index_col="Unnamed: 0")
    # Tracking #####
    if debugging:
        write_print("Loaded X",outfile)
        write_print(f"Time: {time.ctime()}\n",outfile)
    # Drop the highly correlated features to allow better multivarate selection
    corr_lab_cols = ["gcs_total_avg", "gcs_total_min", "gcs_total_max", 
                     "gcs_total_stdevp", "wbc_avg", "wbc_min", "wbc_max", 
                     "wbc_stdevp", "bun_avg", "bun_min", "bun_max", 
                     "bun_stdevp", "platelet_avg", "platelet_min", 
                     "platelet_max", "platelet_stdevp", "gluc_avg", 
                     "gluc_min", "gluc_max", "gluc_stdevp", "proth_avg", 
                     "proth_min", "proth_max", "proth_stdevp", "hemo_avg", 
                     "hemo_min", "hemo_max", "hemo_stdevp", "hemat_avg", 
                     "hemat_min", "hemat_max", "hemat_stdevp", "bili_avg", 
                     "bili_min", "bili_max", "bili_stdevp", "pao2_avg", 
                     "pao2_min", "pao2_max", "pao2_stdevp", "paco2_avg", 
                     "paco2_min", "paco2_max", "paco2_stdevp", "fio2_avg", 
                     "fio2_min", "fio2_max", "fio2_stdevp", "cre_avg", 
                     "cre_min", "cre_max", "cre_stdevp", "lact_avg", 
                     "lact_min", "lact_max", "lact_stdevp"]
    corr_vit_cols = ["BP_DIAST_avg", "BP_DIAST_min", "BP_DIAST_max", 
                     "BP_DIAST_stdevp", "BP_SYST_avg", "BP_SYST_min", 
                     "BP_SYST_max", "BP_SYST_stdevp", "PULSE_avg", 
                     "PULSE_min", "PULSE_max", "PULSE_stdevp", 
                     "RESP_RATE_avg", "RESP_RATE_min", "RESP_RATE_max", 
                     "RESP_RATE_stdevp", "SPO2_avg", "SPO2_min", "SPO2_max", 
                     "SPO2_stdevp", "TEMP_C_avg", "TEMP_C_min", "TEMP_C_max", 
                     "TEMP_C_stdevp"]
    X.drop(columns=corr_lab_cols+corr_vit_cols, inplace=True)
    # y
    y = pd.read_csv("icu_y.csv", index_col="Unnamed: 0")
    # Tracking #####
    if debugging:
        write_print("Loaded y",outfile)
        write_print(f"Time: {time.ctime()}\n",outfile)
    
    # Drop the columns that aren't model inputs or outcomes
    if id_split:
        X.drop(columns=["csn_adt_id", "hr_slot", "hour_slot_admit"], 
               inplace=True)
        y = y[[id_col, out_col]]
        y_out = y[out_col]
    else:
        X.drop(columns=[id_col, "csn_adt_id", "hr_slot", "hour_slot_admit"], 
               inplace=True)
        y = y[out_col]
        y_out = y[out_col]
    
    # Tracking #####
    # How many positive and negative?
    if debugging:
        write_print(f"Positive y values: {y_out.isin(['SEPSIS']).sum()}", 
                    outfile)
        write_print(f"Total y values: {len(y)}", outfile)
        write_print(f"Time: {time.ctime()}\n", outfile)
    
    # Create a one-holdout testing set
    X, y, X_test_final, y_test_final = test_split(X,y,id_col,out_col=out_col)
    y_out = y[out_col]
    # Get a binary 1s and 0s version of y
    y_bi = y.copy()
    y_bi[out_col].where(y_bi[out_col].isin(pos_vals), 0, inplace=True)
    y_bi[out_col].mask(y_bi[out_col].isin(pos_vals), 1, inplace=True)
    y_bi[out_col] = y_bi[out_col].astype("int")
    # Get a binary 1s and 0s version of y_test_final
    y_bi_final = y_test_final.copy()
    y_bi_final[out_col].where(y_bi_final[out_col].isin(pos_vals), 0, 
                              inplace=True)
    y_bi_final[out_col].mask(y_bi_final[out_col].isin(pos_vals), 1, 
                             inplace=True)
    y_bi_final[out_col] = y_bi_final[out_col].astype("int")
    
    # Tracking #####
    if debugging:
        write_print(f"One-holdout size: {len(y_test_final)}", outfile)
        write_print(f"One-holdout pos.: {y_test_final.isin(pos_vals).sum()}", 
        outfile)
        write_print(f"New positive y values: {y_out.isin(pos_vals).sum()}", 
                    outfile)
        write_print(f"New total y values: {len(y)}", outfile)
        write_print(f"Time: {time.ctime()}\n", outfile)
    
    # Create a list of model names
    model_names = ["RF", "GB", "KNN", "LR", "MLP"]
    
    # Track the scores through each bootstrap to get the average performance
    scores = np.zeros((n, len(model_names)))
    train_scores = np.zeros((n, len(model_names)))
    
    # Bootstrap to get model performance
    for i in range(n):
        if debugging:
            write_print(f"\n\nBootstrapping round {i+1}/{n}", outfile)
            write_print(f"Time: {time.ctime()}\n", outfile)
        # Get the bootstrap samples for each model
        X_train, y_train, X_test, y_test = bootstrap(X, y, train_pos_prop, 
                                                     test_pos_prop, 
                                                     train_split_prop, 
                                                     pos_vals, id_split, 
                                                     id_col, out_col, 
                                                     original_size)
        if id_split:
            y_train_out = y_train[out_col]
            y_test = y_test[out_col]
        # Tracking #####
        if debugging:
            write_print(f"Training samples: {len(y_train)}", outfile)
            write_print(f"Testing samples: {len(y_test)}", outfile)
            if id_split:
                write_print(f"Positive training samples: {len(y_train_out)}", 
                            outfile)
            else:
                write_print(f"Positive training samples: {len(y_train)}", 
                            outfile)
            write_print(f"Positive testing samples: {len(y_test)}", outfile)
            write_print(f"Time: {time.ctime()}\n", outfile)
        # Perform feature selection to reduce computational complexity
        if id_split:
            X_train = feature_select_univar(X_train, y_train_out, [id_col], 
                                            debugging, outfile)
        else:
            X_train = feature_select_univar(X_train, y_train, [""], debugging, 
                                            outfile)
        X_train = feature_select_multivar(X_train, [id_col], debugging, 
                                          outfile)
        X_test = X_test[X_train.columns]
        if debugging:
            write_print("\nSelected features:", outfile)
            write_print([col for col in X_test.columns], outfile)
            write_print(f"Total features: {len(X_test.columns)}", outfile)
        
        # Iterate through each model
        for j, name in enumerate(model_names):
            if debugging:
                write_print(f"\nStarting {name}",outfile)
                write_print(f"Time: {time.ctime()}\n",outfile)
            # Run the Optuna study for hyperparameter optimization
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial, X_train, y_train, 
                                                   name, k=k, 
                                                   metric=score_metric, 
                                                   predict_proba=True, 
                                                   id_split=id_split, 
                                                   id_col=id_col, 
                                                   out_col=out_col, 
                                                   debugging=debugging), 
                           n_trials=n_trials)
            # Get the best parameters for training an "official" model
            best_params = study.best_trial.params
            if name == "MLP":
                n_layers = best_params['n_layers']
                n_neurons = best_params['n_neurons']
                best_params["hidden_layer_sizes"] = [n_neurons]*n_layers
                best_params.pop("n_layers")
                best_params.pop("n_neurons")
            elif name == "SVC":
                best_params["probability"] = True
            elif name == "LR":
                best_params["solver"] = "saga"
                best_params["penalty"] = "elasticnet"
            # Tracking #####
            if debugging:
                best_score = study.best_value
                write_print(f"Selected hyperparameters: {best_params}",outfile)
                write_print(f"Average score: {best_score}", outfile)
                write_print(f"Time: {time.ctime()}\n",outfile)
            # Train an "official" model using the selected hyperparameters
            model = new_classifier(name, best_params)
            # Tracking ##########
            if debugging:
                write_print(f"Instantiated \"official\" {name} model.",outfile)
                write_print(f"Time: {time.ctime()}\n",outfile)
            if id_split:
                model.fit(X_train, y_train_out)
            else:
                model.fit(X_train, y_train)
            # Tracking ##########
            if debugging:
                write_print(f"Fit \"official\" {name} model.",outfile)
                write_print(f"Time: {time.ctime()}\n",outfile)
            # Test the model
            prediction = model.predict_proba(X_test)[:,1]
            # Tracking ##########
            if debugging:
                write_print(f"Test predictions for {name} model.",outfile)
                write_print(f"Time: {time.ctime()}\n",outfile)
            train_prediction = model.predict_proba(X_train)[:,1]
            # Tracking ##########
            if debugging:
                write_print(f"Train predictions to check overfit.",outfile)
                write_print(f"Time: {time.ctime()}\n",outfile)
            # Save the score
            scores[i,j] = score_metric(y_test, prediction)
            if id_split:
                train_scores[i,j] = score_metric(y_train_out,train_prediction)
            else:
                train_scores[i,j] = score_metric(y_train, train_prediction)
            # Keep track of progress
            if debugging:
                write_print(f"Score: {scores[i,j]}",outfile)
                write_print(f"Train score: {train_scores[i,j]}",outfile)
                write_print(f"Time: {time.ctime()}\n",outfile)
        
        # Keep track of progress: average performance for the round
        if debugging:
            write_print(f"\nAll scores this round: \n{scores[i,:]}", outfile)
            write_print(f"\nTrain scores this round: \n{train_scores[i,:]}", 
                        outfile)
            write_print(f"Time: {time.ctime()}\n\n\n",outfile)
    
    # Keep track of progress: total performance
    if debugging:
        write_print(f"\n\nAll scores: \n{scores}",outfile)
        write_print(f"\nAll train scores: \n{train_scores}",outfile)
        write_print(f"Time: {time.ctime()}\n\n",outfile)
    
    # Average performance
    for j, name in enumerate(model_names):
        write_print(f"Average score for {name}: {np.mean(scores[:,j])}", 
                    outfile)
    write_print(f"Time: {time.ctime()}\n", outfile)
    
    # One-holdout test performance
    # Get the positive proportions right for the new training set
    X, y = reproportion_pos(X, y, id_col, train_pos_prop, pos_vals, out_col)
    # Get a binary 1s and 0s version of y
    y_bi = y.copy()
    y_bi[out_col].where(y_bi[out_col].isin(pos_vals), 0, inplace=True)
    y_bi[out_col].mask(y_bi[out_col].isin(pos_vals), 1, inplace=True)
    y_bi[out_col] = y_bi[out_col].astype("int")
    # Perform feature selection to reduce computational complexity
    if id_split:
        X = feature_select_univar(X, y_bi[out_col], [id_col], debugging, outfile)
    else:
        X = feature_select_univar(X, y_bi, [""], debugging, outfile)
    X = feature_select_multivar(X, [id_col], debugging, outfile)
    X_test_final = X_test_final[X.columns]
    if debugging:
        write_print("\nSelected features:", outfile)
        write_print([col for col in X_test_final.columns], outfile)
        write_print(f"Total features: {len(X_test_final.columns)}", outfile)
    for name in model_names:
        # Run the Optuna study for hyperparameter optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X, y_bi, name, k=k, 
                                               metric=score_metric, 
                                               predict_proba=True, 
                                               id_split=id_split, 
                                               id_col=id_col, out_col=out_col, 
                                               debugging=debugging), 
                       n_trials=n_trials)
        # Get the best parameters for training the model
        best_params = study.best_trial.params
        if name == "MLP":
            n_layers = best_params['n_layers']
            n_neurons = best_params['n_neurons']
            best_params["hidden_layer_sizes"] = [n_neurons]*n_layers
            best_params.pop("n_layers")
            best_params.pop("n_neurons")
        elif name == "SVC":
            best_params["probability"] = True
        elif name == "LR":
            best_params["solver"] = "saga"
            best_params["penalty"] = "elasticnet"
        # Tracking #####
        if debugging:
            best_score = study.best_value
            write_print(f"One-holdout test info for {name} model", outfile)
            write_print(f"Selected hyperparameters: {best_params}",outfile)
            write_print(f"Average score: {best_score}", outfile)
            write_print(f"Time: {time.ctime()}\n",outfile)
        # Train and test the model using the selected hyperparameters
        model = new_classifier(name, best_params)
        model.fit(X, y_bi[out_col])
        prediction = model.predict_proba(X_test_final)[:,1]
        one_hold_score = score_metric(y_bi_final[out_col], prediction)
        write_print(f"One-holdout score for {name}: {one_hold_score}", outfile)
        write_print(f"Time: {time.ctime()}\n", outfile)
    
    # Statistical analysis
    # Friedman chi-square
    f_res = friedman(*[scores[:,j] for j in range(len(model_names))])
    p_val = f_res.pvalue
    # Report the p-value
    write_print(f"\nFriedman chi-square p-value: {p_val}\n", outfile)
    # Nemenyi post-hoc
    if p_val < 0.05:
        post_hoc_p_vals = nemenyi(scores)
        # Report the p-values
        write_print(f"\nNemenyi post-hoc p-values:", outfile)
        write_print(post_hoc_p_vals, outfile, end="\n\n")
    write_print(f"Time: {time.ctime()}", outfile)