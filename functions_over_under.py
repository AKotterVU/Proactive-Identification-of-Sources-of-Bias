# functions.py
"""Repository for functions to use throughout the project.
10-18-24: Updated to include bootstrapping as a function and require 
'objective()' to use bootstrapping instead of k-fold cross-validation. 
Added metric name to debugging outfile in 'objective()' Fixed some 
code in 'split()' where 'dropna()' wasn't saving.
10-19-24: Updated to allow 'objective()' to determine whether training 
sets are produced with 'train_pos_prop' or 'backup_pos_prop' 
proportion of positive samples more similar to 'val_pos_prop' than 
'train_pos_prop'. The point is to get models that run well on sets 
with 'val_pos_prop' proportion of positive samples, but class 
imbalances can hinder training, so we allow optuna to determine which 
is most effective for which models.
10-30-24: Updated to include 'disparities_analysis()' function.
11-6-24: Updated 'disparities_analysis()' to include option for PPV at 
fixed sensitivity by subgroup or PPV at fixed threshold that gives an 
overall fixed sensitivity that may vary by subgroup. The fixed 
threshold option can be set to fix the sensitivity on the training set 
or on the overall validation set.
11-15-24: Updated 'objective()' to do k-fold cross-validation 
internally instead of bootstrapping under the assumption that 
'objective()' will be passed only in-bag data for the purpose of 
hyperparameter optimization. In case it's ever relevant, 'bootstrap()' 
now has the option to separately return negative samples that are 
neither in-bag nor out-bag because they were excluded to keep the 
proportions of positive samples consistent. Finally, 'split()' was 
deprecated because sklearn's 'StratifiedGroupKFold' does essentially 
the same thing. (Renamed 'val_pos_prop' to 'test_pos_prop' in 
'bootstrap'.) (RandomForestClassifier now chooses a max_depth 
hyperparameter in 'objective()' to prevent memory errors.)
(Two updates in same version)
11-18-24: Added the functions 'feature_select_univar()' and 
'feature_select_multivar()'.
11-22-24: The 'split()' function was undeprecated and renamed 
'test_split()' for use in creating a one-holdout testing set where IDs 
are preserved in either testing or training but not both. Also, 
changed some hyperparameter search spaces to limit computational 
complexity.
12-4-24: Updated 'test_split()' to actually work. Updated the 
'disparities_analysis()' function to include hyperparameter 
optimization with each bootstrap sample and to do minority-majority 
Wilcoxon statistical testing instead of dense pairwise testing. The 
'disparities_analysis()' function now saves intermediate data.
12-13-24: Added AUPRC with random baseline division normalization.
12-17-24: Made positive proportion for subclasses in 
'disparities_analysis()' uniform and added specified comparison 
subclasses for statistical analysis. Modified 'reproportion_pos()' to 
work regardless of the initial proportion of positive samples.
1-3-25: Added 'sdoh_disparities()' function for disparities analysis 
with optional included SDoH feature(s). Added 'disparity_comparison()' 
to do statistical analysis between old and new disparities to see if 
the new disparities were reduced. Updated 'auprc()' function to be a 
defined function instead of just a lambda function so that it gets a 
defined name.
1-8-25: Changed 'disparities_analysis()' to output a DataFrame and 
save a CSV of disparities of the form "majoritized - minoritized" and 
columns of the form "{demographic_category}_{class}_{score}". Added a 
randomized element to the file names of outputs from 
'disparities_analysis()' and 'sdoh_disparities()' to facilitate 
parallelization. Also updated 'bootstrap()' to only set outcomes as 
ints, not IDs.
1-13-25: The 'sdoh_disparities()' function now accepts 'outfile' as an 
argument.
1-27-25: Added 'resamp_disparity()' function. Changed the function 
'feature_select_multivar()' to suppress errors arising from the 
'ignore_cols' argument containing columns that aren't in 'df'. Changed 
'feature_select_multivar()' to accept 'y' as Series when 'id_split' is 
False.
3-29-25: Updated 'resamp_disparity()' function to allow for resampling 
of multiple SDoH categories simultaneously.
4-19-25: Added 'MLSMOTE_disparity()' function to analyze disparities 
with resampling of multiple SDoH features using MLSMOTE.
4-26-25: Ensured that 'MLSMOTE_disparity()' maintains the same length 
between inputs and outputs.
8-15-25: Added 'combo_disp_MLSMOTE()'. 
functions.
9-5-25: Added 'combo_disp_naive()' and removed some debugging from 
'combo_disp_MLSMOTE()'.
7-13-26: Created this version for resampling assesments using only 
undersampling or only oversampling for sensitivity analysis.
"""



# Import statements



import re
import time
import optuna
import mlsmote
import numpy as np
import pandas as pd

from warnings import warn

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold

from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import EditedNearestNeighbours

from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence \
                                        import variance_inflation_factor as vif

from scipy.stats import kstest
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare as friedman
from scikit_posthocs import posthoc_nemenyi_friedman as nemenyi



# Physiologically normal values for imputation

norm = {"BP_DIAST":90, "BP_SYST":105, "PULSE":80, "RESP_RATE":16, 
        "SPO2":98, "TEMP_C":37, "gcs_total":15, "wbc":7.8, "bun":12.5, 
        "platelet":300, "gluc":105, "proth":11.5, "hemo":14.8, 
        "hemat":43, "bili":0.7, "pao2":87.5, "paco2":40, "fio2":21, 
        "cre":0.8, "lact":1}



# Functions



# Utility I/O


def write_print(text, filename, end="\n"):
    """Prints out 'text' to the console and also appends 'text' to the 
    end of the file 'filename' in a new line."""
    with open(filename, 'a') as f:
        f.write(str(text) + str(end))
    print(text, end=end)


# Data preprocessing


def impute_data(df, id_col):
    """Takes a DataFrame 'df' with timeseries data and returns a copy 
    of it imputed such that missing measurements are filled with past 
    data where possible and physiologically normal data otherwise. 
    Columns corresponding to counts and standard deviations are 
    instead imputed with zeros to indicate that no measurements were 
    made. Columns with neither measurements nor counts are ignored.
    Carry-forward imputation only occurs within rows corresponding to 
    the same patient encounter, as indicated by the column 'id_col'. 
    Assumes that the column name(s) to be imputed are of the following 
    form: '{feature}_{measure}', where 'feature' is the name of a 
    vital sign or lab value and 'measure' is a calculated statistic.
    Feature names must be from the following list:
        BP_DIAST
        BP_SYST
        PULSE
        RESP_RATE
        SPO2
        TEMP_C
        gcs_total
        wbc
        bun
        platelet
        gluc
        proth
        hemo
        hemat
        bili
        pao2
        paco2
        fio2
        cre
        lact
    Measure names must be from the following list:
        avg
        min
        max
        last
        count
        stdevp
    """
    
    # Make a copy of the input
    df_out = df.copy()
    
    # Get a list of the columns to fill with zeros
    count_str = "\w+_(count|stdevp)"
    count_cols = [col for col in df_out.columns if re.search(count_str,col)]
    
    # Impute the appropriate columns with zeros
    for col in count_cols:
        df_out[col] = df_out[col].fillna(0)
    
    # Get a list of the columns to forward fill
    ffill_str = "\w+_(avg|min|max|last)"
    ffill_cols = [col for col in df_out.columns if re.search(ffill_str,col)]
    
    # Forward fill the appropriate columns
    for col in ffill_cols:
        df_out[col] = df_out[[col,id_col]].groupby(id_col).ffill()
    
    # Fill remaining nulls in forward fill columns with normal values
    for col in ffill_cols:
        feat = re.search("(\w+)_(avg|min|max|last)",col).group(1)
        df_out[col] = df_out[col].fillna(norm[feat])
    
    # Return the imputed data
    return df_out



# Model development


def reproportion_pos(X, y, pos_prop=0.5):
    """Given input DataFrame 'X' and binary Series 'y', returns 
    'X_out' and 'y_out' such that the proportion of positive outcomes 
    in the output is equal to 'pos_prop', where a row is considered to 
    have a positive outcome if its value in 'out_col' is found in the 
    list or other array-like 'pos_vals'. Rows are shuffled, but 
    indices between 'X_out' and 'y_out' are preserved.
    This splitting is accomplished by randomly downsampling either 
    positive or negative rows to make the total proportion of positive 
    rows equal to 'pos_prop'.
    This function does not stratify or otherwise distinguish based on 
    columns besides 'out_col'."""
    
    # Get copies of the data to modify and return
    X_out = X.copy()
    y_out = y.copy()
    
    # Check the ratio of positive to total rows
    # Number of positive rows
    n_pos_in = y.sum()
    # Number of negative rows
    n_neg_in = len(y) - n_pos_in
    # Initial ratios
    pos_ratio_in = n_pos_in/len(y)
    neg_ratio_in = n_neg_in/len(y)
    
    # Get number of positive and negative rows needed based on initial ratio
    if pos_ratio_in == pos_prop:
        return X_out, y_out
    elif pos_ratio_in < pos_prop:
        n_pos_out = n_pos_in
        n_neg_out = n_pos_in*(1-pos_prop)/(pos_prop)
    else:
        n_pos_out = n_neg_in*(pos_prop)/(1-pos_prop)
        n_neg_out = n_neg_in
    
    # Get the positive row indices
    y_pos = y_out[y_out == 1]
    pos_idxs = np.random.choice(y_pos.index, int(n_pos_out))
    # Get the negative row indices
    y_neg = y_out[y_out == 0]
    neg_idxs = np.random.choice(y_neg.index, int(n_neg_out))
    # Combine and shuffle the indices
    all_idxs = np.concatenate([pos_idxs, neg_idxs])
    np.random.shuffle(all_idxs)
    # Return X_out and y_out
    X_out = X.loc[all_idxs,:]
    y_out = y.loc[all_idxs]
    return X_out, y_out


def augment_pos(X, y, id_col="CSN", out_col="end_point_6"):
    """Given """
    pass


def test_split(X, y, id_col="CSN", test_pos_prop=0.05, 
               test_split_prop=0.1, pos_vals=["SEPSIS"], 
               out_col="end_point_6"):
    """Given input data 'X' and binary outcome data 'y', returns 
    'X_train', 'y_train', 'X_test', and 'y_test' such that the 
    following are true:
        1. A proportion of the rows in 'X_test' and 'y_test' equal to 
           'test_pos_prop' have a positive outcome.
        2. A proportion of all of the rows with positive values 
           approximately equal to 'test_split_prop' is in 'X_test' and 
           'y_test', with the remainder in 'X_train' and 'y_train'.
        3. Rows with the same value in 'id_col' must be in the same 
           split (that is, in the same of either train or test) so as 
           to avoid leaking data.
    Rows with positive outcomes are defined as rows where the value in 
    'out_col' is in the list 'pos_vals'. IDs are kept in the column 
    'id_col'.
    This splitting is accomplished by first associating each value in 
    'id_col' with either a positive or a negative outcome based on 
    whether any rows with that value for 'id_col' have a positive 
    outcome, then randomly assigning 'test_split_prop' proportion of 
    the positive IDs to testing and the rest to training. All of the 
    positive rows are assigned to training and testing accordingly. 
    Next, as many negative IDs are assigned to testing as necessary to 
    make the proportion of positive test IDs 'test_pos_prop'. Enough 
    rows with a negative testing ID as necessary are assigned to 
    testing to make the proportion of positive rows 'test_pos_prop'. 
    The rest of the negative testing IDs are unused. All negative 
    training IDs are assigned to the training set.
    IMPORTANT ASSUMPTIONS:
    ASSUMES THAT THERE ARE SIGNIFICANTLY MORE NEGATIVE IDS THAN 
    POSITIVE IDS.
    Specifically, there must be at least:
    'test_split_prop'*(1-'test_pos_prop')/'test_pos_prop' 
    times more negative IDs than positive IDs available, and each of 
    those negative IDs must come with an 'id_col' value that does NOT 
    have any associated positive IDs.
    ASSUMES THAT THE TESTING SET IS RELATIVELY SMALL!!! 
    SOME NEGATIVE DATA WITH IDS ASSIGNED TO TESTING WILL NOT BE 
    RETURNED!!!
    Also assumes that the number of rows associated with each 'id_col' 
    value is approximately the same across all 'id_col' values when 
    deciding how to split negative 'id_col' values into training and 
    testing.
    Note also that the relative sizes between training and testing 
    might be nowhere near the proportion given in 'test_split_prop', 
    since that refers exclusively to the ratio of positive samples. 
    This is done to counter imbalance issues during training when 
    positive samples are relatively rare.
    """
    
    # Input validation
    assert test_split_prop >= 0
    assert test_split_prop <= 1
    assert test_pos_prop >= 0
    assert test_pos_prop <= 1
    
    # Make sure 'y' is binary, then get relevant columns
    y_bi = y.copy()
    y_bi[out_col].where(y_bi[out_col].isin(pos_vals), 0, inplace=True)
    y_bi[out_col].mask(y_bi[out_col].isin(pos_vals), 1, inplace=True)
    y_out = y_bi[out_col].copy()
    
    # Label the IDs based on whether they ever contain a positive value
    y_id = y_bi.loc[:,[id_col,out_col]].groupby([id_col]).mean()
    y_id[out_col].where(y_id[out_col] > 0, 0, inplace=True)
    y_id[out_col].mask(y_id[out_col] > 0, 1, inplace=True)
    
    # How many IDs are needed for training/validation and pos/neg?
    n_pos_ids = int(y_id[out_col].sum())
    n_neg_ids = int(len(y_id[out_col]) - n_pos_ids)
    n_pos_ids_test = round(n_pos_ids*test_split_prop)
    n_neg_ids_test = round(n_pos_ids_test*(1-test_split_prop)/test_split_prop)
    n_pos_ids_train = n_pos_ids - n_pos_ids_test
    n_neg_ids_train = n_neg_ids - n_neg_ids_test
    
    # Set size validation
    assert n_pos_ids_test > 0
    assert n_neg_ids_test > 0
    assert n_pos_ids_train > 0
    assert n_neg_ids_train > 0
    
    # Split the positive IDs
    p_test_ids = np.random.choice(y_id[y_id[out_col] == 1].index, 
                                  n_pos_ids_test, False)
    p_train_ids = y_id[y_id[out_col] == 1]
    p_train_ids = p_train_ids[~p_train_ids.index.isin(p_test_ids)].index
    # Split the negative IDs
    n_test_ids = np.random.choice(y_id[y_id[out_col] == 0].index, 
                                  n_neg_ids_test, False)
    n_train_ids = y_id[y_id[out_col] == 0]
    n_train_ids = n_train_ids[~n_train_ids.index.isin(n_test_ids)].index
    # Combine the training and testing IDs
    test_ids = np.concatenate([p_test_ids, n_test_ids])
    train_ids = np.concatenate([p_train_ids, n_train_ids])
    
    # Get the training rows
    X_train = X[X[id_col].isin(train_ids)]
    y_train = y[y[id_col].isin(train_ids)]
    
    # Get the testing rows
    y_p_test_raw = y[y[id_col].isin(p_test_ids)]
    y_p_test = y_p_test_raw[y_p_test_raw[out_col].isin(pos_vals)]
    y_n_test_raw = y[y[id_col].isin(n_test_ids)]
    num_n_test_idxs = round(len(y_p_test)*(1-test_pos_prop)/test_pos_prop)
    y_n_test_idxs = np.random.choice(y_n_test_raw.index, num_n_test_idxs, 
                                     False)
    y_n_test = y.loc[y_n_test_idxs,:]
    y_test = pd.concat([y_p_test,y_n_test])
    X_test = X.loc[y_test.index,:]
    
    # Return the split data
    return X_train, y_train, X_test, y_test


def feature_select_univar(X, y, ignore_cols=["CSN"], debugging=False, 
                          outfile=None, alpha=0.05):
    """Produces a copy of a given DataFrame 'X' where all features are 
    removed that do not show statistically significant differences 
    between positive and negative samples, where labels are contained 
    in a Series 'y'. Uses the two-sample, two-tailed Kolmogorov-
    Smirnov test with alpha set to 0.05. Features contained in the 
    list 'ignore_cols' are not considered and instead passed directly 
    to the output DataFrame."""
    
    # Separate the data into positive and negative samples
    X_pos = X[y == 1]
    X_neg = X[y == 0]
    
    # If debugging, output the p-values for each column
    if debugging:
        if outfile is None:
            print("Univariate selection p-values:")
        else:
            write_print("Univariate selection p-values:", outfile)
    
    # Calculate the p-values for each feature
    p_values = {col:None for col in X.columns}
    for col in X.columns:
        res = kstest(X_pos[col].values.flatten(), X_neg[col].values.flatten())
        p_values[col] = res.pvalue
        # If debugging, output the p-values for each column
        if debugging:
            if outfile is None:
                print(col, p_values[col])
            else:
                write_print(f"{col}: {p_values[col]} {p_values[col] < alpha}", 
                            outfile)
    
    # Keep only the features that have p < alpha or that were ignored
    cols_to_keep = [col for col in X.columns if p_values[col] < alpha or \
                                                            col in ignore_cols]
    
    # Return the feature-selected DataFrame
    return X[cols_to_keep]


def feature_select_multivar(df, ignore_cols=["CSN"], debugging=False, 
                            outfile=None, vif_max=10):
    """Returns a copy of a given DataFrame 'df' where all features are 
    removed that exhibit strong multicolinearity as defined by VIF 
    above a certain cutoff value. Specified columns to be ignored are 
    included without participating in the colinearity calculations."""
    
    # Get a copy of the DataFrame with the right columns
    # Add a constant (https://github.com/statsmodels/statsmodels/issues/2376)
    X = add_constant(df)
    X = X.drop(columns=ignore_cols, errors="ignore")
    
    # Calculate the VIF for multivariate feature selection
    vifs = pd.Series([vif(X.values.astype(float), i) for i in \
                                                       range(len(X.columns))], 
                     index=X.columns)
    
    # If debugging, output the p-values for each column
    if debugging:
        if outfile is None:
            print("Multivariate selection p-values:")
            for col in vifs.index:
                print(col, vifs[col])
        else:
            write_print("Multivariate selection p-values:", outfile)
            for col in vifs.index:
                write_print(f"{col}: {vifs[col]} {vifs[col] < vif_max}", 
                            outfile)
    
    # Keep only the features that have small enough VIFs
    vifs = vifs.drop("const")
    X = X.drop(columns="const")
    cols_to_keep_vif = X.columns[vifs <= vif_max]
    cols_to_keep = [col for col in df.columns if col in cols_to_keep_vif or \
                                                            col in ignore_cols]
    
    # Return the feature-selected DataFrame
    return df[cols_to_keep].copy()


def model_optimizer(name, X, y, score_metric=accuracy_score, 
                    predict_proba=False, n_trials=30, 
                    pos_vals=[1], id_split=False, id_col="CSN", 
                    out_col="end_point_6"):
    """Finds the best features and hyperparameters to use for a given 
    model type 'name' on training data 'X' (DataFrame) and 'y' (Series 
    or DataFrame) using 10-fold cross-validation.
    Values in 'y' are considered positive if they are in 'pos_vals'.
    All other arguments are as given in 'objective()' above."""
    
    # Get a binary 1s and 0s version of y
    y_bi = y.copy()
    if id_split:
        y_bi[out_col].where(y_bi[out_col].isin(pos_vals), 0, inplace=True)
        y_bi[out_col].mask(y_bi[out_col].isin(pos_vals), 1, inplace=True)
        y_bi[out_col] = y_bi[out_col].astype("int")
    else:
        y_bi.where(y_bi.isin(pos_vals), 0, inplace=True)
        y_bi.mask(y_bi.isin(pos_vals), 1, inplace=True)
        y_bi = y_bi.astype("int")
    
    # Perform feature selection to reduce computational complexity
    if id_split:
        X_uni = feature_select_univar(X, y_bi[out_col], [id_col])
    else:
        X_uni = feature_select_univar(X, y_bi, [""])
    X_multi = feature_select_multivar(X_uni, [id_col])
    
    # Run the Optuna study for hyperparameter optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_multi, y_bi, name, k=10, 
                                           metric=score_metric, 
                                           predict_proba=True, 
                                           id_split=id_split, 
                                           id_col=id_col, out_col=out_col), 
                   n_trials=n_trials)#, gc_after_trial=True)
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
    
    # Return
    return X_multi.columns, best_params


def bootstrap(X, y, train_pos_prop=0.5, test_pos_prop=0.05, 
              train_split_prop=0.7, pos_vals=[1], id_split=False, 
              id_col="CSN", out_col="end_point_6", original_size=True, 
              return_X_neg=False):
    """Given input data 'X' with binary outcome 'y' (where 'X' and 'y' 
    have the same indices), returns a bootstrap sample 'X_train', 
    'y_train', 'X_val', 'y_val' such that: 
        1. 'train_pos_prop' proprotion of the training outputs are 
           positive outcomes
        2. 'test_pos_prop' proportion of the validation outputs are 
           positive outcomes, and 
        3. the number of positive samples in the training outputs is 
           equal to 'train_split_prop' proportion of the original 
           number of positive samples in the input data.
    Note that 'train_split_prop' must be a value in the inclusive 
    range [0,1] for sample size calculations.
    If 'y' has outcomes that aren't 1 or 0, specifying 'pos_vals' 
    indicates which values should be considered positive. All other 
    values are considered negative outcomes. The outputs 'y_train' and 
    'y_val' have strictly binary outcomes, 0 or 1.
    This bootstrapping method is designed specifically to account for 
    binary class imbalance. Additionally, this method can account for 
    time-series data by keeping all samples of a given ID (e.g., the 
    same patient over a certain time) together in either training or 
    validation so as to prevent information leakage. To enable this 
    feature, specify 'id_split'=True and provide values for 'id_col' 
    and 'out_col'. Note that this will divide 'id_col' values into 
    training and validation based on the specified proportions and 
    whether any positive samples exist within a value of 'id_col' 
    before any samples are drawn. Also note that, while 'y' should be 
    1D (like a Series) if 'id_split'=False, 'y' should be a DataFrame 
    with columns 'id_col' and 'out_col' if 'id_split'=True. The 
    returned 'y_train' and 'y_val' have columns 'id_col' and 'out_col' 
    if 'id_split'=True.
    If 'id_split'=True and you want to have a positive training set 
    equal in size to the original positive set, as in normal 
    bootstrapping, set 'original_size'=True. The proportion of IDs in 
    'id_col' allowed to be in the training set will be 
    'train_split_prop', but the number of positive rows in the 
    training set will be the same size as in the original. The 
    positive rows in the validation set will be exactly all of the 
    positive rows with a validation ID in 'id_col'.
    Optionally, returns 'X_neg' consisting of all the unused samples 
    when 'return_X_neg' is True. Because of the handling of class 
    imbalance, all positive samples are accounted for at least once in 
    'X_train' and 'X_val', but many negative samples might not be 
    returned in either. Returning 'X_neg' accounts for all of the 
    unused negative samples. If 'id_split' is True, then two 
    DataFrames 'X_neg_train' and 'X_neg_val' are returned to keep the 
    training and validation IDs separate. In the 'id_split'=True case, 
    it's possible that not all positive samples will be returned, but 
    the X_neg DataFrames will still only have negative samples.
    IMPORTANT ASSUMPTION:
    ASSUMES THAT THERE ARE SIGNIFICANTLY MORE NEGATIVE ROWS THAN 
    POSITIVE ROWS. Specifically, there must be at least:
    'train_split_prop'*(1-'train_pos_prop')/'train_pos_prop' +
    1-'train_split_prop'*(1-'test_pos_prop')/'test_pos_prop'
    times more negative rows than positive rows available, and each of 
    those negative rows must come with an 'id_col' value that does NOT 
    have any associated positive rows if 'id_split' is True. If this 
    assumption is not met, an error will be raised.
    Also assumes that the number of rows associated with each 'id_col' 
    value is approximately the same across all 'id_col' values when 
    deciding how to split negative 'id_col' values into training and 
    testing.
    This is essentially a more efficient version of 'split()' with 
    drawing with replacement."""
    
    # Input validation
    assert train_split_prop >= 0
    assert train_split_prop <= 1
    
    # Make sure 'y' is binary, then get relevant columns
    y_bi = y.copy()
    if id_split:
        y_bi[out_col].where(y_bi[out_col].isin(pos_vals), 0, inplace=True)
        y_bi[out_col].mask(y_bi[out_col].isin(pos_vals), 1, inplace=True)
        y_out = y_bi[out_col].copy()
    else:
        y_bi.where(y_bi.isin(pos_vals), 0, inplace=True)
        y_bi.mask(y_bi.isin(pos_vals), 1, inplace=True)
        y_out = y_bi.copy()
    
    # Check the ratio of positive to negative rows
    # Number of positive rows
    n_pos = y_out.sum()
    # Number of negative rows
    n_neg = len(y_out) - n_pos
    # Number of negative training rows needed
    if id_split and original_size:
        n_pos_train = int(n_pos)
    else:
        n_pos_train = round(n_pos*train_split_prop)
    n_neg_train = round(n_pos_train*(1-train_pos_prop)/train_pos_prop)
    # Approximate number of validation rows needed
    n_pos_val = round(n_pos*(1-train_split_prop))
    n_neg_val = round(n_pos_val*(1-test_pos_prop)/test_pos_prop)
    assert n_neg >= n_neg_train + n_neg_val
    
    # Divide the dataset based on whether IDs are kept together or not
    if id_split:
        # Label the IDs based on whether they ever contain a positive value
        y_id = y_bi.loc[:,[id_col,out_col]].groupby([id_col]).mean()
        y_id[out_col].where(y_id[out_col] > 0, 0, inplace=True)
        y_id[out_col].mask(y_id[out_col] > 0, 1, inplace=True)
        
        # How many IDs are needed for training/validation and pos/neg?
        n_pos_ids = int(y_id[out_col].sum())
        n_p_id_train = round(n_pos_ids*train_split_prop)
        n_neg_ids_train = round(n_p_id_train*(1-train_pos_prop)/train_pos_prop)
        n_pos_ids_val = n_pos_ids - n_p_id_train
        n_neg_ids_val = round(n_pos_ids_val*(1-test_pos_prop)/test_pos_prop)
        # If more negative IDs are available than necessary, use more
        n_neg_ids = len(y_id) - n_pos_ids
        neg_need = n_neg_ids_train + n_neg_ids_val
        if n_neg_ids == neg_need:
            pass
        elif n_neg_ids < neg_need:
            raise ValueError(f"{n_neg_ids} neg IDs available, need {neg_need}")
        else:
            neg_ids_left = n_neg_ids - neg_need
            n_neg_ids_train += neg_ids_left//2
            n_neg_ids_val += neg_ids_left//2
        
        # Get the training IDs
        p_train_ids = np.random.choice(y_id[y_id[out_col] == 1].index, 
                                       n_p_id_train, False)
        n_train_ids = np.random.choice(y_id[y_id[out_col] == 0].index, 
                                       n_neg_ids_train, False)
        train_ids = np.concatenate([p_train_ids, n_train_ids])
        # Get the validation IDs
        val_ids = y_id[~y_id.index.isin(train_ids)].index
        
        # Get the training row indices
        y_train_raw = y_bi[y_bi[id_col].isin(train_ids)]
        y_train_pos = y_train_raw[y_train_raw[out_col] == 1]
        y_train_pos_idxs = np.random.choice(y_train_pos.index, n_pos_train)
        y_train_neg = y_train_raw[y_train_raw[out_col] == 0]
        y_train_neg_idxs = np.random.choice(y_train_neg.index, n_neg_train)
        train_idxs = np.concatenate([y_train_pos_idxs, y_train_neg_idxs])
        # Get the validation row indices
        y_val_raw = y_bi[y_bi[id_col].isin(val_ids)]
        y_val_pos = y_val_raw[y_val_raw[out_col] == 1]
        y_val_pos_idxs = np.array(y_val_pos.index)
        y_val_neg = y_val_raw[y_val_raw[out_col] == 0]
        n_neg_val_idxs = round(len(y_val_pos)*(1-test_pos_prop)/test_pos_prop)
        y_val_neg_idxs = np.random.choice(y_val_neg.index, n_neg_val_idxs, 
                                          False)
        val_idxs = np.concatenate([y_val_pos_idxs, y_val_neg_idxs])
        
        # Fill in the training and validation sets
        X_train = X.loc[train_idxs,:]
        y_train = y_bi.loc[train_idxs,[id_col,out_col]]
        X_val = X.loc[val_idxs,:]
        y_val = y_bi.loc[val_idxs,[id_col,out_col]]
        
        # If requested, get the unused samples
        if return_X_neg:
            # Training
            xtra_train = y_train_neg[~y_train_neg.index.isin(y_train_neg_idxs)]
            X_neg_train = X.loc[xtra_train.index,:]
            # Validation
            xtra_val = y_val_neg[~y_val_neg.index.isin(y_val_neg_idxs)]
            X_neg_val = X.loc[xtra_val.index,:]
        
        # Make sure the outcomes are properly binary
        y_train[out_col] = y_train[out_col].astype("int")
        y_val[out_col] = y_val[out_col].astype("int")
        
    else:
        # Divide into positive and negative indices
        y_pos = y_bi[y_bi == 1]
        y_neg = y_bi[y_bi == 0]
        # Get the training row indices
        y_train_pos_idxs = np.random.choice(y_pos.index, n_pos_train)
        y_train_neg_idxs = np.random.choice(y_neg.index, n_neg_train)
        train_idxs = np.concatenate([y_train_pos_idxs, y_train_neg_idxs])
        # Get the validation row indices
        y_val_p_idxs = y_pos[~y_pos.index.isin(y_train_pos_idxs)].index
        y_val_n_raw = y_neg[~y_neg.index.isin(y_train_neg_idxs)]
        n_neg_val_idxs = round(len(y_val_p_idxs)*\
                                               (1-test_pos_prop)/test_pos_prop)
        y_val_neg_idxs = np.random.choice(y_val_n_raw.index, n_neg_val_idxs, 
                                          False)
        val_idxs = np.concatenate([y_val_p_idxs, y_val_neg_idxs])
        
        # Fill in the training and validation sets
        X_train = X.loc[train_idxs,:]
        y_train = y_bi[train_idxs]
        X_val = X.loc[val_idxs,:]
        y_val = y_bi[val_idxs]
        
        # If requested, get the unused samples
        if return_X_neg:
            combined_idxs = np.concatenate([train_idxs, val_idxs])
            X_neg = X[~X.index.isin(combined_idxs)]
        
        # Make sure the outcomes are properly binary
        y_train = y_train.astype("int")
        y_val = y_val.astype("int")
        
    # Return with proper binary outcomes
    if not return_X_neg:
        return X_train, y_train, X_val, y_val
    # Account for the unused negative samples if requested
    else:
        if id_split:
            return X_train, y_train, X_val, y_val, X_neg_train, X_neg_val
        else:
            return X_train, y_train, X_val, y_val, X_neg


def new_classifier(name, args={}):
    """Utility function to instantiate a given classifier. Given the 
    name 'name' of a classifier, returns a new classifier of the given 
    type. Passes all arguments in dictionary 'args' into the 
    classifier to instantiate."""
    # Support Vector Machine
    if name.lower() in ["svm", "svc", "svmc", "support vector machine"]:
        return SVC(**args)
    # Random Forest
    elif name.lower() in ["rf", "rfc", "random forest"]:
        return RandomForestClassifier(**args)
    # Gradient Boosting
    elif name.lower() in ["gb", "gbc", "gradient boosting"]:
        return GradientBoostingClassifier(**args)
    # K-Nearest Neighbors
    elif name.lower() in ["knn", "knnc", "k neighbors", "k-nearest neighbors"]:
        return KNeighborsClassifier(**args)
    # Logistic Regression
    elif name.lower() in ["lr", "lrc", "log reg", "logistic regression"]:
        return LogisticRegression(**args)
    # Multi-Layer Perceptron
    elif name.lower() in ["mlp", "mlpc", "multi-layer perceptron"]:
        return MLPClassifier(**args)
    # Gaussian Process
    elif name.lower() in ["gp", "gpc", "gaussian process"]:
        return GaussianProcessClassifier(**args)


def objective(trial, X_in, y_in, name, metric=accuracy_score, k=10, 
              predict_proba=False, id_split=False, id_col="CSN", 
              out_col="end_point_6", debugging=False):
    """Finds the best hyperparameters and performance for a model of 
    the architecture given by 'model_name' in terms of the performance 
    measure 'metric' (default accuracy, must be an sklearn-compatible 
    performance function for classification). If 'metric' scores 
    against probabilities instead of predicted classes, as with AUROC, 
    then 'predict_proba' must be True. Otherwise, 'predict_proba' must 
    be False, which is the default.
    Performs stratified 'k'-fold cross-validation (default 'k'=10) to 
    determine best hyperparameters and average performance of those 
    hyperparameters. Splitting is done row by row and stratified based 
    on the binary value in 'y_in' without regard for any other 
    features. Alternatively, if 'id_split' is True, then the 'k'-fold 
    cross-validation is done in a way that keeps rows with the same 
    'id_col' value together in either training or validation as well 
    as stratifies by 'out_col'.
    Requires training data 'X_in' and outcomes 'y_in' to be given as 
    arguments in addition to 'model_name'. 'X_in' and 'y_in' must have 
    the same indices, all features in 'X_in' must be model input 
    features, and 'y_in' must be a 1D array such as a Series with 
    binary classes if 'id_split' is False.
    Alternatively, if 'id_split' is True, then 'X_in' and 'y_in' 
    should both be DataFrames with the same columns as described 
    earlier plus the addition of the column 'id_col' in both. The 
    outcome in 'y_in' should be in the column 'out_col'. If 'id_split' 
    is True, then all splits in the cross-validation are done such 
    that rows with the same value in 'id_col' are always kept 
    together. Stratification is done according to sklearn's 
    StratifiedGroupKFold class, which prioritizes keeping 'id_col' 
    values together over perfect 'out_col' stratification.
    Uses the same conversion of model names to architectures used in 
    the function 'new_classifier' here.
    If 'debugging' is true, then progress is tracked by printing flags 
    to the console and to a file with a systematic name.
    """
    
    # If debugging, get a filename for tracking purposes
    if debugging:
        date_str = "{}_{}_{}".format(*time.localtime()[0:3])
        outfile = f"objective_tracking_{name}_{date_str}.txt"
        write_print("Starting new iteration of objective function",outfile)
        write_print(f"Metric: {str(metric).split(' ')[1]}", outfile)
        write_print(f"Time: {time.ctime()}\n",outfile)
    
    # Get the hyperparameters based on the model type
    params = {}
    # Support Vector Machine
    if name.lower() in ["svm", "svc", "svmc", "support vector machine"]:
        params["C"] = trial.suggest_float("C", 1e-4, 1e2, log=True)
        params["kernel"] = trial.suggest_categorical("kernel", 
                                          ["linear", "poly", "rbf", "sigmoid"])
        params["probability"] = predict_proba
    # Random Forest
    elif name.lower() in ["rf", "rfc", "random forest"]:
        params["n_estimators"] = trial.suggest_int("n_estimators", 10, 150)
        params["max_depth"] = trial.suggest_int("max_depth", 2, 10)
    # Gradient Boosting
    elif name.lower() in ["gb", "gbc", "gradient boosting"]:
        params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 
                                                      1e3, log=True)
        params["n_estimators"] = trial.suggest_int("n_estimators", 10, 150)
        params["max_depth"] = trial.suggest_int("max_depth", 2, 10)
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    # K-Nearest Neighbors
    elif name.lower() in ["knn", "knnc", "k neighbors", "k-nearest neighbors"]:
        max_n = min(len(X_in)//k,100)
        params["n_neighbors"] = trial.suggest_int("n_neighbors", 1, max_n)
        params["weights"] = trial.suggest_categorical("weights", ["uniform", 
                                                                  "distance"])
    # Logistic Regression
    elif name.lower() in ["lr", "lrc", "log reg", "logistic regression"]:
        params["solver"] = "saga"
        params["penalty"] = "elasticnet"
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0, 1)
        params["C"] = trial.suggest_float("C", 1e-5, 1e3, log=True)
    # Multi-Layer Perceptron
    elif name.lower() in ["mlp", "mlpc", "multi-layer perceptron"]:
        n_layers = trial.suggest_int("n_layers", 1, 10)
        n_neurons = trial.suggest_int("n_neurons", 10, 200)
        params["hidden_layer_sizes"] = [n_neurons]*n_layers
        params["activation"] = trial.suggest_categorical("activation", 
                                      ["identity", "logistic", "tanh", "relu"])
        params["alpha"] = trial.suggest_float("alpha", 1e-7, 1e-1, log=True)
        params["learning_rate_init"]=trial.suggest_float("learning_rate_init", 
                                                                       1e-6, 1)
    # Gaussian Process
    elif name.lower() in ["gp", "gpc", "gaussian process"]:
        pass
    
    # Tracking
    if debugging:
        write_print(f"Parameters for {name} model: {params}",outfile)
        write_print(f"Time: {time.ctime()}\n",outfile)
    
    # Instantiate the model
    model = new_classifier(name, params)
    
    # Tracking
    if debugging:
        write_print(f"Instantiated {name} model",outfile)
        write_print(f"Time: {time.ctime()}\n",outfile)
    
    # Track the scores through each split to get the average performance
    scores = np.zeros(k)
    train_scores = np.zeros(k)
    
    # Pick a cross-validation method based on whether IDs are kept together
    if id_split:
        kfold = StratifiedGroupKFold(k)
        splits = enumerate(kfold.split(X_in, y_in[out_col], X_in[id_col]))
    else:
        kfold = StratifiedKFold(k)
        splits = enumerate(kfold.split(X_in, y_in))
    
    # Cross validate the model
    for i, (train_idxs, val_idxs) in splits:
        # Get the correct columns if multiple columns are given
        if id_split:
            X_train = X_in.iloc[train_idxs,:].drop(columns=id_col)
            y_train = y_in.iloc[train_idxs,:].loc[:,out_col]
            X_val = X_in.iloc[val_idxs,:].drop(columns=id_col)
            y_val = y_in.iloc[val_idxs,:].loc[:,out_col]
        else:
            X_train = X_in.loc[train_idxs,:]
            y_train = y_in.loc[train_idxs]
            X_val = X_in.loc[val_idxs,:]
            y_val = y_in.loc[val_idxs]
        
        # Keep track of progress if requested
        if debugging:
            write_print(f"Starting fold {i+1}/{k} for {name}",outfile)
            write_print(f"Training size: {len(X_train)}",outfile)
            write_print(f"Validation size: {len(X_val)}",outfile)
            write_print(f"Positive training samples: {y_train.sum()}",outfile)
            write_print(f"Positive validation samples: {y_val.sum()}",outfile)
            write_print(f"Time: {time.ctime()}\n",outfile)
        # Train the model
        model.fit(X_train, y_train)
        # Keep track of progress
        if debugging:
            write_print(f"{name} model trained",outfile)
            write_print(f"Time: {time.ctime()}\n",outfile)
        # Validate the model
        if predict_proba:
            if debugging:
                write_print(f"Classes: {model.classes_}",outfile)
            prediction = model.predict_proba(X_val)[:,1]
            train_prediction = model.predict_proba(X_train)[:,1]
        else:
            prediction = model.predict(X_val)
            train_prediction = model.predict(X_train)
        # Keep track of progress
        if debugging:
            write_print(f"{name} model produced prediction",outfile)
            write_print(f"Time: {time.ctime()}\n",outfile)
        # Save the scores
        scores[i] = metric(y_val, prediction)
        train_scores[i] = metric(y_train, train_prediction)
        # Keep track of progress
        if debugging:
            write_print(f"Score: {scores[i]}",outfile)
            write_print(f"Train score: {train_scores[i]}",outfile)
            write_print(f"Time: {time.ctime()}\n\n",outfile)
    
    # Return the average performance
    # Keep track of progress
    if debugging:
        write_print(f"All scores: \n{scores}",outfile)
        write_print(f"All train scores: \n{train_scores}",outfile)
        write_print(f"Time: {time.ctime()}\n\n",outfile)
    return np.mean(scores)


# Model evaluation


# Label imbalance ratio
def IRLbl(L):
    """Given a set of integer label frequencies 'L', returns an array 
    with the IRLbl of each label."""
    return np.array([np.max(L)/l for l in L])


# Average precison (AUPRC) without averaging over classes
def auprc(y_true, y_probs):
    """AUPRC (average precision) as defined by the sklearn function 
    'average_precision_score()' with the argument 'average=None'."""
    return average_precision_score(y_true, y_probs, average=None)


# Average precison (AUPRC) normalized by random baseline division
def auprc_norm(y_true, y_probs):
    """Normalizes AUPRC (average precision) by dividing by the 
    proportion of positive samples in 'y_true'. Assumes 'y_true' is 
    binary with 1's and 0's."""
    baseline = sum(y_true)/len(y_true)
    return average_precision_score(y_true, y_probs, average=None)/baseline


def ppv_fixed_thresh(y_true, y_probs, thresh=0.5):
    """An sklearn-compatible function for evaluating binary model 
    precision with a fixed probability threshold at or above which to 
    predict a positive outcome. Requires predicted probabilities 
    'y_probs' such as those produced by predict_proba(), not binary 
    predictions."""
    
    # Get the binary predictions for the indicated threshold
    thresh_pred = [int(p >= thresh) for p in y_probs]
    
    # Return the precision for the given fixed sensitivity
    return precision_score(y_true, thresh_pred)


def recall_fixed_thresh(y_true, y_probs, thresh=0.5):
    """An sklearn-compatible function for evaluating binary model 
    recall with a fixed probability threshold at or above which to 
    predict a positive outcome. Requires predicted probabilities 
    'y_probs' such as those produced by predict_proba(), not binary 
    predictions."""
    
    # Get the binary predictions for the indicated threshold
    thresh_pred = [int(p >= thresh) for p in y_probs]
    
    # Return the precision for the given fixed sensitivity
    return recall_score(y_true, thresh_pred)


def get_sens_threshold(y_true, y_probs, sens=0.9):
    """Given binary true values and predicted positive probabilities, 
    returns the threshold between 0 and 1 that gives a prediction 
    sensitivity closest to 'sens'."""
    
    # Find the ROC information for fixed thresholding
    fprs, tprs, threshs = roc_curve(y_true, y_probs)
    
    # Get the threshold that gives the sensitivity closest to the fixed value
    sens_idx = np.argmin(np.abs(tprs - sens))
    return threshs[sens_idx]


def ppv_fixed_sens(y_true, y_probs, sens=0.9):
    """An sklearn-compatible function for evaluating binary model 
    precision with a fixed sensitivity/recall of 'sens'. Requires 
    predicted probabilities 'y_probs', not binary predictions."""
    # Get the threshold for the sensitivity
    sens_thresh = get_sens_threshold(y_true, y_probs, sens)
    # Return the PPV
    return ppv_fixed_thresh(y_true, y_probs, sens_thresh)


def disparities_analysis(name, X, y, dem_cols=["GENDER", "RACE"], 
                         majority_classes=["M", "White or Caucasian"],
                         score_funcs=[auprc, auprc_norm, 
                                      ppv_fixed_sens, 
                                      recall_fixed_thresh, 
                                      roc_auc_score], 
                         probas=[True, True, True, True, True], 
                         threshs=[False, False, False, True, False], 
                         fix_thresh_on_val=True, k=10, 
                         train_pos_prop=0.5, val_pos_prop=0.05, 
                         train_split_prop=0.7, pos_vals=[1], 
                         id_split=True, id_col="CSN", 
                         out_col="end_point_6", original_size=True, 
                         n_trials=30):
    """Given the name of an sklearn-compatible model 'name', DataFrame 
    'X', and outcomes Series 'y', prints and saves a report of 
    relative model performance stratified by the columns 'dem_cols' in 
    'X', then saves a CSV of disparities of the form "majoritized - 
    minoritized" and with columns of the form:
        "{demographic_category}_{class}_{score}"
    and also saves a CSV of raw model performances.
    The model is not trained on data in 'dem_cols'.
    The evaluated scores are given in the array-like 'score_funcs', 
    all of which must be sklearn-compatible. The corresponding 
    array-like 'probas' contains a boolean for each score based on 
    whether the model must make a prediction with 'predict_proba()' 
    (True) or just 'predict()' (False). Similarly, the array-like 
    'threshs' contains a boolean for each score based on whether the 
    score needs a threshold argument passed to it. The first score is 
    used for hyperparameter optimization.
    If one of the scores in 'score_funcs' is 'ppv_fixed_thresh', then 
    'fix_thresh_on_val' determines whether the fixed threshold is 
    calculated from the overall validation set (True) or from the 
    training set (False). Note that the model must have the 
    'predict_proba()' method for this to work.
    Analyses are performed on 'k' bootstrapped samples.
    The remaining arguments are as given in 'bootstrap()'.
    Statistical analyses are Wilcoxon signed-rank tests between 
    classes in a demographic column and a non-minoritized class 
    designated in 'majority_classes', which must be an array-like of 
    the same length as 'dem_cols'.
    Saves intermediate and final results to a file with the current 
    date and model name.
    """
    
    # Set up the outfile for saving intermediate results
    date_str = "{}_{}_{}".format(*time.localtime()[0:3])
    rn = np.random.randint(1e4,1e5)
    outfile = f"baseline_disparities_{name}_{date_str}_{rn}.txt"
    write_print(f"Starting disparities analysis: {time.asctime()}", outfile)
    
    # Which classes are there for each demographic category?
    classes = {col:X[col].unique() for col in dem_cols}
    
    # Initialize arrays to hold scores
    num_scores = len(score_funcs)
    overall_scores = [np.zeros(k) for _ in score_funcs]
    scores = [{col:{clss:np.zeros((k,1)) for clss in classes[col]} \
                                                         for col in dem_cols} \
                                                          for _ in score_funcs]
    # Thresholds, if applicable
    if any(threshs):
        saved_threshes = []
    # Array to hold number in each class in each bootstrap sample
    num_clss = {col:{clss:np.zeros((k,1)) for clss in classes[col]} \
                                                           for col in dem_cols}
    pos_clss = {col:{clss:np.zeros((k,1)) for clss in classes[col]} \
                                                           for col in dem_cols}
    
    # Bootstrap the model k times
    for i in range(k):
        # Get the bootstrap sample
        X_train, y_train, X_val, y_val = bootstrap(X, y, train_pos_prop, 
                                                   val_pos_prop, 
                                                   train_split_prop, pos_vals, 
                                                   id_split, id_col, 
                                                   out_col, original_size)
        # Drop the demographic category columns from the training data
        X_train.drop(columns=dem_cols, inplace=True)
        # Make a copy of the validation data without demographic categories
        X_val_clean = X_val.drop(columns=dem_cols)
        # Get the best hyperparameters and model input features
        cols, best_params = model_optimizer(name, X_train, y_train, 
                                            score_funcs[0], probas[0], 
                                            n_trials, [1], id_split, 
                                            id_col, out_col)
        # Get the right columns based on whether IDs are kept together
        if id_split:
            X_train = X_train.drop(columns=id_col)
            y_train = y_train.loc[:,out_col]
            X_val = X_val.drop(columns=id_col)
            X_val_clean = X_val_clean.drop(columns=id_col)
            y_val = y_val.loc[:,out_col]
        # Instantiate the model
        model = new_classifier(name, best_params)
        # Train the model
        model.fit(X_train, y_train)
        # Find a 90% sensitivity threshold for the model
        if any(threshs):
            if fix_thresh_on_val:
                probs_pred = model.predict_proba(X_val_clean)[:,1]
                sens_thresh = get_sens_threshold(y_val, probs_pred)
            else:
                probs_pred = model.predict_proba(X_train)[:,1]
                sens_thresh = get_sens_threshold(y_train, probs_pred)
            saved_threshes.append(sens_thresh)
        # Validate the model over each score function
        for j, score in enumerate(score_funcs):
            # Validate the model
            if probas[j]:
                prediction = model.predict_proba(X_val_clean)[:,1]
            else:
                prediction = model.predict(X_val_clean)
            # Get and save the overall model performance
            # Check if more arguments need to be passed
            if threshs[j]:
                overall_scores[j][i] = score(y_val, prediction, sens_thresh)
            else:
                overall_scores[j][i] = score(y_val, prediction)
            # Iterate through each demographic category to be assessed
            for col in dem_cols:
                # Iterate through each class in the demographic category
                for clss in classes[col]:
                    # Get the data for the class
                    X_clss = X_val.where(X_val[col]==clss).dropna(subset=[col])
                    X_clss.drop(columns=dem_cols, inplace=True)
                    y_clss = y_val.loc[X_clss.index]
                    # Set the proportion of positive samples in the subset
                    X_clss, y_clss = reproportion_pos(X_clss, y_clss, 
                                                      val_pos_prop)
                    # Tracking info
                    num_clss[col][clss][i] = len(y_clss)
                    pos_clss[col][clss][i] = sum(y_clss)
                    # Get the model predictions for the class
                    if probas[j]:
                        prediction_clss = model.predict_proba(X_clss)[:,1]
                    else:
                        prediction_clss = model.predict(X_clss)
                    # Get and save the model performance for this class
                    # Check if more arguments need to be passed
                    if threshs[j]:
                        scores[j][col][clss][i] = score(y_clss, 
                                                        prediction_clss, 
                                                        sens_thresh)
                    else:
                        scores[j][col][clss][i] = score(y_clss, 
                                                        prediction_clss)
        # Intermediate results
        write_print(f"Time: {time.asctime()}", outfile)
        write_print(f"Results at bootstrap round {i+1}/{k}:", outfile)
        write_print("\tOverall scores:", outfile)
        for j in range(len(score_funcs)):
            write_print(f"\t\t{overall_scores[j][i]}", outfile)
        for col in dem_cols:
            write_print(f"\t{col}:", outfile)
            for clss in classes[col]:
                write_print(f"\t\t{clss}:", outfile)
                for j in range(len(score_funcs)):
                    write_print(f"\t\t\t{scores[j][col][clss][i]}", outfile)
        write_print("\n", outfile)
    
    # Initialize a report string
    report = "\nOverall model performance:\n"
    for j, score in enumerate(score_funcs):
        report += f"\tMean {str(score).split(' ')[1]}: "
        report += f"{np.mean(overall_scores[j])}\n"
        report += "\tTotal performance by round :"
        report += f"\n\t\t{overall_scores[j].flatten()}\n"
    report += "\n\n"
    
    if any(threshs):
        report += f"90% sensitivity thresholds: {saved_threshes}\n\n"
    for j, score in enumerate(score_funcs):
        report += f"Score: {str(score).split(' ')[1]}\n"
        for col in dem_cols:
            report += f"\tColumn: {col}\n"
            for clss in classes[col]:
                report += f"\t\tClass: {clss}\n"
                report += f"\t\t\t{scores[j][col][clss].flatten()}\n"
                report += f"\t\t\tLengths: {num_clss[col][clss].flatten()}\n"
                report += f"\t\t\tPos.vals.: {pos_clss[col][clss].flatten()}\n"
                report += "\n"
    
    # Perform statistical analyses on each demographic category
    for i, col in enumerate(dem_cols):
        # Report which demographic category this is
        report += f"\nPerformance for demographic category {col}:\n"
        # Perform statistical analyses for each score
        for j, score in enumerate(score_funcs):
            # Report which metric this is
            report += f"\tMean {str(score).split(' ')[1]} values:\n"
            # Report performance for each class
            for clss in classes[col]:
                report += f"\t\t{clss}: {np.mean(scores[j][col][clss])}\n"
                mean_len = np.mean(num_clss[col][clss])
                report += f"\t\t\tMean length: {mean_len}\n"
                mean_pos = np.mean(pos_clss[col][clss])
                report += f"\t\t\tMean number of positives: {mean_pos}\n"
                report += f"\t\t\tMean pos. proportion: {mean_pos/mean_len}\n"
                report += "\n"
            # Are there more than two classes?
            if len(classes[col]) > 2:
                # Identify the majoritized class
                m_clss = majority_classes[i]
                report += f"\tMajoritized class: {m_clss}\n"
                # Get the majoritized scores
                scores_maj = scores[j][col][m_clss].flatten()
                # Iterate over each minority class
                for clss in [c for c in classes[col] if c != m_clss]:
                    scores_min = scores[j][col][clss].flatten()
                    # Get the p-value
                    w_res = wilcoxon(scores_maj - scores_min)
                    p_val = w_res.pvalue
                    # Report the p-value
                    report += f"\tWilcoxon signed-rank p-value {m_clss} v. "
                    report += str(clss)
                    report += f": {p_val}\n"
            elif len(classes[col]) == 2:
                # Wilcoxon signed-rank test
                class_list = [clss for clss in classes[col]]
                scores_1 = scores[j][col][class_list[0]].flatten()
                scores_2 = scores[j][col][class_list[1]].flatten()
                w_res = wilcoxon(scores_1 - scores_2)
                p_val = w_res.pvalue
                # Report the p-value
                report += f"\tWilcoxon signed-rank p-value: {p_val}\n"
            else:
                raise AttributeError(f"Must have 2+ classes for {col}")
            report += "\n\n"
        report += "\n"
    
    # Print and save the report
    write_print(report, outfile)
    write_print(time.asctime(), outfile)
    
    # Create an array of disparities
    n_cols = len(score_funcs)*sum([len(classes[col])-1 for col in dem_cols])
    disp_array = np.zeros((k,n_cols))
    # Current column tracker
    col_i = 0
    # Track disparity column names
    disp_cols = []
    # Iterate through each score function
    for j, score in enumerate(score_funcs):
        # Iterate through each demographic_category
        for i, col in enumerate(dem_cols):
            # Get the "majoritized" class and scores
            m_clss = majority_classes[i]
            scores_maj = scores[j][col][m_clss].flatten()
            # Iterate over each minoritized class
            for l, clss in enumerate([c for c in classes[col] if c != m_clss]):
                scores_min = scores[j][col][clss].flatten()
                # Fill in the values of the disparities array
                disp_array[:,col_i] = scores_maj - scores_min
                # Update the column tracker
                col_i += 1
                # Update the column names
                disp_cols.append(f"{col}_{clss}_{str(score).split(' ')[1]}")
    
    # Save the disparities array
    disp_df = pd.DataFrame(disp_array, columns=disp_cols)
    disp_df.to_csv(f"disparities_{name}_{date_str}_{rn}.csv", index=False)
    
    # Create an array of raw model performances
    n_cols_raw = len(score_funcs)*sum([len(classes[col]) for col in dem_cols])
    perf_array = np.zeros((k,n_cols_raw))
    # Current column tracker
    col_i = 0
    # Track performance column names
    perf_cols = []
    # Iterate through each score function
    for j, score in enumerate(score_funcs):
        # Iterate through each demographic_category
        for i, col in enumerate(dem_cols):
            # Iterate over each class
            for l, clss in enumerate(classes[col]):
                scores_min = scores[j][col][clss].flatten()
                # Fill in the values of the disparities array
                perf_array[:,col_i] = scores[j][col][clss].flatten()
                # Update the column tracker
                col_i += 1
                # Update the column names
                perf_cols.append(f"{col}_{clss}_{str(score).split(' ')[1]}")
    
    # Save the performance array
    perf_df = pd.DataFrame(perf_array, columns=perf_cols)
    perf_df.to_csv(f"performance_{name}_{date_str}_{rn}.csv", index=False)
    


# Disparities evaluation


def disparity_comparison(old_disp, new_disp, alpha=0.05):
    """Given a set of initial disparities and corresponding new 
    disparities, determines using Wilcoxon signed-rank testing which 
    if any of the new disparities were significantly reduced.
    The 'old_disp' and 'new_disp' arguments must be ndarrays of 
    equal shape, with columns corresponding to the exact same 
    disparity class. The 'alpha' argument is the p-value below which a 
    difference is considered statistically significant.
    Returns an array of p-values for whether the disparities were 
    significantly different and an array of booleans for whether the 
    disparities were significantly reduced."""
    
    # Make sure the inputs have the same number of columns
    assert old_disp.shape[1] == new_disp.shape[1]
    n_cols = old_disp.shape[1]
    
    # Initiate lists for holding p-values and reduction booleans
    p_vals = np.zeros(n_cols)
    reduced = np.full(n_cols, False)
    
    # Iterate through each column
    for j in range(n_cols):
        # Perform the statistical test
        w_res = wilcoxon(old_disp[:,j] - new_disp[:,j])
        p_val = w_res.pvalue
        # Save the p-value
        p_vals[j] = p_val
        # Was the difference significant?
        if p_val < alpha:
            # Was the disparity reduced?
            if np.mean(new_disp[:,j]) < np.mean(old_disp[:,j]):
                # Record that the disparity was significantly reduced
                reduced[j] = True
    
    # Return the results
    return p_vals, reduced


def sdoh_disparities(name, X_in, y_in, dem_cols=["GENDER", "RACE"], 
                     sdoh=["RACE"], majority_classes=["M", 
                                                "White or Caucasian"], 
                     score_funcs=[auprc, ppv_fixed_sens, 
                                  recall_fixed_thresh], 
                     probas=[True, True, True], 
                     threshs=[False, False, True], 
                     fix_thresh_on_val=True, k=10, train_pos_prop=0.5, 
                     val_pos_prop=0.05, train_split_prop=0.7, 
                     pos_vals=[1], id_split=True, id_col="CSN", 
                     out_col="end_point_6", original_size=True, 
                     n_trials=30, outfile="sdoh_disparities.txt"):
    """Given the name of an sklearn-compatible model 'name', DataFrame 
    'X_in', and outcomes Series 'y_in', returns a DataFrame of 
    differences in model performance based on the columns 'dem_cols' 
    in 'X'. The returned DataFrame has columns of the form: 
        "{demographic_category}_{class}_{score}"
    where the row values are the difference in performance between the 
    majoritized class in 'majority_classes' for 'demographic_category' 
    and 'class' in terms of 'score'. Thus, the column 'GENDER_F_auprc' 
    would have values of the form "{AUPRC for M} - {AUPRC for F}". 
    Also returns a DataFrame of raw model performance by columns of 
    'dem_cols'.
    The model is not trained on any data in 'dem_cols' EXCEPT for the 
    features in the array-like 'sdoh'.
    Hyperparameter optimizer optimizes for the first entry of 
    'score_funcs'.
    The 'outfile' argument is the filename of a TXT file to hold 
    intermediate and final results.
    The remaining arguments are as given in 'disparities_analysis()'.
    Wilcoxon signed-rank tests are performed to detect the existence 
    of disparities.
    Saves intermediate and final results, including the existence of 
    disparities, to a file 'outfile'.
    """
    
    # Track the time
    write_print(f"Starting SDoH assessment: {time.asctime()}", outfile)
    
    # Get copies of X_in and y_in
    X = X_in.copy()
    y = y_in.copy()
    
    # Which classes are there for each demographic category?
    classes = {col:X[col].unique() for col in dem_cols}
    
    # Initialize arrays to hold scores
    num_scores = len(score_funcs)
    overall_scores = [np.zeros(k) for _ in score_funcs]
    scores = [{col:{clss:np.zeros((k,1)) for clss in classes[col]} \
                                                         for col in dem_cols} \
                                                          for _ in score_funcs]
    # Thresholds, if applicable
    if any(threshs):
        saved_threshes = []
    # Array to hold number in each class in each bootstrap sample
    num_clss = {col:{clss:np.zeros((k,1)) for clss in classes[col]} \
                                                           for col in dem_cols}
    pos_clss = {col:{clss:np.zeros((k,1)) for clss in classes[col]} \
                                                           for col in dem_cols}
    
    # One-hot encode the SDoH column
    prefix_sdoh = sdoh.copy()
    if type(sdoh) != str and "SVI_QUARTILE" in sdoh:
        # Remove an already categorical column prefix if it exists
        prefix_sdoh.remove("SVI_QUARTILE")
    one_hot = pd.get_dummies(X[sdoh], prefix=prefix_sdoh, dtype=int)
    # Handle non-prefixed columns
    one_hot.rename(columns={"SVI_QUARTILE":"SVI_QUARTILE_all"}, inplace=True)
    ########## DEBUGGING ##########
    print(one_hot.columns)
    print(one_hot)
    ########## DEBUGGING ##########
    X = X.join(one_hot)
    
    # Bootstrap the model k times
    for i in range(k):
        # Get the bootstrap sample
        X_train, y_train, X_val, y_val = bootstrap(X, y, train_pos_prop, 
                                                   val_pos_prop, 
                                                   train_split_prop, pos_vals, 
                                                   id_split, id_col, 
                                                   out_col, original_size)
        # Drop the demographic category columns from the training data
        X_train.drop(columns=dem_cols, inplace=True)
        # Make a copy of the validation data without demographic categories
        X_val_clean = X_val.drop(columns=dem_cols)
        # Get the best hyperparameters and model input features
        cols, best_params = model_optimizer(name, X_train, y_train, 
                                            score_funcs[0], probas[0], 
                                            n_trials, [1], id_split, 
                                            id_col, out_col)
        # Get the right columns based on whether IDs are kept together
        if id_split:
            X_train = X_train.drop(columns=id_col)
            y_train = y_train.loc[:,out_col]
            X_val = X_val.drop(columns=id_col)
            X_val_clean = X_val_clean.drop(columns=id_col)
            y_val = y_val.loc[:,out_col]
        # Instantiate the model
        model = new_classifier(name, best_params)
        # Train the model
        model.fit(X_train, y_train)
        print(X_train.columns) ####
        # Find a 90% sensitivity threshold for the model
        if any(threshs):
            if fix_thresh_on_val:
                probs_pred = model.predict_proba(X_val_clean)[:,1]
                sens_thresh = get_sens_threshold(y_val, probs_pred)
            else:
                probs_pred = model.predict_proba(X_train)[:,1]
                sens_thresh = get_sens_threshold(y_train, probs_pred)
            saved_threshes.append(sens_thresh)
        # Validate the model over each score function
        for j, score in enumerate(score_funcs):
            # Validate the model
            if probas[j]:
                prediction = model.predict_proba(X_val_clean)[:,1]
            else:
                prediction = model.predict(X_val_clean)
            # Get and save the overall model performance
            # Check if more arguments need to be passed
            if threshs[j]:
                overall_scores[j][i] = score(y_val, prediction, sens_thresh)
            else:
                overall_scores[j][i] = score(y_val, prediction)
            # Iterate through each demographic category to be assessed
            for col in dem_cols:
                # Iterate through each class in the demographic category
                for clss in classes[col]:
                    # Get the data for the class
                    X_clss = X_val.where(X_val[col]==clss).dropna(subset=[col])
                    X_clss.drop(columns=dem_cols, inplace=True)
                    y_clss = y_val.loc[X_clss.index]
                    # Set the proportion of positive samples in the subset
                    X_clss, y_clss = reproportion_pos(X_clss, y_clss, 
                                                      val_pos_prop)
                    # Tracking info
                    num_clss[col][clss][i] = len(y_clss)
                    pos_clss[col][clss][i] = sum(y_clss)
                    # Get the model predictions for the class
                    if probas[j]:
                        prediction_clss = model.predict_proba(X_clss)[:,1]
                    else:
                        prediction_clss = model.predict(X_clss)
                    # Get and save the model performance for this class
                    # Check if more arguments need to be passed
                    if threshs[j]:
                        scores[j][col][clss][i] = score(y_clss, 
                                                        prediction_clss, 
                                                        sens_thresh)
                    else:
                        scores[j][col][clss][i] = score(y_clss, 
                                                        prediction_clss)
        # Intermediate results
        write_print(f"Results at bootstrap round {i+1}/{k}:", outfile)
        write_print("\tOverall scores:", outfile)
        for j in range(len(score_funcs)):
            write_print(f"\t\t{overall_scores[j][i]}", outfile)
        for col in dem_cols:
            write_print(f"\t{col}:", outfile)
            for clss in classes[col]:
                write_print(f"\t\t{clss}:", outfile)
                for j in range(len(score_funcs)):
                    write_print(f"\t\t\t{scores[j][col][clss][i]}", outfile)
        write_print("\n", outfile)
    
    # Initialize a report string
    report = f"\nOverall model performance with SDoH {sdoh}:\n"
    for j, score in enumerate(score_funcs):
        report += f"\tMean {str(score).split(' ')[1]}: "
        report += f"{np.mean(overall_scores[j])}\n"
        report += "\tTotal performance by round :"
        report += f"\n\t\t{overall_scores[j].flatten()}\n"
    report += "\n\n"
    
    if any(threshs):
        report += f"90% sensitivity thresholds: {saved_threshes}\n\n"
    for j, score in enumerate(score_funcs):
        report += f"Score: {str(score).split(' ')[1]}\n"
        for i, col in enumerate(dem_cols):
            report += f"\tColumn: {col}\n"
            for clss in classes[col]:
                report += f"\t\tClass: {clss}\n"
                report += f"\t\t\tMean: {np.mean(scores[j][col][clss])}\n"
                report += f"\t\t\t{scores[j][col][clss].flatten()}\n"
                report += f"\t\t\tLengths: {num_clss[col][clss].flatten()}\n"
                mean_len = np.mean(num_clss[col][clss])
                report += f"\t\t\tMean length: {mean_len}\n"
                report += f"\t\t\tPos.vals.: {pos_clss[col][clss].flatten()}\n"
                mean_pos = np.mean(pos_clss[col][clss])
                report += f"\t\t\tMean number of positives: {mean_pos}\n"
                report += f"\t\t\tMean pos. proportion: {mean_pos/mean_len}\n"
                # Is there still a disparity here?
                if clss != majority_classes[i]:
                    # Identify the majoritized class
                    m_clss = majority_classes[i]
                    # Get the respective scores
                    scores_maj = scores[j][col][m_clss].flatten()
                    scores_min = scores[j][col][clss].flatten()
                    # Get the p-value
                    diffs = scores_maj - scores_min
                    w_res = wilcoxon(diffs)
                    p_val = w_res.pvalue
                    # Report the p-value
                    report += f"\tWilcoxon signed-rank p-value {m_clss} v. "
                    report += str(clss)
                    report += f": {p_val}\n"
                    # Report the disparity/difference
                    report += f"\tMean difference: {np.mean(diffs)}\n"
                    report += f"\tAbsolute differences: {diffs}\n"
                report += "\n\n"
    
    # Print the report
    write_print(report, outfile)
    
    # Create an array of disparities
    n_cols = len(score_funcs)*sum([len(classes[col])-1 for col in dem_cols])
    disp_array = np.zeros((k,n_cols))
    # Current column tracker
    col_i = 0
    # Track disparity column names
    disp_cols = []
    # Iterate through each score function
    for j, score in enumerate(score_funcs):
        # Iterate through each demographic_category
        for i, col in enumerate(dem_cols):
            # Get the "majoritized" class and scores
            m_clss = majority_classes[i]
            scores_maj = scores[j][col][m_clss].flatten()
            # Iterate over each minoritized class
            for l, clss in enumerate([c for c in classes[col] if c != m_clss]):
                scores_min = scores[j][col][clss].flatten()
                # Fill in the values of the disparities array
                disp_array[:,col_i] = scores_maj - scores_min
                # Update the column tracker
                col_i += 1
                # Update the column names
                disp_cols.append(f"{col}_{clss}_{str(score).split(' ')[1]}")
    
    # Create an array of raw model performances
    n_cols_raw = len(score_funcs)*sum([len(classes[col]) for col in dem_cols])
    perf_array = np.zeros((k,n_cols_raw))
    # Current column tracker
    col_i = 0
    # Track performance column names
    perf_cols = []
    # Iterate through each score function
    for j, score in enumerate(score_funcs):
        # Iterate through each demographic_category
        for i, col in enumerate(dem_cols):
            # Iterate over each class
            for l, clss in enumerate(classes[col]):
                scores_min = scores[j][col][clss].flatten()
                # Fill in the values of the disparities array
                perf_array[:,col_i] = scores[j][col][clss].flatten()
                # Update the column tracker
                col_i += 1
                # Update the column names
                perf_cols.append(f"{col}_{clss}_{str(score).split(' ')[1]}")
    
    # Return the DataFrames
    disp_df = pd.DataFrame(disp_array, columns=disp_cols)
    perf_df = pd.DataFrame(perf_array, columns=perf_cols)
    write_print(time.asctime(), outfile)
    return disp_df, perf_df


def resamp_disparity(name, X_in, y_in, dem_cols=["GENDER", "RACE"], 
                     sdoh="RACE", over_under=0, 
                     majority_classes=["M", "White or Caucasian"], 
                     score_funcs=[auprc, ppv_fixed_sens, 
                                  recall_fixed_thresh], 
                     probas=[True, True, True], 
                     threshs=[False, False, True], 
                     fix_thresh_on_val=True, k=10, val_pos_prop=0.05, 
                     train_split_prop=0.7, pos_vals=[1], 
                     id_split=True, id_col="CSN", 
                     out_col="end_point_6", original_size=True, 
                     n_trials=30, outfile="sdoh_disparities.txt"):
    """Given the name of an sklearn-compatible model 'name', DataFrame 
    'X_in', and outcomes Series 'y_in', returns a DataFrame of 
    differences in model performance based on the columns 'dem_cols' 
    in 'X'. The returned DataFrame has columns of the form: 
        "{demographic_category}_{class}_{score}"
    where the row values are the difference in performance between the 
    majoritized class in 'majority_classes' for 'demographic_category' 
    and 'class' in terms of 'score'. Thus, the column 'GENDER_F_auprc' 
    would have values of the form "{AUPRC for M} - {AUPRC for F}". 
    Also returns a DataFrame of raw model performance by columns of 
    'dem_cols'.
    The model is trained on data resampled based on the column 'sdoh' 
    by a combination of SMOTE and random undersampling, or potentially 
    just one or the other of SMOTE or random undersampling based on 
    whether 'over_under' is equal to -1 (under), 1 (over), or 0 (both).
    If 'sdoh' is a list instead of a string, then the data are 
    resampled based on all of the columns in 'sdoh'. The model is not 
    trained on any data in 'dem_cols'.
    Hyperparameter optimizer optimizes for the first entry of 
    'score_funcs'.
    REQUIRES THE 'train_pos_prop' ARGUMENT FROM OTHER FUNCTIONS TO BE 
    EQUAL TO 0.5 AND (except for initial train/val bootstrap) 
    'id_split' TO BE FALSE!
    The 'outfile' argument is the filename of a TXT file to hold 
    intermediate and final results.
    The remaining arguments are as given in 'disparities_analysis()'.
    Wilcoxon signed-rank tests are performed to detect the existence 
    of disparities.
    Saves intermediate and final results, including the existence of 
    disparities, to a file 'outfile'.
    """
    
    # Track the time
    write_print(f"Starting resampling assessment: {time.asctime()}", outfile)
    
    # Get copies of X_in and y_in
    X = X_in.copy()
    y = y_in.copy()
    
    # Which classes are there for each demographic (and sdoh) category?
    classes = {col:X[col].unique() for col in dem_cols}
    if type(sdoh) != str:
        sdoh_classes = {col:X[col].unique() for col in sdoh}
    
    # Initialize arrays to hold scores
    num_scores = len(score_funcs)
    overall_scores = [np.zeros(k) for _ in score_funcs]
    scores = [{col:{clss:np.zeros((k,1)) for clss in classes[col]} \
                                                         for col in dem_cols} \
                                                          for _ in score_funcs]
    # Thresholds, if applicable
    if any(threshs):
        saved_threshes = []
    # Array to hold number in each class in each bootstrap sample
    num_clss = {col:{clss:np.zeros((k,1)) for clss in classes[col]} \
                                                           for col in dem_cols}
    pos_clss = {col:{clss:np.zeros((k,1)) for clss in classes[col]} \
                                                           for col in dem_cols}
    if type(sdoh) != str:
        num_sdoh_pre = {col:{clss:np.zeros(k) for clss in sdoh_classes[col]} \
                                                               for col in sdoh}
        pos_sdoh_pre = {col:{clss:np.zeros(k) for clss in sdoh_classes[col]} \
                                                               for col in sdoh}
        num_sdoh_post = {col:{clss:np.zeros(k) for clss in sdoh_classes[col]} \
                                                               for col in sdoh}
        pos_sdoh_post = {col:{clss:np.zeros(k) for clss in sdoh_classes[col]} \
                                                               for col in sdoh}
    
    # Bootstrap the model k times
    for i in range(k):
        # Get the bootstrap sample
        X_train, y_train, X_val, y_val = bootstrap(X, y, 0.5, val_pos_prop, 
                                                   train_split_prop, pos_vals, 
                                                   id_split, id_col, 
                                                   out_col, original_size)
        
        # Reindex X_train and y_train to avoid duplicate index problems
        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        # Combine SDoH class with outcome for composite resample class
        if type(sdoh) == str:
            resamp_y = X_train[sdoh].astype(str) + y_train[out_col].astype(str)
        else:
            resamp_y = y_train[out_col].astype(str)
            for col in sdoh:
                resamp_y += X_train[col].astype(str)
        resamp_y.rename(out_col, inplace=True)
        # Drop the demographic category columns from the training data
        X_train.drop(columns=dem_cols, inplace=True)
        # Make a copy of the validation data without demographic categories
        X_val_clean = X_val.drop(columns=dem_cols)
        # Drop ID columns since those won't work with SMOTE
        if id_split:
            X_train = X_train.drop(columns=id_col)
            y_train = y_train.loc[:,out_col]
            X_val = X_val.drop(columns=id_col)
            X_val_clean = X_val_clean.drop(columns=id_col)
            y_val = y_val.loc[:,out_col]
        
        # Resample the training data, keeping outcomes at the right proportion
        # How many should each class have?
        n_per_class = round(len(resamp_y)/len(resamp_y.unique()))
        # How many are in each composite class?
        resamp_counts = resamp_y.value_counts()
        resamp_prop = resamp_y.value_counts(normalize=True)
        # How many are in each individual class, if applicable
        if type(sdoh) != str:
            for col in sdoh:
                for clss in sdoh_classes[col]:
                    y_sdoh_class = resamp_y[resamp_y.str.contains(str(clss))]
                    num_sdoh_pre[col][clss][i] = len(y_sdoh_class)
                    pos_sdoh_pre[col][clss][i] = len(y_sdoh_class) - \
                              len(y_sdoh_class[y_sdoh_class.str.contains("0")])
        
        # Add copies of patients below the KNN representation threshold
        for clss in resamp_counts.index:
            if resamp_counts[clss] < 6: # Default k neighbors
                n_needed = 6 - resamp_counts[clss]
                add_idxs = resamp_y.where(resamp_y == clss).dropna().index
                # Duplicate random patients if fewer than 3 are needed
                if n_needed < 3:
                    add_idxs = np.random.choice(add_idxs, n_needed)
                # Duplicate all patients otherwise
                else:
                    n_copies = 6//resamp_counts[clss] - 1
                    add_idxs = list(add_idxs)*n_copies
                X_train = pd.concat([X_train, X_train.loc[add_idxs,:]], 
                                     ignore_index=True)
                resamp_y = pd.concat([resamp_y, resamp_y.loc[add_idxs]], 
                                     ignore_index=True)
        # Divide classes into oversample versus undersample
        # Undersample
        under_samp = resamp_counts > n_per_class
        under_samp_vals = under_samp.where(under_samp).dropna().index
        under_bool = resamp_y.isin(under_samp_vals)
        resamp_y_u = resamp_y.where(under_bool).dropna()
        X_train_under = X_train.where(under_bool).dropna()
        # Oversample
        over_samp_vals = under_samp.where(~under_samp).dropna().index
        resamp_y_o = resamp_y.where(~under_bool).dropna()
        X_train_over = X_train.where(~under_bool).dropna()
        # Do the resampling
        if over_under != -1:
            # Oversample
            over_dict = {c:n_per_class for c in over_samp_vals}
            smt = SMOTE(sampling_strategy=over_dict)
            X_train_over, resamp_y_o = smt.fit_resample(X_train_over, 
                                                        resamp_y_o)
        if over_under != 1:
            # Undersample
            under_dict = {c:n_per_class for c in under_samp_vals}
            rus = RandomUnderSampler(sampling_strategy=under_dict)
            X_train_under, resamp_y_u = rus.fit_resample(X_train_under, 
                                                         resamp_y_u)
        # Recombine
        X_train = pd.concat([X_train_over, X_train_under], ignore_index=True)
        y_train = pd.concat([resamp_y_o, resamp_y_u], ignore_index=True)
        # Save class proportions for tracking
        new_counts = y_train.value_counts()
        new_counts_prop = y_train.value_counts(normalize=True)
        # Individual proportions, if applicable
        if type(sdoh) != str:
            for col in sdoh:
                for clss in sdoh_classes[col]:
                    print(y_train, end="\n\n")
                    print(clss, end="\n\n")
                    print(y_train.str.contains(str(clss)), end="\n\n")
                    print(y_train[y_train.str.contains(str(clss))], 
                          end="\n\n")
                    y_sdoh_class = y_train[y_train.str.contains(str(clss))]
                    num_sdoh_post[col][clss][i] = len(y_sdoh_class)
                    pos_sdoh_post[col][clss][i] = len(y_sdoh_class) - \
                              len(y_sdoh_class[y_sdoh_class.str.contains("0")])
        
        # Extract outcome from combination SDoH class/outcome
        if type(sdoh) == str:
            # Binary outcome at the end
            y_train = y_train.apply(lambda x: int(x[-1])).astype(int)
        else:
            # Binary outcome at the beginning
            y_train = y_train.apply(lambda x: int(x[0])).astype(int)
        
        # Shuffle to avoid training on order
        X_train, y_train = shuffle(X_train,y_train)
        
        # Get the best hyperparameters and model input features
        cols, best_params = model_optimizer(name, X_train, y_train, 
                                            score_funcs[0], probas[0], 
                                            n_trials, [1], False, 
                                            id_col, out_col)
        
        # Instantiate the model
        model = new_classifier(name, best_params)
        # Train the model
        model.fit(X_train, y_train)
        # Find a 90% sensitivity threshold for the model
        if any(threshs):
            if fix_thresh_on_val:
                probs_pred = model.predict_proba(X_val_clean)[:,1]
                sens_thresh = get_sens_threshold(y_val, probs_pred)
            else:
                probs_pred = model.predict_proba(X_train)[:,1]
                sens_thresh = get_sens_threshold(y_train, probs_pred)
            saved_threshes.append(sens_thresh)
        # Validate the model over each score function
        for j, score in enumerate(score_funcs):
            # Validate the model
            if probas[j]:
                prediction = model.predict_proba(X_val_clean)[:,1]
            else:
                prediction = model.predict(X_val_clean)
            # Get and save the overall model performance
            # Check if more arguments need to be passed
            if threshs[j]:
                overall_scores[j][i] = score(y_val, prediction, sens_thresh)
            else:
                overall_scores[j][i] = score(y_val, prediction)
            # Iterate through each demographic category to be assessed
            for col in dem_cols:
                # Iterate through each class in the demographic category
                for clss in classes[col]:
                    # Get the data for the class
                    X_clss = X_val.where(X_val[col]==clss).dropna(subset=[col])
                    X_clss.drop(columns=dem_cols, inplace=True)
                    y_clss = y_val.loc[X_clss.index]
                    # Set the proportion of positive samples in the subset
                    X_clss, y_clss = reproportion_pos(X_clss, y_clss, 
                                                      val_pos_prop)
                    # Tracking info
                    num_clss[col][clss][i] = len(y_clss)
                    pos_clss[col][clss][i] = sum(y_clss)
                    # Get the model predictions for the class
                    if probas[j]:
                        prediction_clss = model.predict_proba(X_clss)[:,1]
                    else:
                        prediction_clss = model.predict(X_clss)
                    # Get and save the model performance for this class
                    # Check if more arguments need to be passed
                    if threshs[j]:
                        scores[j][col][clss][i] = score(y_clss, 
                                                        prediction_clss, 
                                                        sens_thresh)
                    else:
                        scores[j][col][clss][i] = score(y_clss, 
                                                        prediction_clss)
        # Intermediate results
        write_print(f"Results at bootstrap round {i+1}/{k}:", outfile)
        write_print("\tOverall scores:", outfile)
        for j in range(len(score_funcs)):
            write_print(f"\t\t{overall_scores[j][i]}", outfile)
        for col in dem_cols:
            write_print(f"\t{col}:", outfile)
            for clss in classes[col]:
                write_print(f"\t\t{clss}:", outfile)
                for j in range(len(score_funcs)):
                    write_print(f"\t\t\t{scores[j][col][clss][i]}", outfile)
        # Counts of just one resampled category
        if type(sdoh) == str:
            write_print(f"\tCounts for each {sdoh} class:", outfile)
            write_print(f"\t\t\tBefore resampling\t\tAfter resampling", 
                        outfile)
            for clss in classes[sdoh]:
                # Negative counts for before and after
                # Check that at least one member exists in each category
                try:
                    b_c = resamp_counts[str(clss)+"0"]
                except KeyError:
                    b_c = 0
                try:
                    b_p = resamp_prop[str(clss)+"0"]
                except KeyError:
                    b_p = 0
                try:
                    a_c = new_counts[str(clss)+"0"]
                except KeyError:
                    a_c = 0
                try:
                    a_p = new_counts_prop[str(clss)+"0"]
                except KeyError:
                    a_p = 0
                write_print(f"\t\t{clss} 0: {b_c} {b_p}\t\t{a_c} {a_p}", 
                            outfile)
                # Positive counts for before and after
                # Check that at least one member exists in each category
                try:
                    b_c = resamp_counts[str(clss)+"1"]
                except KeyError:
                    b_c = 0
                try:
                    b_p = resamp_prop[str(clss)+"1"]
                except KeyError:
                    b_p = 0
                try:
                    a_c = new_counts[str(clss)+"1"]
                except KeyError:
                    a_c = 0
                try:
                    a_p = new_counts_prop[str(clss)+"1"]
                except KeyError:
                    a_p = 0
                write_print(f"\t\t{clss} 1: {b_c} {b_p}\t\t{a_c} {a_p}", 
                            outfile)
        # Counts of many resampled categories
        else:
            write_print(f"\tCategories: {', '.join(sdoh)}", outfile)
            write_print("\tCounts for each combination class:", outfile)
            write_print(f"\t\t\tBefore resampling\t\tAfter resampling", 
                        outfile)
            for clss in resamp_counts.index:
                b_c = resamp_counts[clss]
                b_p = resamp_prop[clss]
                a_c = new_counts[clss]
                a_p = new_counts_prop[clss]
                write_print(f"\t\t{clss}: {b_c} ({round(b_p,4)})", outfile, 
                            end="")
                write_print(f"\t\t{a_c} ({round(a_p,4)})", outfile)
            write_print("\n\tCounts for each individual class:", outfile)
            write_print(f"\t\t\tBefore resampling\t\tAfter resampling", 
                        outfile)
            for col in sdoh:
                for clss in sdoh_classes[col]:
                    b_c = num_sdoh_pre[col][clss][i]
                    a_c = num_sdoh_post[col][clss][i]
                    write_print(f"\t\t{clss} count: {b_c}\t\t{a_c}", outfile)
                    b_p = pos_sdoh_pre[col][clss][i]
                    a_p = pos_sdoh_post[col][clss][i]
                    write_print(f"\t\t\t positive: {b_p}\t\t{a_p}", outfile)
        write_print("\n", outfile)
    
    # Initialize a report string
    report = "\nOverall model performance with resampled "
    if type(sdoh) == str:
        report += f"{sdoh}:\n"
    else:
        report += f"{', '.join(sdoh)}:\n"
    for j, score in enumerate(score_funcs):
        report += f"\tMean {str(score).split(' ')[1]}: "
        report += f"{np.mean(overall_scores[j])}\n"
        report += "\tTotal performance by round :"
        report += f"\n\t\t{overall_scores[j].flatten()}\n"
    report += "\n\n"
    
    if any(threshs):
        report += f"90% sensitivity thresholds: {saved_threshes}\n\n"
    for j, score in enumerate(score_funcs):
        report += f"Score: {str(score).split(' ')[1]}\n"
        for i, col in enumerate(dem_cols):
            report += f"\tColumn: {col}\n"
            for clss in classes[col]:
                report += f"\t\tClass: {clss}\n"
                report += f"\t\t\tMean: {np.mean(scores[j][col][clss])}\n"
                report += f"\t\t\t{scores[j][col][clss].flatten()}\n"
                report += f"\t\t\tLengths: {num_clss[col][clss].flatten()}\n"
                mean_len = np.mean(num_clss[col][clss])
                report += f"\t\t\tMean length: {mean_len}\n"
                report += f"\t\t\tPos.vals.: {pos_clss[col][clss].flatten()}\n"
                mean_pos = np.mean(pos_clss[col][clss])
                report += f"\t\t\tMean number of positives: {mean_pos}\n"
                report += f"\t\t\tMean pos. proportion: {mean_pos/mean_len}\n"
                # Is there still a disparity here?
                if clss != majority_classes[i]:
                    # Identify the majoritized class
                    m_clss = majority_classes[i]
                    # Get the respective scores
                    scores_maj = scores[j][col][m_clss].flatten()
                    scores_min = scores[j][col][clss].flatten()
                    # Get the p-value
                    diffs = scores_maj - scores_min
                    w_res = wilcoxon(diffs)
                    p_val = w_res.pvalue
                    # Report the p-value
                    report += f"\tWilcoxon signed-rank p-value {m_clss} v. "
                    report += str(clss)
                    report += f": {p_val}\n"
                    # Report the disparity/difference
                    report += f"\tMean difference: {np.mean(diffs)}\n"
                    report += f"\tAbsolute differences: {diffs}\n"
                report += "\n\n"
    
    # Print the report
    write_print(report, outfile)
    
    # Create an array of disparities
    n_cols = len(score_funcs)*sum([len(classes[col])-1 for col in dem_cols])
    disp_array = np.zeros((k,n_cols))
    # Current column tracker
    col_i = 0
    # Track disparity column names
    disp_cols = []
    # Iterate through each score function
    for j, score in enumerate(score_funcs):
        # Iterate through each demographic_category
        for i, col in enumerate(dem_cols):
            # Get the "majoritized" class and scores
            m_clss = majority_classes[i]
            scores_maj = scores[j][col][m_clss].flatten()
            # Iterate over each minoritized class
            for l, clss in enumerate([c for c in classes[col] if c != m_clss]):
                scores_min = scores[j][col][clss].flatten()
                # Fill in the values of the disparities array
                disp_array[:,col_i] = scores_maj - scores_min
                # Update the column tracker
                col_i += 1
                # Update the column names
                disp_cols.append(f"{col}_{clss}_{str(score).split(' ')[1]}")
    
    # Create an array of raw model performances
    n_cols_raw = len(score_funcs)*sum([len(classes[col]) for col in dem_cols])
    perf_array = np.zeros((k,n_cols_raw))
    # Current column tracker
    col_i = 0
    # Track performance column names
    perf_cols = []
    # Iterate through each score function
    for j, score in enumerate(score_funcs):
        # Iterate through each demographic_category
        for i, col in enumerate(dem_cols):
            # Iterate over each class
            for l, clss in enumerate(classes[col]):
                scores_min = scores[j][col][clss].flatten()
                # Fill in the values of the disparities array
                perf_array[:,col_i] = scores[j][col][clss].flatten()
                # Update the column tracker
                col_i += 1
                # Update the column names
                perf_cols.append(f"{col}_{clss}_{str(score).split(' ')[1]}")
    
    # Return the DataFrames
    disp_df = pd.DataFrame(disp_array, columns=disp_cols)
    perf_df = pd.DataFrame(perf_array, columns=perf_cols)
    write_print(time.asctime(), outfile)
    return disp_df, perf_df


def MLSMOTE_disparity(name, X_in, y_in, dem_cols=["GENDER", "RACE"], 
                     sdoh_list=["RACE"], 
                     majority_classes=["M", "White or Caucasian"], 
                     score_funcs=[auprc, ppv_fixed_sens, 
                                  recall_fixed_thresh], 
                     probas=[True, True, True], 
                     threshs=[False, False, True], 
                     fix_thresh_on_val=True, k=10, val_pos_prop=0.05, 
                     train_split_prop=0.7, pos_vals=[1], 
                     id_split=True, id_col="CSN", 
                     out_col="end_point_6", original_size=True, 
                     n_trials=30, outfile="sdoh_disparities.txt"):
    """Given the name of an sklearn-compatible model 'name', DataFrame 
    'X_in', and outcomes Series 'y_in', returns a DataFrame of 
    differences in model performance based on the columns 'dem_cols' 
    in 'X'. The returned DataFrame has columns of the form: 
        "{demographic_category}_{class}_{score}"
    where the row values are the difference in performance between the 
    majoritized class in 'majority_classes' for 'demographic_category' 
    and 'class' in terms of 'score'. Thus, the column 'GENDER_F_auprc' 
    would have values of the form "{AUPRC for M} - {AUPRC for F}". 
    Also returns a DataFrame of raw model performance by columns of 
    'dem_cols'.
    The model is trained on data resampled based on the columns in 
    'sdoh_list' using MLSMOTE and random undersampling. The model is 
    not trained on any data in 'dem_cols' even if 'sdoh_list' and 
    'dem_cols' overlap.
    Hyperparameter optimizer optimizes for the first entry of 
    'score_funcs'.
    REQUIRES THE 'train_pos_prop' ARGUMENT FROM OTHER FUNCTIONS TO BE 
    EQUAL TO 0.5 AND (except for initial train/val bootstrap) 
    'id_split' TO BE FALSE!
    The 'outfile' argument is the filename of a TXT file to hold 
    intermediate and final results.
    The remaining arguments are as given in 'disparities_analysis()'.
    Wilcoxon signed-rank tests are performed to detect the existence 
    of disparities.
    Saves intermediate and final results, including the existence of 
    disparities, to a file 'outfile'.
    """
    
    # Track the time
    write_print(f"Starting resampling assessment: {time.asctime()}", outfile)
    # Get copies of X_in and y_in
    X = X_in.copy()
    y = y_in.copy()
    # Which classes are there for each demographic category?
    classes = {col:X[col].unique() for col in dem_cols}
    # Initialize arrays to hold scores
    num_scores = len(score_funcs)
    overall_scores = [np.zeros(k) for _ in score_funcs]
    scores = [{col:{clss:np.zeros((k,1)) for clss in classes[col]} \
                                                         for col in dem_cols} \
                                                          for _ in score_funcs]
    # Thresholds, if applicable
    if any(threshs):
        saved_threshes = []
    # Array to hold number in each class in each bootstrap sample
    num_clss = {col:{clss:np.zeros((k,1)) for clss in classes[col]} \
                                                           for col in dem_cols}
    pos_clss = {col:{clss:np.zeros((k,1)) for clss in classes[col]} \
                                                           for col in dem_cols}
    
    # Bootstrap the model k times
    for i in range(k):
        # Get the bootstrap sample
        X_train, y_train, X_val, y_val = bootstrap(X, y, 0.5, val_pos_prop, 
                                                   train_split_prop, pos_vals, 
                                                   id_split, id_col, 
                                                   out_col, original_size)
        # Reindex X_train and y_train to avoid duplicate index problems
        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        # Make a copy of the validation data without demographic categories
        X_val_clean = X_val.drop(columns=dem_cols)
        # Drop ID columns since those won't work with MLSMOTE
        if id_split:
            X_train = X_train.drop(columns=id_col)
            y_train = y_train.loc[:,out_col]
            X_val = X_val.drop(columns=id_col)
            X_val_clean = X_val_clean.drop(columns=id_col)
            y_val = y_val.loc[:,out_col]
        # Insert the outcomes into X to preserve them through MLSMOTE
        X_resamp = pd.concat([X_train, y_train], axis=1)
        # Create a binary label DataFrame for use with MLSMOTE
        labels = pd.get_dummies(X_resamp[sdoh_list], prefix=sdoh_list, 
                                columns=sdoh_list, dtype=int)
        # Save demographic label proportions for tracking
        pre_counts = labels.sum()
        pre_prop = labels.sum()/len(labels)
        ########## DEBUGGING ##########
        irlbls = IRLbl(labels.sum().values)
        write_print("Labels:", outfile)
        for nc, col in enumerate(labels.columns):
            n_in_col = labels[col].sum()
            write_print(f"\t{col}: {n_in_col}, {n_in_col/len(labels)}", 
                        outfile)
            write_print(f"\t\tIRLbl: {irlbls[nc]} vs. {np.mean(irlbls)}", 
                        outfile)
        ########## DEBUGGING ##########
        
        # Get the samples to upsample based on label imbalance ratio (IRLbl)
        X_train_ups, labels_ups = mlsmote.get_minority_instace(X_resamp, 
                                                               labels)
        # Get the samples to downsample
        ups_indices = mlsmote.get_index(labels)
        X_train_maj = X_resamp[~X_resamp.index.isin(ups_indices)]
        labels_maj = labels[~labels.index.isin(ups_indices)]
        # How many additional minority samples are necessary?
        n_upsamp = len(X_resamp)//2 - len(X_train_ups)
        ########## DEBUGGING ##########
        write_print(f"\nlen(X_train_maj): {len(X_train_maj)}", outfile)
        write_print(f"len(X_train_ups): {len(X_train_ups)}", outfile)
        ########## DEBUGGING ##########
        # Get the additional samples, temporarily one-hot encode categoricals
        cat_cols = dem_cols.copy()
        if "SVI_QUARTILE" in cat_cols:
            # Remove an already categorical column prefix if it exists
            cat_cols.remove("SVI_QUARTILE")
        X_train_ups = pd.get_dummies(X_train_ups, prefix=cat_cols, 
                                     columns=cat_cols, dtype=int)
        X_train_ups, labels_ups = mlsmote.MLSMOTE(X_train_ups, labels_ups, 
                                                  n_upsamp)
        dum_cols = [col for col in X_train_ups.columns if \
                                             any([c in col for c in cat_cols])]
        X_train_ups.drop(columns=dum_cols, inplace=True)
        # Downsample the majority samples
        X_train_maj = X_train_maj.sample(n=len(X_resamp)//2)
        labels_maj = labels.loc[X_train_maj.index,:]
        # Add the additional samples to the original training set
        X_train = pd.concat([X_train_maj, X_train_ups], ignore_index=True)
        labels = pd.concat([labels_maj, labels_ups], ignore_index=True)
        # Extract the outcomes from X now that MLSMOTE is over
        y_train = X_train[out_col].astype(int)
        X_train.drop(columns=out_col, inplace=True)
        # Drop the demographic category columns from the training data
        X_train.drop(columns=dem_cols, inplace=True)
        
        # Save demographic label proportions for tracking
        post_counts = labels.sum()
        post_prop = labels.sum()/len(labels)
        # Save outcome proportions for tracking
        y_out_count = {"total":y_train.value_counts()}
        y_out_prop = {"total":y_train.value_counts(normalize=True)}
        for col in labels.columns:
            y_col = y_train[labels[col] == 1]
            y_out_count[col] = y_col.value_counts()
            y_out_prop[col] = y_col.value_counts(normalize=True)
        
        # Shuffle to avoid training on order
        X_train, y_train = shuffle(X_train,y_train)
        
        # Get the best hyperparameters and model input features
        cols, best_params = model_optimizer(name, X_train, y_train, 
                                            score_funcs[0], probas[0], 
                                            n_trials, [1], False, 
                                            id_col, out_col)
        
        # Instantiate the model
        model = new_classifier(name, best_params)
        # Train the model
        model.fit(X_train, y_train)
        # Find a 90% sensitivity threshold for the model
        if any(threshs):
            if fix_thresh_on_val:
                probs_pred = model.predict_proba(X_val_clean)[:,1]
                sens_thresh = get_sens_threshold(y_val, probs_pred)
            else:
                probs_pred = model.predict_proba(X_train)[:,1]
                sens_thresh = get_sens_threshold(y_train, probs_pred)
            saved_threshes.append(sens_thresh)
        # Validate the model over each score function
        for j, score in enumerate(score_funcs):
            # Validate the model
            if probas[j]:
                prediction = model.predict_proba(X_val_clean)[:,1]
            else:
                prediction = model.predict(X_val_clean)
            # Get and save the overall model performance
            # Check if more arguments need to be passed
            if threshs[j]:
                overall_scores[j][i] = score(y_val, prediction, sens_thresh)
            else:
                overall_scores[j][i] = score(y_val, prediction)
            # Iterate through each demographic category to be assessed
            for col in dem_cols:
                # Iterate through each class in the demographic category
                for clss in classes[col]:
                    # Get the data for the class
                    X_clss = X_val.where(X_val[col]==clss).dropna(subset=[col])
                    X_clss.drop(columns=dem_cols, inplace=True)
                    y_clss = y_val.loc[X_clss.index]
                    # Set the proportion of positive samples in the subset
                    X_clss, y_clss = reproportion_pos(X_clss, y_clss, 
                                                      val_pos_prop)
                    # Tracking info
                    num_clss[col][clss][i] = len(y_clss)
                    pos_clss[col][clss][i] = sum(y_clss)
                    # Get the model predictions for the class
                    if probas[j]:
                        prediction_clss = model.predict_proba(X_clss)[:,1]
                    else:
                        prediction_clss = model.predict(X_clss)
                    # Get and save the model performance for this class
                    # Check if more arguments need to be passed
                    if threshs[j]:
                        scores[j][col][clss][i] = score(y_clss, 
                                                        prediction_clss, 
                                                        sens_thresh)
                    else:
                        scores[j][col][clss][i] = score(y_clss, 
                                                        prediction_clss)
        # Intermediate results
        write_print(f"\nResults at bootstrap round {i+1}/{k}:", outfile)
        write_print("\tOverall scores:", outfile)
        for j in range(len(score_funcs)):
            write_print(f"\t\t{overall_scores[j][i]}", outfile)
        for col in dem_cols:
            write_print(f"\t{col}:", outfile)
            for clss in classes[col]:
                write_print(f"\t\t{clss}:", outfile)
                for j in range(len(score_funcs)):
                    write_print(f"\t\t\t{scores[j][col][clss][i]}", outfile)
        # Counts of resampled categories
        write_print(f"\n\tCategories: {', '.join(sdoh_list)}", outfile)
        write_print("\tCounts for each class:", outfile)
        write_print(f"\t\t\tBefore resampling\t\tAfter resampling", outfile)
        for clss in pre_counts.index:
            b_c = pre_counts[clss]
            b_p = pre_prop[clss]
            a_c = post_counts[clss]
            a_p = post_prop[clss]
            write_print(f"\t\t{clss}: {b_c} ({round(b_p,4)})", outfile, end="")
            write_print(f"\t\t{a_c} ({round(a_p,4)})", outfile)
        # Outcome counts overall and by categories
        write_print("\n\tOutcomes:", outfile)
        write_print("\t\tOverall:\t", outfile, end="")
        write_print(f"1: {y_out_count['total'][1]} ", outfile, end="")
        write_print(f"({round(y_out_prop['total'][1],4)})\t", outfile, end="")
        write_print(f"0: {y_out_count['total'][0]} ", outfile, end="")
        write_print(f"({round(y_out_prop['total'][0],4)})", outfile)
        for col in labels.columns:
            write_print(f"\t\t{col}:\t", outfile, end="")
            write_print(f"1: {y_out_count[col][1]} ", outfile, end="")
            write_print(f"({y_out_prop[col][1]})\t", outfile, end="")
            write_print(f"0: {y_out_count[col][0]} ", outfile, end="")
            write_print(f"0: ({y_out_prop[col][0]})", outfile)
        write_print("\n", outfile)
    
    # Initialize a report string
    report = "\nOverall model performance with resampled "
    report += f"{', '.join(sdoh_list)}:\n"
    for j, score in enumerate(score_funcs):
        report += f"\tMean {str(score).split(' ')[1]}: "
        report += f"{np.mean(overall_scores[j])}\n"
        report += "\tTotal performance by round :"
        report += f"\n\t\t{overall_scores[j].flatten()}\n"
    report += "\n\n"
    
    if any(threshs):
        report += f"90% sensitivity thresholds: {saved_threshes}\n\n"
    for j, score in enumerate(score_funcs):
        report += f"Score: {str(score).split(' ')[1]}\n"
        for i, col in enumerate(dem_cols):
            report += f"\tColumn: {col}\n"
            for clss in classes[col]:
                report += f"\t\tClass: {clss}\n"
                report += f"\t\t\tMean: {np.mean(scores[j][col][clss])}\n"
                report += f"\t\t\t{scores[j][col][clss].flatten()}\n"
                report += f"\t\t\tLengths: {num_clss[col][clss].flatten()}\n"
                mean_len = np.mean(num_clss[col][clss])
                report += f"\t\t\tMean length: {mean_len}\n"
                report += f"\t\t\tPos.vals.: {pos_clss[col][clss].flatten()}\n"
                mean_pos = np.mean(pos_clss[col][clss])
                report += f"\t\t\tMean number of positives: {mean_pos}\n"
                report += f"\t\t\tMean pos. proportion: {mean_pos/mean_len}\n"
                # Is there still a disparity here?
                if clss != majority_classes[i]:
                    # Identify the majoritized class
                    m_clss = majority_classes[i]
                    # Get the respective scores
                    scores_maj = scores[j][col][m_clss].flatten()
                    scores_min = scores[j][col][clss].flatten()
                    # Get the p-value
                    diffs = scores_maj - scores_min
                    w_res = wilcoxon(diffs)
                    p_val = w_res.pvalue
                    # Report the p-value
                    report += f"\tWilcoxon signed-rank p-value {m_clss} v. "
                    report += str(clss)
                    report += f": {p_val}\n"
                    # Report the disparity/difference
                    report += f"\tMean difference: {np.mean(diffs)}\n"
                    report += f"\tAbsolute differences: {diffs}\n"
                report += "\n\n"
    
    # Print the report
    write_print(report, outfile)
    
    # Create an array of disparities
    n_cols = len(score_funcs)*sum([len(classes[col])-1 for col in dem_cols])
    disp_array = np.zeros((k,n_cols))
    # Current column tracker
    col_i = 0
    # Track disparity column names
    disp_cols = []
    # Iterate through each score function
    for j, score in enumerate(score_funcs):
        # Iterate through each demographic_category
        for i, col in enumerate(dem_cols):
            # Get the "majoritized" class and scores
            m_clss = majority_classes[i]
            scores_maj = scores[j][col][m_clss].flatten()
            # Iterate over each minoritized class
            for l, clss in enumerate([c for c in classes[col] if c != m_clss]):
                scores_min = scores[j][col][clss].flatten()
                # Fill in the values of the disparities array
                disp_array[:,col_i] = scores_maj - scores_min
                # Update the column tracker
                col_i += 1
                # Update the column names
                disp_cols.append(f"{col}_{clss}_{str(score).split(' ')[1]}")
    
    # Create an array of raw model performances
    n_cols_raw = len(score_funcs)*sum([len(classes[col]) for col in dem_cols])
    perf_array = np.zeros((k,n_cols_raw))
    # Current column tracker
    col_i = 0
    # Track performance column names
    perf_cols = []
    # Iterate through each score function
    for j, score in enumerate(score_funcs):
        # Iterate through each demographic_category
        for i, col in enumerate(dem_cols):
            # Iterate over each class
            for l, clss in enumerate(classes[col]):
                scores_min = scores[j][col][clss].flatten()
                # Fill in the values of the disparities array
                perf_array[:,col_i] = scores[j][col][clss].flatten()
                # Update the column tracker
                col_i += 1
                # Update the column names
                perf_cols.append(f"{col}_{clss}_{str(score).split(' ')[1]}")
    
    # Return the DataFrames
    disp_df = pd.DataFrame(disp_array, columns=disp_cols)
    perf_df = pd.DataFrame(perf_array, columns=perf_cols)
    write_print(time.asctime(), outfile)
    return disp_df, perf_df


def combo_disp_MLSMOTE(name, X_in, y_in, dem_cols=["GENDER", "RACE"], 
                       sdoh_list_rep=["RACE"], sdoh_list_inf=["RACE"], 
                       majority_classes=["M", "White or Caucasian"], 
                       score_funcs=[auprc, ppv_fixed_sens, 
                                    recall_fixed_thresh], 
                       probas=[True, True, True], 
                       threshs=[False, False, True], 
                       fix_thresh_on_val=True, k=10, val_pos_prop=0.05,
                       train_split_prop=0.7, pos_vals=[1], 
                       id_split=True, id_col="CSN", 
                       out_col="end_point_6", original_size=True, 
                       n_trials=30, outfile="sdoh_disparities.txt"):
    """Given the name of an sklearn-compatible model 'name', DataFrame 
    'X_in', and outcomes Series 'y_in', returns a DataFrame of 
    differences in model performance based on the columns 'dem_cols' 
    in 'X'. The returned DataFrame has columns of the form: 
        "{demographic_category}_{class}_{score}"
    where the row values are the difference in performance between the 
    majoritized class in 'majority_classes' for 'demographic_category' 
    and 'class' in terms of 'score'. Thus, the column 'GENDER_F_auprc' 
    would have values of the form "{AUPRC for M} - {AUPRC for F}". 
    Also returns a DataFrame of raw model performance by columns of 
    'dem_cols'.
    The model is trained on data resampled based on the columns in 
    'sdoh_list_rep' using MLSMOTE and random undersampling. The model 
    is not trained on any data in 'dem_cols' except for the columns in 
    'sdoh_list_inf'.
    Hyperparameter optimizer optimizes for the first entry of 
    'score_funcs'.
    REQUIRES THE 'train_pos_prop' ARGUMENT FROM OTHER FUNCTIONS TO BE 
    EQUAL TO 0.5 AND (except for initial train/val bootstrap) 
    'id_split' TO BE FALSE!
    The 'outfile' argument is the filename of a TXT file to hold 
    intermediate and final results.
    The remaining arguments are as given in 'disparities_analysis()'.
    Wilcoxon signed-rank tests are performed to detect the existence 
    of disparities.
    Saves intermediate and final results, including the existence of 
    disparities, to a file 'outfile'.
    """
    
    # Track the time
    write_print(f"Starting combination mitigation: {time.asctime()}", outfile)
    # Get copies of X_in and y_in
    X = X_in.copy()
    y = y_in.copy()
    # Which classes are there for each demographic category?
    classes = {col:X[col].unique() for col in dem_cols}
    # Initialize arrays to hold scores
    num_scores = len(score_funcs)
    overall_scores = [np.zeros(k) for _ in score_funcs]
    scores = [{col:{clss:np.zeros((k,1)) for clss in classes[col]} \
                                                         for col in dem_cols} \
                                                          for _ in score_funcs]
    # Thresholds, if applicable
    if any(threshs):
        saved_threshes = []
    # Array to hold number in each class in each bootstrap sample
    num_clss = {col:{clss:np.zeros((k,1)) for clss in classes[col]} \
                                                           for col in dem_cols}
    pos_clss = {col:{clss:np.zeros((k,1)) for clss in classes[col]} \
                                                           for col in dem_cols}
    
    # One-hot encode the selected SDoH column(s)
    prefix_sdoh = sdoh_list_inf.copy()
    if type(sdoh_list_inf) != str and "SVI_QUARTILE" in sdoh_list_inf:
        # Remove an already categorical column prefix if it exists
        prefix_sdoh.remove("SVI_QUARTILE")
    one_hot = pd.get_dummies(X[sdoh_list_inf], prefix=prefix_sdoh, dtype=int)
    # Handle non-prefixed columns
    one_hot.rename(columns={"SVI_QUARTILE":"SVI_QUARTILE_all"}, inplace=True)
    # Add in the one-hot columns; the original columns will be dropped
    X = X.join(one_hot)
    
    # Bootstrap the model k times
    for i in range(k):
        # Get the bootstrap sample
        X_train, y_train, X_val, y_val = bootstrap(X, y, 0.5, val_pos_prop, 
                                                   train_split_prop, pos_vals, 
                                                   id_split, id_col, 
                                                   out_col, original_size)
        # Reindex X_train and y_train to avoid duplicate index problems
        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        # Make a copy of the validation data without demographic categories
        X_val_clean = X_val.drop(columns=dem_cols)
        # Drop ID columns since those won't work with MLSMOTE
        if id_split:
            X_train = X_train.drop(columns=id_col)
            y_train = y_train.loc[:,out_col]
            X_val = X_val.drop(columns=id_col)
            X_val_clean = X_val_clean.drop(columns=id_col)
            y_val = y_val.loc[:,out_col]
        # Insert the outcomes into X to preserve them through MLSMOTE
        X_resamp = pd.concat([X_train, y_train], axis=1)
        # Create a binary label DataFrame for use with MLSMOTE
        labels = pd.get_dummies(X_resamp[sdoh_list_rep], prefix=sdoh_list_rep, 
                                columns=sdoh_list_rep, dtype=int)
        # Save demographic label proportions for tracking
        pre_counts = labels.sum()
        pre_prop = labels.sum()/len(labels)
        ########## DEBUGGING ##########
        irlbls = IRLbl(labels.sum().values)
        write_print("Labels:", outfile)
        for nc, col in enumerate(labels.columns):
            n_in_col = labels[col].sum()
            write_print(f"\t{col}: {n_in_col}, {n_in_col/len(labels)}", 
                        outfile)
            write_print(f"\t\tIRLbl: {irlbls[nc]} vs. {np.mean(irlbls)}", 
                        outfile)
        ########## DEBUGGING ##########
        
        # Get the samples to upsample based on label imbalance ratio (IRLbl)
        X_train_ups, labels_ups = mlsmote.get_minority_instace(X_resamp, 
                                                               labels)
        # Get the samples to downsample
        ups_indices = mlsmote.get_index(labels)
        X_train_maj = X_resamp[~X_resamp.index.isin(ups_indices)]
        labels_maj = labels[~labels.index.isin(ups_indices)]
        # How many additional minority samples are necessary?
        n_upsamp = len(X_resamp)//2 - len(X_train_ups)
        ########## DEBUGGING ##########
        write_print(f"\nlen(X_train_maj): {len(X_train_maj)}", outfile)
        write_print(f"len(X_train_ups): {len(X_train_ups)}", outfile)
        ########## DEBUGGING ##########
        # Get rid of unnecessary categorical columns for MLSMOTE
        cat_cols = dem_cols.copy()
        if "SVI_QUARTILE" in cat_cols:
            # Remove an already categorical column prefix if it exists
            cat_cols.remove("SVI_QUARTILE")
        # Make sure to keep the necessary columns
        for col in sdoh_list_inf:
            if col in cat_cols:
                cat_cols.remove(col)
        # Filter out unneeded dummy variables
        cat_cols = [col for col in X_train_ups.columns if \
                                             any([c in col for c in cat_cols])]
        # Get rid of unnecessary dummy and non-dummy categorical columns
        X_train_ups.drop(columns=set(dem_cols+cat_cols), inplace=True)
        # Get the additional samples
        X_train_ups, labels_ups = mlsmote.MLSMOTE(X_train_ups, labels_ups, 
                                                  n_upsamp)
        # Downsample the majority samples
        X_train_maj = X_train_maj.sample(n=len(X_resamp)//2)
        labels_maj = labels.loc[X_train_maj.index,:]
        # Add the additional samples to the original training set
        X_train = pd.concat([X_train_maj, X_train_ups], ignore_index=True)
        labels = pd.concat([labels_maj, labels_ups], ignore_index=True)
        # Extract the outcomes from X now that MLSMOTE is over
        y_train = X_train[out_col].astype(int)
        X_train.drop(columns=out_col, inplace=True)
        # Drop the demographic category columns from the training data
        X_train.drop(columns=dem_cols, inplace=True)
        
        # Save demographic label proportions for tracking
        post_counts = labels.sum()
        post_prop = labels.sum()/len(labels)
        # Save outcome proportions for tracking
        y_out_count = {"total":y_train.value_counts()}
        y_out_prop = {"total":y_train.value_counts(normalize=True)}
        for col in labels.columns:
            y_col = y_train[labels[col] == 1]
            y_out_count[col] = y_col.value_counts()
            y_out_prop[col] = y_col.value_counts(normalize=True)
        
        # Shuffle to avoid training on order
        X_train, y_train = shuffle(X_train,y_train)
        
        # Handle nans
        X_train.dropna(axis=0, inplace=True)
        y_train = y_train[X_train.index]
        
        # Get the best hyperparameters and model input features
        cols, best_params = model_optimizer(name, X_train, y_train, 
                                            score_funcs[0], probas[0], 
                                            n_trials, [1], False, 
                                            id_col, out_col)
        
        # Instantiate the model
        model = new_classifier(name, best_params)
        # Train the model
        model.fit(X_train, y_train)
        # Find a 90% sensitivity threshold for the model
        if any(threshs):
            if fix_thresh_on_val:
                probs_pred = model.predict_proba(X_val_clean)[:,1]
                sens_thresh = get_sens_threshold(y_val, probs_pred)
            else:
                probs_pred = model.predict_proba(X_train)[:,1]
                sens_thresh = get_sens_threshold(y_train, probs_pred)
            saved_threshes.append(sens_thresh)
        # Validate the model over each score function
        for j, score in enumerate(score_funcs):
            # Validate the model
            if probas[j]:
                prediction = model.predict_proba(X_val_clean)[:,1]
            else:
                prediction = model.predict(X_val_clean)
            # Get and save the overall model performance
            # Check if more arguments need to be passed
            if threshs[j]:
                overall_scores[j][i] = score(y_val, prediction, sens_thresh)
            else:
                overall_scores[j][i] = score(y_val, prediction)
            # Iterate through each demographic category to be assessed
            for col in dem_cols:
                # Iterate through each class in the demographic category
                for clss in classes[col]:
                    # Get the data for the class
                    X_clss = X_val.where(X_val[col]==clss).dropna(subset=[col])
                    X_clss.drop(columns=dem_cols, inplace=True)
                    y_clss = y_val.loc[X_clss.index]
                    # Set the proportion of positive samples in the subset
                    X_clss, y_clss = reproportion_pos(X_clss, y_clss, 
                                                      val_pos_prop)
                    # Tracking info
                    num_clss[col][clss][i] = len(y_clss)
                    pos_clss[col][clss][i] = sum(y_clss)
                    # Get the model predictions for the class
                    if probas[j]:
                        prediction_clss = model.predict_proba(X_clss)[:,1]
                    else:
                        prediction_clss = model.predict(X_clss)
                    # Get and save the model performance for this class
                    # Check if more arguments need to be passed
                    if threshs[j]:
                        scores[j][col][clss][i] = score(y_clss, 
                                                        prediction_clss, 
                                                        sens_thresh)
                    else:
                        scores[j][col][clss][i] = score(y_clss, 
                                                        prediction_clss)
        # Intermediate results
        write_print(f"\nResults at bootstrap round {i+1}/{k}:", outfile)
        write_print("\tOverall scores:", outfile)
        for j in range(len(score_funcs)):
            write_print(f"\t\t{overall_scores[j][i]}", outfile)
        for col in dem_cols:
            write_print(f"\t{col}:", outfile)
            for clss in classes[col]:
                write_print(f"\t\t{clss}:", outfile)
                for j in range(len(score_funcs)):
                    write_print(f"\t\t\t{scores[j][col][clss][i]}", outfile)
        # Counts of resampled categories
        write_print(f"\n\tCategories: {', '.join(sdoh_list_rep)}", outfile)
        write_print("\tCounts for each class:", outfile)
        write_print(f"\t\t\tBefore resampling\t\tAfter resampling", outfile)
        for clss in pre_counts.index:
            b_c = pre_counts[clss]
            b_p = pre_prop[clss]
            a_c = post_counts[clss]
            a_p = post_prop[clss]
            write_print(f"\t\t{clss}: {b_c} ({round(b_p,4)})", outfile, end="")
            write_print(f"\t\t{a_c} ({round(a_p,4)})", outfile)
        # Outcome counts overall and by categories
        write_print("\n\tOutcomes:", outfile)
        write_print("\t\tOverall:\t", outfile, end="")
        write_print(f"1: {y_out_count['total'][1]} ", outfile, end="")
        write_print(f"({round(y_out_prop['total'][1],4)})\t", outfile, end="")
        write_print(f"0: {y_out_count['total'][0]} ", outfile, end="")
        write_print(f"({round(y_out_prop['total'][0],4)})", outfile)
        for col in labels.columns:
            write_print(f"\t\t{col}:\t", outfile, end="")
            write_print(f"1: {y_out_count[col][1]} ", outfile, end="")
            write_print(f"({y_out_prop[col][1]})\t", outfile, end="")
            write_print(f"0: {y_out_count[col][0]} ", outfile, end="")
            write_print(f"0: ({y_out_prop[col][0]})", outfile)
        write_print("\n", outfile)
    
    # Initialize a report string
    report = "\nOverall model performance with resampled "
    report += f"{', '.join(sdoh_list_rep)}:\n"
    for j, score in enumerate(score_funcs):
        report += f"\tMean {str(score).split(' ')[1]}: "
        report += f"{np.mean(overall_scores[j])}\n"
        report += "\tTotal performance by round :"
        report += f"\n\t\t{overall_scores[j].flatten()}\n"
    report += "\n\n"
    
    if any(threshs):
        report += f"90% sensitivity thresholds: {saved_threshes}\n\n"
    for j, score in enumerate(score_funcs):
        report += f"Score: {str(score).split(' ')[1]}\n"
        for i, col in enumerate(dem_cols):
            report += f"\tColumn: {col}\n"
            for clss in classes[col]:
                report += f"\t\tClass: {clss}\n"
                report += f"\t\t\tMean: {np.mean(scores[j][col][clss])}\n"
                report += f"\t\t\t{scores[j][col][clss].flatten()}\n"
                report += f"\t\t\tLengths: {num_clss[col][clss].flatten()}\n"
                mean_len = np.mean(num_clss[col][clss])
                report += f"\t\t\tMean length: {mean_len}\n"
                report += f"\t\t\tPos.vals.: {pos_clss[col][clss].flatten()}\n"
                mean_pos = np.mean(pos_clss[col][clss])
                report += f"\t\t\tMean number of positives: {mean_pos}\n"
                report += f"\t\t\tMean pos. proportion: {mean_pos/mean_len}\n"
                # Is there still a disparity here?
                if clss != majority_classes[i]:
                    # Identify the majoritized class
                    m_clss = majority_classes[i]
                    # Get the respective scores
                    scores_maj = scores[j][col][m_clss].flatten()
                    scores_min = scores[j][col][clss].flatten()
                    # Get the p-value
                    diffs = scores_maj - scores_min
                    w_res = wilcoxon(diffs)
                    p_val = w_res.pvalue
                    # Report the p-value
                    report += f"\tWilcoxon signed-rank p-value {m_clss} v. "
                    report += str(clss)
                    report += f": {p_val}\n"
                    # Report the disparity/difference
                    report += f"\tMean difference: {np.mean(diffs)}\n"
                    report += f"\tAbsolute differences: {diffs}\n"
                report += "\n\n"
    
    # Print the report
    write_print(report, outfile)
    
    # Create an array of disparities
    n_cols = len(score_funcs)*sum([len(classes[col])-1 for col in dem_cols])
    disp_array = np.zeros((k,n_cols))
    # Current column tracker
    col_i = 0
    # Track disparity column names
    disp_cols = []
    # Iterate through each score function
    for j, score in enumerate(score_funcs):
        # Iterate through each demographic_category
        for i, col in enumerate(dem_cols):
            # Get the "majoritized" class and scores
            m_clss = majority_classes[i]
            scores_maj = scores[j][col][m_clss].flatten()
            # Iterate over each minoritized class
            for l, clss in enumerate([c for c in classes[col] if c != m_clss]):
                scores_min = scores[j][col][clss].flatten()
                # Fill in the values of the disparities array
                disp_array[:,col_i] = scores_maj - scores_min
                # Update the column tracker
                col_i += 1
                # Update the column names
                disp_cols.append(f"{col}_{clss}_{str(score).split(' ')[1]}")
    
    # Create an array of raw model performances
    n_cols_raw = len(score_funcs)*sum([len(classes[col]) for col in dem_cols])
    perf_array = np.zeros((k,n_cols_raw))
    # Current column tracker
    col_i = 0
    # Track performance column names
    perf_cols = []
    # Iterate through each score function
    for j, score in enumerate(score_funcs):
        # Iterate through each demographic_category
        for i, col in enumerate(dem_cols):
            # Iterate over each class
            for l, clss in enumerate(classes[col]):
                scores_min = scores[j][col][clss].flatten()
                # Fill in the values of the disparities array
                perf_array[:,col_i] = scores[j][col][clss].flatten()
                # Update the column tracker
                col_i += 1
                # Update the column names
                perf_cols.append(f"{col}_{clss}_{str(score).split(' ')[1]}")
    
    # Return the DataFrames
    disp_df = pd.DataFrame(disp_array, columns=disp_cols)
    perf_df = pd.DataFrame(perf_array, columns=perf_cols)
    write_print(time.asctime(), outfile)
    return disp_df, perf_df


def combo_disp_naive(name, X_in, y_in, dem_cols=["GENDER", "RACE"], 
                     sdoh_list_rep=["RACE"], sdoh_list_inf=["RACE"], 
                     majority_classes=["M", "White or Caucasian"], 
                     score_funcs=[auprc, ppv_fixed_sens, 
                                  recall_fixed_thresh], 
                     probas=[True, True, True], 
                     threshs=[False, False, True], 
                     fix_thresh_on_val=True, k=10, val_pos_prop=0.05,
                     train_split_prop=0.7, pos_vals=[1], 
                     id_split=True, id_col="CSN", 
                     out_col="end_point_6", original_size=True, 
                     n_trials=30, outfile="sdoh_disparities.txt"):
    """Given the name of an sklearn-compatible model 'name', DataFrame 
    'X_in', and outcomes Series 'y_in', returns a DataFrame of 
    differences in model performance based on the columns 'dem_cols' 
    in 'X'. The returned DataFrame has columns of the form: 
        "{demographic_category}_{class}_{score}"
    where the row values are the difference in performance between the 
    majoritized class in 'majority_classes' for 'demographic_category' 
    and 'class' in terms of 'score'. Thus, the column 'GENDER_F_auprc' 
    would have values of the form "{AUPRC for M} - {AUPRC for F}". 
    Also returns a DataFrame of raw model performance by columns of 
    'dem_cols'.
    The model is trained on data resampled based on the columns in 
    'sdoh_list_rep' using SMOTE and random undersampling. The model 
    is not trained on any data in 'dem_cols' except for the columns in 
    'sdoh_list_inf'.
    Hyperparameter optimizer optimizes for the first entry of 
    'score_funcs'.
    REQUIRES THE 'train_pos_prop' ARGUMENT FROM OTHER FUNCTIONS TO BE 
    EQUAL TO 0.5 AND (except for initial train/val bootstrap) 
    'id_split' TO BE FALSE!
    The 'outfile' argument is the filename of a TXT file to hold 
    intermediate and final results.
    The remaining arguments are as given in 'disparities_analysis()'.
    Wilcoxon signed-rank tests are performed to detect the existence 
    of disparities.
    Saves intermediate and final results, including the existence of 
    disparities, to a file 'outfile'.
    """
    
    # Track the time
    write_print(f"Starting combination mitigation: {time.asctime()}", outfile)
    # Get copies of X_in and y_in
    X = X_in.copy()
    y = y_in.copy()
    # Which classes are there for each demographic/SDoH category?
    classes = {col:X[col].unique() for col in dem_cols}
    # Classes for resampling
    resamp_classes = {col:X[col].unique() for col in sdoh_list_rep}
    # Initialize arrays to hold scores
    num_scores = len(score_funcs)
    overall_scores = [np.zeros(k) for _ in score_funcs]
    scores = [{col:{clss:np.zeros((k,1)) for clss in classes[col]} \
                                                         for col in dem_cols} \
                                                          for _ in score_funcs]
    # Thresholds, if applicable
    if any(threshs):
        saved_threshes = []
    # Arrays to hold number per class before and after each bootstrap sample
    num_sdoh_pre = {col:{clss:np.zeros(k) for clss in resamp_classes[col]} \
                                                      for col in sdoh_list_rep}
    pos_sdoh_pre = {col:{clss:np.zeros(k) for clss in resamp_classes[col]} \
                                                      for col in sdoh_list_rep}
    num_sdoh_post = {col:{clss:np.zeros(k) for clss in resamp_classes[col]} \
                                                      for col in sdoh_list_rep}
    pos_sdoh_post = {col:{clss:np.zeros(k) for clss in resamp_classes[col]} \
                                                      for col in sdoh_list_rep}
    num_clss = {col:{clss:np.zeros((k,1)) for clss in classes[col]} \
                                                           for col in dem_cols}
    pos_clss = {col:{clss:np.zeros((k,1)) for clss in classes[col]} \
                                                           for col in dem_cols}
    
    # One-hot encode the selected SDoH column(s) to keep
    prefix_sdoh = sdoh_list_inf.copy()
    if type(sdoh_list_inf) != str and "SVI_QUARTILE" in sdoh_list_inf:
        # Remove an already categorical column prefix if it exists
        prefix_sdoh.remove("SVI_QUARTILE")
    one_hot = pd.get_dummies(X[sdoh_list_inf], prefix=prefix_sdoh, dtype=int)
    # Handle non-prefixed columns
    one_hot.rename(columns={"SVI_QUARTILE":"SVI_QUARTILE_all"}, inplace=True)
    # Add in the one-hot columns; the original columns will be dropped later
    X = X.join(one_hot)
    
    # Bootstrap the model k times
    for i in range(k):
        # Get the bootstrap sample
        X_train, y_train, X_val, y_val = bootstrap(X, y, 0.5, val_pos_prop, 
                                                   train_split_prop, pos_vals, 
                                                   id_split, id_col, 
                                                   out_col, original_size)
        # Reindex X_train and y_train to avoid duplicate index problems
        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        # Combine SDoH class with outcome for composite resample class
        resamp_y = y_train[out_col].astype(str)
        for col in sdoh_list_rep:
            resamp_y += X_train[col].astype(str)
        resamp_y.rename(out_col, inplace=True)
        # Drop the demographic category columns from the training data
        X_train.drop(columns=dem_cols, inplace=True)
        # Make a copy of the validation data without demographic categories
        X_val_clean = X_val.drop(columns=dem_cols)
        # Drop ID columns since those won't work with MLSMOTE
        if id_split:
            X_train = X_train.drop(columns=id_col)
            y_train = y_train.loc[:,out_col]
            X_val = X_val.drop(columns=id_col)
            X_val_clean = X_val_clean.drop(columns=id_col)
            y_val = y_val.loc[:,out_col]
        
        # Resample the training data, keeping outcomes at the right proportion
        # How many should each class have?
        n_per_class = round(len(resamp_y)/len(resamp_y.unique()))
        # How many are in each composite class?
        resamp_counts = resamp_y.value_counts()
        resamp_prop = resamp_y.value_counts(normalize=True)
        # How many are in each individual class, if applicable
        for col in sdoh_list_rep:
            for clss in resamp_classes[col]:
                y_sdoh_class = resamp_y[resamp_y.str.contains(str(clss))]
                num_sdoh_pre[col][clss][i] = len(y_sdoh_class)
                pos_sdoh_pre[col][clss][i] = len(y_sdoh_class) - \
                              len(y_sdoh_class[y_sdoh_class.str.contains("0")])
        
        # Add copies of patients below the KNN representation threshold
        for clss in resamp_counts.index:
            if resamp_counts[clss] < 6: # Default k neighbors
                n_needed = 6 - resamp_counts[clss]
                add_idxs = resamp_y.where(resamp_y == clss).dropna().index
                # Duplicate random patients if fewer than 3 are needed
                if n_needed < 3:
                    add_idxs = np.random.choice(add_idxs, n_needed)
                # Duplicate all patients otherwise
                else:
                    n_copies = 6//resamp_counts[clss] - 1
                    add_idxs = list(add_idxs)*n_copies
                X_train = pd.concat([X_train, X_train.loc[add_idxs,:]], 
                                     ignore_index=True)
                resamp_y = pd.concat([resamp_y, resamp_y.loc[add_idxs]], 
                                     ignore_index=True)
        # Divide classes into oversample versus undersample
        # Undersample
        under_samp = resamp_counts > n_per_class
        under_samp_vals = under_samp.where(under_samp).dropna().index
        under_bool = resamp_y.isin(under_samp_vals)
        resamp_y_u = resamp_y.where(under_bool).dropna()
        X_train_under = X_train.where(under_bool).dropna()
        # Oversample
        over_samp_vals = under_samp.where(~under_samp).dropna().index
        resamp_y_o = resamp_y.where(~under_bool).dropna()
        X_train_over = X_train.where(~under_bool).dropna()
        # Do the resampling
        # Oversample
        over_dict = {c:n_per_class for c in over_samp_vals}
        smt = SMOTE(sampling_strategy=over_dict)
        X_train_over, resamp_y_o = smt.fit_resample(X_train_over, resamp_y_o)
        # Undersample
        under_dict = {c:n_per_class for c in under_samp_vals}
        rus = RandomUnderSampler(sampling_strategy=under_dict)
        X_train_under, resamp_y_u = rus.fit_resample(X_train_under, resamp_y_u)
        # Recombine
        X_train = pd.concat([X_train_over, X_train_under], ignore_index=True)
        y_train = pd.concat([resamp_y_o, resamp_y_u], ignore_index=True)
        # Save class proportions for tracking
        new_counts = y_train.value_counts()
        new_counts_prop = y_train.value_counts(normalize=True)
        # Individual proportions, if applicable
        for col in sdoh_list_rep:
            for clss in resamp_classes[col]:
                """print(y_train, end="\n\n")
                print(clss, end="\n\n")
                print(y_train.str.contains(str(clss)), end="\n\n")
                print(y_train[y_train.str.contains(str(clss))], 
                      end="\n\n")"""
                y_sdoh_class = y_train[y_train.str.contains(str(clss))]
                num_sdoh_post[col][clss][i] = len(y_sdoh_class)
                pos_sdoh_post[col][clss][i] = len(y_sdoh_class) - \
                              len(y_sdoh_class[y_sdoh_class.str.contains("0")])
        
        # Extract outcome from combination SDoH class/outcome
        y_train = y_train.apply(lambda x: int(x[0])).astype(int)
        
        # Shuffle to avoid training on order
        X_train, y_train = shuffle(X_train,y_train)
        
        """# Handle nans
        X_train.dropna(axis=0, inplace=True)
        y_train = y_train[X_train.index]"""
        
        # Get the best hyperparameters and model input features
        cols, best_params = model_optimizer(name, X_train, y_train, 
                                            score_funcs[0], probas[0], 
                                            n_trials, [1], False, 
                                            id_col, out_col)
        
        # Instantiate the model
        model = new_classifier(name, best_params)
        # Train the model
        model.fit(X_train, y_train)
        # Find a 90% sensitivity threshold for the model
        if any(threshs):
            if fix_thresh_on_val:
                probs_pred = model.predict_proba(X_val_clean)[:,1]
                sens_thresh = get_sens_threshold(y_val, probs_pred)
            else:
                probs_pred = model.predict_proba(X_train)[:,1]
                sens_thresh = get_sens_threshold(y_train, probs_pred)
            saved_threshes.append(sens_thresh)
        # Validate the model over each score function
        for j, score in enumerate(score_funcs):
            # Validate the model
            if probas[j]:
                prediction = model.predict_proba(X_val_clean)[:,1]
            else:
                prediction = model.predict(X_val_clean)
            # Get and save the overall model performance
            # Check if more arguments need to be passed
            if threshs[j]:
                overall_scores[j][i] = score(y_val, prediction, sens_thresh)
            else:
                overall_scores[j][i] = score(y_val, prediction)
            # Iterate through each demographic category to be assessed
            for col in dem_cols:
                # Iterate through each class in the demographic category
                for clss in classes[col]:
                    # Get the data for the class
                    X_clss = X_val.where(X_val[col]==clss).dropna(subset=[col])
                    X_clss.drop(columns=dem_cols, inplace=True)
                    y_clss = y_val.loc[X_clss.index]
                    # Set the proportion of positive samples in the subset
                    X_clss, y_clss = reproportion_pos(X_clss, y_clss, 
                                                      val_pos_prop)
                    # Tracking info
                    num_clss[col][clss][i] = len(y_clss)
                    pos_clss[col][clss][i] = sum(y_clss)
                    # Get the model predictions for the class
                    if probas[j]:
                        prediction_clss = model.predict_proba(X_clss)[:,1]
                    else:
                        prediction_clss = model.predict(X_clss)
                    # Get and save the model performance for this class
                    # Check if more arguments need to be passed
                    if threshs[j]:
                        scores[j][col][clss][i] = score(y_clss, 
                                                        prediction_clss, 
                                                        sens_thresh)
                    else:
                        scores[j][col][clss][i] = score(y_clss, 
                                                        prediction_clss)
        # Intermediate results
        write_print(f"\nResults at bootstrap round {i+1}/{k}:", outfile)
        write_print("\tOverall scores:", outfile)
        for j in range(len(score_funcs)):
            write_print(f"\t\t{overall_scores[j][i]}", outfile)
        for col in dem_cols:
            write_print(f"\t{col}:", outfile)
            for clss in classes[col]:
                write_print(f"\t\t{clss}:", outfile)
                for j in range(len(score_funcs)):
                    write_print(f"\t\t\t{scores[j][col][clss][i]}", outfile)
        # Counts of resampled categories
        write_print(f"\n\tCategories: {', '.join(sdoh_list_rep)}", outfile)
        write_print("\tCounts for each combination class:", outfile)
        write_print(f"\t\t\tBefore resampling\t\tAfter resampling", outfile)
        for clss in resamp_counts.index:
            b_c = resamp_counts[clss]
            b_p = resamp_prop[clss]
            a_c = new_counts[clss]
            a_p = new_counts_prop[clss]
            write_print(f"\t\t{clss}: {b_c} ({round(b_p,4)})", outfile, end="")
            write_print(f"\t\t{a_c} ({round(a_p,4)})", outfile)
        write_print("\n\tCounts for each individual resampled class:", outfile)
        write_print(f"\t\t\tBefore resampling\t\tAfter resampling", 
                    outfile)
        for col in sdoh_list_rep:
            for clss in resamp_classes[col]:
                b_c = num_sdoh_pre[col][clss][i]
                a_c = num_sdoh_post[col][clss][i]
                write_print(f"\t\t{clss} count: {b_c}\t\t{a_c}", outfile)
                b_p = pos_sdoh_pre[col][clss][i]
                a_p = pos_sdoh_post[col][clss][i]
                write_print(f"\t\t\t positive: {b_p}\t\t{a_p}", outfile)
        write_print("\n", outfile)
        """# Outcome counts overall and by categories
        write_print("\n\tOutcomes:", outfile)
        write_print("\t\tOverall:\t", outfile, end="")
        write_print(f"1: {y_out_count['total'][1]} ", outfile, end="")
        write_print(f"({round(y_out_prop['total'][1],4)})\t", outfile, end="")
        write_print(f"0: {y_out_count['total'][0]} ", outfile, end="")
        write_print(f"({round(y_out_prop['total'][0],4)})", outfile)
        for col in labels.columns:
            write_print(f"\t\t{col}:\t", outfile, end="")
            write_print(f"1: {y_out_count[col][1]} ", outfile, end="")
            write_print(f"({y_out_prop[col][1]})\t", outfile, end="")
            write_print(f"0: {y_out_count[col][0]} ", outfile, end="")
            write_print(f"0: ({y_out_prop[col][0]})", outfile)
        write_print("\n", outfile)"""
    
    # Initialize a report string
    report = "\nOverall model performance with resampled "
    report += f"{', '.join(sdoh_list_rep)}:\n"
    for j, score in enumerate(score_funcs):
        report += f"\tMean {str(score).split(' ')[1]}: "
        report += f"{np.mean(overall_scores[j])}\n"
        report += "\tTotal performance by round :"
        report += f"\n\t\t{overall_scores[j].flatten()}\n"
    report += "\n\n"
    
    if any(threshs):
        report += f"90% sensitivity thresholds: {saved_threshes}\n\n"
    for j, score in enumerate(score_funcs):
        report += f"Score: {str(score).split(' ')[1]}\n"
        for i, col in enumerate(dem_cols):
            report += f"\tColumn: {col}\n"
            for clss in classes[col]:
                report += f"\t\tClass: {clss}\n"
                report += f"\t\t\tMean: {np.mean(scores[j][col][clss])}\n"
                report += f"\t\t\t{scores[j][col][clss].flatten()}\n"
                report += f"\t\t\tLengths: {num_clss[col][clss].flatten()}\n"
                mean_len = np.mean(num_clss[col][clss])
                report += f"\t\t\tMean length: {mean_len}\n"
                report += f"\t\t\tPos.vals.: {pos_clss[col][clss].flatten()}\n"
                mean_pos = np.mean(pos_clss[col][clss])
                report += f"\t\t\tMean number of positives: {mean_pos}\n"
                report += f"\t\t\tMean pos. proportion: {mean_pos/mean_len}\n"
                # Is there still a disparity here?
                if clss != majority_classes[i]:
                    # Identify the majoritized class
                    m_clss = majority_classes[i]
                    # Get the respective scores
                    scores_maj = scores[j][col][m_clss].flatten()
                    scores_min = scores[j][col][clss].flatten()
                    # Get the p-value
                    diffs = scores_maj - scores_min
                    w_res = wilcoxon(diffs)
                    p_val = w_res.pvalue
                    # Report the p-value
                    report += f"\tWilcoxon signed-rank p-value {m_clss} v. "
                    report += str(clss)
                    report += f": {p_val}\n"
                    # Report the disparity/difference
                    report += f"\tMean difference: {np.mean(diffs)}\n"
                    report += f"\tAbsolute differences: {diffs}\n"
                report += "\n\n"
    
    # Print the report
    write_print(report, outfile)
    
    # Create an array of disparities
    n_cols = len(score_funcs)*sum([len(classes[col])-1 for col in dem_cols])
    disp_array = np.zeros((k,n_cols))
    # Current column tracker
    col_i = 0
    # Track disparity column names
    disp_cols = []
    # Iterate through each score function
    for j, score in enumerate(score_funcs):
        # Iterate through each demographic_category
        for i, col in enumerate(dem_cols):
            # Get the "majoritized" class and scores
            m_clss = majority_classes[i]
            scores_maj = scores[j][col][m_clss].flatten()
            # Iterate over each minoritized class
            for l, clss in enumerate([c for c in classes[col] if c != m_clss]):
                scores_min = scores[j][col][clss].flatten()
                # Fill in the values of the disparities array
                disp_array[:,col_i] = scores_maj - scores_min
                # Update the column tracker
                col_i += 1
                # Update the column names
                disp_cols.append(f"{col}_{clss}_{str(score).split(' ')[1]}")
    
    # Create an array of raw model performances
    n_cols_raw = len(score_funcs)*sum([len(classes[col]) for col in dem_cols])
    perf_array = np.zeros((k,n_cols_raw))
    # Current column tracker
    col_i = 0
    # Track performance column names
    perf_cols = []
    # Iterate through each score function
    for j, score in enumerate(score_funcs):
        # Iterate through each demographic_category
        for i, col in enumerate(dem_cols):
            # Iterate over each class
            for l, clss in enumerate(classes[col]):
                scores_min = scores[j][col][clss].flatten()
                # Fill in the values of the disparities array
                perf_array[:,col_i] = scores[j][col][clss].flatten()
                # Update the column tracker
                col_i += 1
                # Update the column names
                perf_cols.append(f"{col}_{clss}_{str(score).split(' ')[1]}")
    
    # Return the DataFrames
    disp_df = pd.DataFrame(disp_array, columns=disp_cols)
    perf_df = pd.DataFrame(perf_array, columns=perf_cols)
    write_print(time.asctime(), outfile)
    return disp_df, perf_df



"""
References:
Normal values for imputation were derived by taking the midpoint of 
the healthy ranges found in the below references.
https://www.mayoclinic.org/healthy-lifestyle/fitness/expert-answers/heart-rate/faq-20057979
https://www.ncbi.nlm.nih.gov/books/NBK537306/
https://www.nia.nih.gov/health/high-blood-pressure-and-older-adults
https://www.mayoclinic.org/symptoms/hypoxemia/basics/definition/sym-20050930
https://www.ncbi.nlm.nih.gov/books/NBK331/
https://www.mdcalc.com/calc/1096/sirs-sepsis-septic-shock-criteria
https://www.ucsfhealth.org/medical-tests/wbc-count
https://www.ncbi.nlm.nih.gov/books/NBK536919/
https://www.nhlbi.nih.gov/health/thrombocytopenia
https://emedicine.medscape.com/article/2074068-overview?form=fpf
https://www.ncbi.nlm.nih.gov/books/NBK305/ (min for male, max for female)
https://www.ncbi.nlm.nih.gov/books/NBK305/
https://www.cdc.gov/diabetes/treatment/index.html
https://www.ncbi.nlm.nih.gov/books/NBK544269/
https://www.redcrossblood.org/donate-blood/dlp/hematocrit.html (halfway between female min and male max)
https://www.ncbi.nlm.nih.gov/books/NBK551648/
https://www.ncbi.nlm.nih.gov/books/NBK470202/

Hyperparameter ranges chosen in consultation with documentation from 
sklearn and the following references:
https://quantdare.com/decision-trees-gini-vs-entropy/
https://towardsdatascience.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb
https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d
Hyperparameter search limited by computational constraints
SVC dropped because training time scales quadratically with the number of samples, impractical above 1e4 samples:
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
GP dropped because inefficient beyond a few dozen features
https://scikit-learn.org/1.5/modules/gaussian_process.html
"""