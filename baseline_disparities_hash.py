# baseline_disparities_hash.py
"""Given a hard-coded model, hyperparameters, and data, evaluates the 
baseline performance disparities of the model on the data.
12-7-24: Updated to not include hard-coded hyperparameters.
12-14-24: Updated to engineer ethnicity feature into Non-Hispanic 
White, Hispanic/Latino, and Other (which isn't reported in statistics).
12-17-24: Updated to include specified non-minoritized subclasses.
1-8-25: Updated to use CSVs de-identified with hashes instead of CSVs 
for use on non-PHI servers.
Command-line arguments:
    1. k (int): how many bootstrap samples to use for model evaluation
    2. n_trials (int): how many Optuna trials to run for 
                       hyperparameter optimization
    3. name (str): the name of the sklearn model type to use
"""


# Import statements

import sys
import time
import pandas as pd
from functions import auprc
from functions import ppv_fixed_sens
from functions import recall_fixed_thresh
from functions import disparities_analysis


# Run the file


if __name__ == "__main__":
    
    print(time.asctime()) ##########
    
    # Command-line arguments
    k = int(sys.argv[1])
    n_trials = int(sys.argv[2])
    name = sys.argv[3]
    
    # Which prediction window to use?
    out_col = "end_point_6"
    # IDs are kept in 'CSN' or somewhere else?
    id_col = "CSN"
    
    # Get the data
    X = pd.read_csv("icu_X_hash.csv")
    y = pd.read_csv("icu_y_hash.csv")
    print(time.asctime()) ##########
   
    # Drop the columns that aren't model inputs or outcomes
    X.drop(columns=["hr_slot", "hour_slot_admit"], inplace=True)
    y = y[[id_col, out_col]]
    y_out = y[out_col]
    
    # How many positive and negative?
    print(f"Positive y values: {y_out.isin(['SEPSIS']).sum()}")
    print(f"Total y values: {len(y)}")
    
    # Get the demographic data
    dem_df = pd.read_csv("demographics_hash.csv", index_col=id_col)
    # Get the language data
    lang_df = pd.read_csv("pat_language_deid_hash.csv", index_col=id_col)
    # Get the composite SVI quartile data
    svi_df = pd.read_csv("pat_SVI_quartiles_hash.csv", index_col=id_col)
    
    
    # Narrow down the patients to the ones with all demographic data available
    X = X[X[id_col].isin(lang_df.index)]
    X = X[X[id_col].isin(svi_df.index)]
    X = X[X[id_col].isin(dem_df.index)]
    
    # Add the demographic data to the patient data
    X["LANGUAGE"] = X[id_col].apply(lambda x: lang_df.loc[x,"LANGUAGE"])
    X["SVI_QUARTILE"] = X[id_col].apply(lambda x: svi_df.loc[x,"quartile"])
    X["GENDER"] = X[id_col].apply(lambda x: dem_df.loc[x,"GENDER"])
    X["RACE"] = X[id_col].apply(lambda x: dem_df.loc[x,"RACE"])
    X["ETHNICITY"] = X[id_col].apply(lambda x: dem_df.loc[x,"ETHNICITY"])
                                                                                
    # Engineer 'ETHNICITY' into Non-Hispanic White, Hispanic/Latino, and Other
    nhw = (X.ETHNICITY=="Not Hispanic/Latino")&(X.RACE=="White or Caucasian")
    X.loc[nhw, "ETHNICITY"] = "Non-Hispanic White"
    oth = ~X["ETHNICITY"].isin(["Hispanic/Latino","Non-Hispanic White"])
    X.loc[oth, "ETHNICITY"] = "Other"
    
    # Drop any classes that comprise less than 1% of the data to avoid 
    # insufficient data issues
    # Administrative sex
    X["GENDER"].mask(X["GENDER"].isin(["X","U"]), inplace=True)
    X.dropna(subset=["GENDER"], inplace=True)
    # Ethnicity
    X["ETHNICITY"].mask(X["ETHNICITY"] == \
                        "Native Hawaiian/Other Pacific Islander", inplace=True)
    X.dropna(subset=["ETHNICITY"], inplace=True)
    
    # Synch y with X
    y = y[y.index.isin(X.index)]
    
    # Perform the disparities analysis
    disparities_analysis(name, X, y, dem_cols=["GENDER", "SVI_QUARTILE", 
                                               "ETHNICITY", "LANGUAGE", 
                                               "RACE"],
                         majority_classes=["M", 1, "Non-Hispanic White", 
                                           "ENGLISH", "White or Caucasian"], 
                         score_funcs=[auprc, ppv_fixed_sens, 
                                      recall_fixed_thresh], 
                         probas=[True, True, True], threshs=[False, False, 
                                                             True], 
                         k=k, pos_vals=["SEPSIS"], n_trials=n_trials)
    
    print(time.asctime()) ##########