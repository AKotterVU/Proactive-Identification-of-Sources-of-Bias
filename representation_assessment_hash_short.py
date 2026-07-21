# representation_assessment_hash.py
"""Given a model type, data, disparities data from a previous model 
iteration, and an SDoH feature, outputs performance and disparity data 
which can be used to determine which if any of the disparities are 
significantly reduced when exactly one SDoH feature is resampled to 
equal representation across classes. The old disparities file is only 
used to determine if the column names are correct.
Uses CSVs de-identified with hashes instead of CSNs for use on non-PHI 
servers.
Saves disparities and performances to CSV files with a randomized ID 
matching the TXT output. Statistical analysis must be done separately 
on the outputs since this program is designed to be run multiple times 
in parallel.
Command-line arguments:
    1. k (int): how many bootstrap samples to use for model evaluation
    2. n_trials (int): how many Optuna trials to run for 
                       hyperparameter optimization
    3. name (str): the name of the sklearn model type to use
    4. sdoh (str): the SDoH to assess for affect on informativeness 
                   bias
"""


# Import statements

import sys
import time
import numpy as np
import pandas as pd
from functions import write_print
from functions import resamp_disparity


# Run the file


if __name__ == "__main__":
    
    print(time.asctime()) ##########
    
    # Command-line arguments
    k = int(sys.argv[1])
    n_trials = int(sys.argv[2])
    name = sys.argv[3]
    sdoh = sys.argv[4]
    
    # Which prediction window to use?
    out_col = "end_point_6"
    # IDs are kept in 'CSN' or somewhere else?
    id_col = "CSN"
    # Which file has the old disparities in it?
    old_disp_file = "Results/disparities_RF_2025_1_10_joined.csv"
    
    # Create filename for saving outputs
    date_str = "{}_{}_{}".format(*time.localtime()[0:3])
    rn = np.random.randint(1e4,1e5)
    outfile = f"representation_assessment_{sdoh}_{name}_{date_str}_{rn}.txt"
    
    # Track progress
    write_print("Start program", outfile)
    write_print(time.asctime(), outfile)
    
    # Get the data
    X = pd.read_csv("icu_X_hash.csv")
    y = pd.read_csv("icu_y_hash.csv")
   
    # Drop the columns that aren't model inputs or outcomes
    X.drop(columns=["hr_slot", "hour_slot_admit"], inplace=True)
    y = y[[id_col, out_col]]
    y_out = y[out_col]
    
    # How many positive and negative?
    write_print(f"Positive y values: {y_out.isin(['SEPSIS']).sum()}", outfile)
    write_print(f"Total y values: {len(y)}", outfile)
    
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
    
    # Get the old disparities
    old_disp_df = pd.read_csv(old_disp_file)
    
    # Get the new disparities
    new_disp_df, new_perf_df = resamp_disparity(name, X, y, 
                                                dem_cols=["GENDER",
                                                          "SVI_QUARTILE", 
                                                          "ETHNICITY", 
                                                          "LANGUAGE", "RACE"], 
                                                sdoh=sdoh, 
                                                majority_classes=["M", 1, 
                                                         "Non-Hispanic White", 
                                                                  "ENGLISH", 
                                                        "White or Caucasian"], 
                                                k=k, pos_vals=["SEPSIS"], 
                                                n_trials=n_trials, 
                                                outfile=outfile)
    # Debugging
    for i in range(len(old_disp_df.columns)):
        if old_disp_df.columns[i] != new_disp_df.columns[i]:
            print(f"{old_disp_df.columns[i]} == {new_disp_df.columns[i]}?")
    
    # Save the disparities and performances
    new_disp_df.to_csv(f"disparities_{name}_equal_{sdoh}_{date_str}_{rn}.csv", 
                       index=False)
    new_perf_df.to_csv(f"performance_{name}_equal_{sdoh}_{date_str}_{rn}.csv", 
                       index=False)