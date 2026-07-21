# join_all_files.py

"""Combines all the runs into their corresponding joined versions."""

import pandas as pd
import re
import os

if __name__ == "__main__":
    
    # Ethnicity
    # Performance
    eth_perf_re = re.compile("performance_RF_equal_ETHNICITY_2026")
    eth_files_p = ["All Runs Raw/"+f for f in os.listdir("All Runs Raw") if \
                                                         eth_perf_re.search(f)]
    eth_dfs_p = [pd.read_csv(f) for f in eth_files_p]
    eth_df_p = pd.concat(eth_dfs_p)
    eth_df_p.to_csv("Joined Runs/perf_RF_equal_ETHNICITY_UNDER_joined.csv", 
                    index=False)
    # Disparities
    eth_perf_re = re.compile("disparities_RF_equal_ETHNICITY_2026")
    eth_files_d = ["All Runs Raw/"+f for f in os.listdir("All Runs Raw") if \
                                                         eth_perf_re.search(f)]
    eth_dfs_d = [pd.read_csv(f) for f in eth_files_d]
    eth_df_d = pd.concat(eth_dfs_d)
    eth_df_d.to_csv("Joined Runs/disp_RF_equal_ETHNICITY_UNDER_joined.csv", 
                    index=False)
    
    # Race   -   RACE has 1200 runs instead of 800
    # Performance
    rac_perf_re = re.compile("performance_RF_equal_RACE_2026")
    rac_files_p = ["All Runs Raw/"+f for f in os.listdir("All Runs Raw") if \
                                                         rac_perf_re.search(f)]
    rac_dfs_p = [pd.read_csv(f) for f in rac_files_p]
    rac_df_p = pd.concat(rac_dfs_p)
    rac_df_p.to_csv("Joined Runs/perf_RF_equal_RACE_UNDER_joined.csv", 
                    index=False)
    # Disparities
    rac_perf_re = re.compile("disparities_RF_equal_RACE_2026")
    rac_files_d = ["All Runs Raw/"+f for f in os.listdir("All Runs Raw") if \
                                                         rac_perf_re.search(f)]
    rac_dfs_d = [pd.read_csv(f) for f in rac_files_d]
    rac_df_d = pd.concat(rac_dfs_d)
    rac_df_d.to_csv("Joined Runs/disp_RF_equal_RACE_UNDER_joined.csv", 
                    index=False)
    
    # Language
    # Performance
    lan_perf_re = re.compile("performance_RF_equal_LANGUAGE_2026")
    lan_files_p = ["All Runs Raw/"+f for f in os.listdir("All Runs Raw") if \
                                                         lan_perf_re.search(f)]
    lan_dfs_p = [pd.read_csv(f) for f in lan_files_p]
    lan_df_p = pd.concat(lan_dfs_p)
    lan_df_p.to_csv("Joined Runs/perf_RF_equal_LANGUAGE_UNDER_joined.csv", 
                    index=False)
    # Disparities
    lan_perf_re = re.compile("disparities_RF_equal_LANGUAGE_2026")
    lan_files_d = ["All Runs Raw/"+f for f in os.listdir("All Runs Raw") if \
                                                         lan_perf_re.search(f)]
    lan_dfs_d = [pd.read_csv(f) for f in lan_files_d]
    lan_df_d = pd.concat(lan_dfs_d)
    lan_df_d.to_csv("Joined Runs/disp_RF_equal_LANGUAGE_UNDER_joined.csv", 
                    index=False)
    
    # Sex   -   GENDER has 900 runs instead of 800
    # Performance
    sex_perf_re = re.compile("performance_RF_equal_GENDER_2026")
    sex_files_p = ["All Runs Raw/"+f for f in os.listdir("All Runs Raw") if \
                                                         sex_perf_re.search(f)]
    sex_dfs_p = [pd.read_csv(f) for f in sex_files_p]
    sex_df_p = pd.concat(sex_dfs_p)
    sex_df_p.to_csv("Joined Runs/perf_RF_equal_GENDER_UNDER_joined.csv", 
                    index=False)
    # Disparities
    sex_perf_re = re.compile("disparities_RF_equal_GENDER_2026")
    sex_files_d = ["All Runs Raw/"+f for f in os.listdir("All Runs Raw") if \
                                                         sex_perf_re.search(f)]
    sex_dfs_d = [pd.read_csv(f) for f in sex_files_d]
    sex_df_d = pd.concat(sex_dfs_d)
    sex_df_d.to_csv("Joined Runs/disp_RF_equal_GENDER_UNDER_joined.csv", 
                    index=False)
    
    # SVI   -   RACE has 631 runs instead of 800
    # Performance
    svi_perf_re = re.compile("performance_RF_equal_SVI_QUARTILE_2026")
    svi_files_p = ["All Runs Raw/"+f for f in os.listdir("All Runs Raw") if \
                                                         svi_perf_re.search(f)]
    svi_dfs_p = [pd.read_csv(f) for f in svi_files_p]
    svi_df_p = pd.concat(svi_dfs_p)
    svi_df_p.to_csv("Joined Runs/perf_RF_equal_SVI_QUARTILE_UNDER_joined.csv", 
                    index=False)
    # Disparities
    svi_perf_re = re.compile("disparities_RF_equal_SVI_QUARTILE_2026")
    svi_files_d = ["All Runs Raw/"+f for f in os.listdir("All Runs Raw") if \
                                                         svi_perf_re.search(f)]
    svi_dfs_d = [pd.read_csv(f) for f in svi_files_d]
    svi_df_d = pd.concat(svi_dfs_d)
    svi_df_d.to_csv("Joined Runs/disp_RF_equal_SVI_QUARTILE_UNDER_joined.csv", 
                    index=False)