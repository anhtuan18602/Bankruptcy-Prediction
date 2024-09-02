from scipy.stats.mstats import winsorize
import pandas as pd
import numpy as np

def min_max_normalize(df, columns):
    result = df.copy()
    for column in columns:
        min_value = df[column].min()
        max_value = df[column].max()
        result[column] = (df[column] - min_value) / (max_value - min_value)
    return result

def calculate_growth(df, val_name):
    dt = df.copy()
    dt[f'G_{val_name}'] = (dt[val_name] - dt[val_name].shift(1)) / dt[val_name].shift(1)
    return dt   

def calculate_change(df, val_name):
    dt = df.copy()
    dt[f'C_{val_name}'] = (dt[val_name] - dt[val_name].shift(1))
    return dt 

def calculate_avg(df, val_name):
    dt = df.copy()
    dt[f'AVG_{val_name}'] = (dt[val_name] + dt[val_name].shift(1)) / 2
    return dt        

def winsorize_series(series, winsorization_limits= (0.03,0.03)):
    return winsorize(series, limits=winsorization_limits)
def winsorize_df(df,columns):
    winsorized_df = df.copy()
    for column in columns:
        winsorized_df[column] = winsorize_series(df[column])
    return winsorized_df