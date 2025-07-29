#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature engineering module for the sales forecasting system.
Contains functions for creating advanced time-based features.
"""

import numpy as np
import pandas as pd
from .config import TARGET_COL

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced time-based features for modeling.
    Enhanced with outlier handling.
    """
    df = df.copy()
    df = df.sort_values("week").reset_index(drop=True)
    
    # Simple outlier handling using IQR method
    Q1 = df[TARGET_COL].quantile(0.25)
    Q3 = df[TARGET_COL].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers (preserve time series structure)
    df.loc[df[TARGET_COL] < lower_bound, TARGET_COL] = lower_bound
    df.loc[df[TARGET_COL] > upper_bound, TARGET_COL] = upper_bound
    
    # Basic time features
    df["week_of_year"] = df["week"].dt.isocalendar().week
    df["month"] = df["week"].dt.month
    df["quarter"] = df["week"].dt.quarter
    df["day_of_year"] = df["week"].dt.dayofyear
    
    # Advanced rolling features
    windows = [2, 3, 4, 6, 8, 12]
    for window in windows:
        df[f"rolling_mean_{window}"] = df[TARGET_COL].rolling(window=window, min_periods=1).mean()
        df[f"rolling_std_{window}"] = df[TARGET_COL].rolling(window=window, min_periods=1).std()
        df[f"rolling_median_{window}"] = df[TARGET_COL].rolling(window=window, min_periods=1).median()
        df[f"rolling_max_{window}"] = df[TARGET_COL].rolling(window=window, min_periods=1).max()
        df[f"rolling_min_{window}"] = df[TARGET_COL].rolling(window=window, min_periods=1).min()
    
    # Extended lag features
    for lag in [1, 2, 3, 4, 6, 8, 12]:
        df[f"lag_{lag}"] = df[TARGET_COL].shift(lag)
    
    # Trend and momentum features
    df["diff_1"] = df[TARGET_COL].diff(1)
    df["diff_2"] = df[TARGET_COL].diff(2)
    df["pct_change_1"] = df[TARGET_COL].pct_change(1)
    df["pct_change_4"] = df[TARGET_COL].pct_change(4)
    
    # Seasonal features
    df["weekly_cycle_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["weekly_cycle_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["monthly_cycle_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["monthly_cycle_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Exponential moving averages
    df["ema_3"] = df[TARGET_COL].ewm(span=3, adjust=False).mean()
    df["ema_7"] = df[TARGET_COL].ewm(span=7, adjust=False).mean()
    df["ema_14"] = df[TARGET_COL].ewm(span=14, adjust=False).mean()
    
    # Statistical features
    df["z_score_4"] = (df[TARGET_COL] - df["rolling_mean_4"]) / (df["rolling_std_4"] + 1e-8)
    df["z_score_8"] = (df[TARGET_COL] - df["rolling_mean_8"]) / (df["rolling_std_8"] + 1e-8)
    
    # Interaction features
    df["lag1_x_month"] = df["lag_1"] * df["month"]
    df["rolling_mean_4_x_trend"] = df["rolling_mean_4"] * df["diff_1"]
    
    # Fill NaN values with advanced methods
    # Forward fill first, then backward fill, then fill with 0
    df = df.ffill().bfill().fillna(0)  # New pandas syntax
    
    return df
