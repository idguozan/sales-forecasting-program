#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forecasting models module.
Contains simple and ML-based forecasting functions.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Dict
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..config import (
    TARGET_COL, FORECAST_HORIZON_WEEKS, FEATURE_COLS, 
    LIBRARY_AVAILABILITY
)
from ..utils.metrics import log_output, calculate_mape, calculate_wape
from ..data_loader import prepare_weekly_data
from ..feature_engineering import create_features

# Import optional libraries based on availability
if LIBRARY_AVAILABILITY['xgboost']:
    import xgboost as xgb  # type: ignore

if LIBRARY_AVAILABILITY['catboost']:
    import catboost as cb  # type: ignore

if LIBRARY_AVAILABILITY['prophet']:
    from prophet import Prophet  # type: ignore

# ============================================================================
# Simple Forecasting
# ============================================================================

def simple_forecast_for_sheet(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """
    Simple forecasting method for small datasets (<100 rows).
    Uses recent trend and seasonal patterns.
    """
    log_output(f"🔮 {sheet_name}: Applying simple forecasting method...")
    
    # Prepare weekly data
    weekly_data = prepare_weekly_data(df, sheet_name)
    
    if len(weekly_data) < 4:
        log_output(f"⚠️  {sheet_name}: Insufficient data, using constant forecast")
        last_value = weekly_data[TARGET_COL].iloc[-1] if len(weekly_data) > 0 else 1000
        
        # Create future weeks
        last_week = weekly_data["week"].max() if len(weekly_data) > 0 else pd.Timestamp.now()
        future_weeks = [last_week + timedelta(weeks=i) for i in range(1, FORECAST_HORIZON_WEEKS + 1)]
        
        forecast_df = pd.DataFrame({
            "week": future_weeks,
            "forecast": [last_value] * FORECAST_HORIZON_WEEKS,
            "method": "constant",
            "sheet_name": sheet_name
        })
        
        return forecast_df
    
    # Calculate trend and seasonality
    recent_data = weekly_data.tail(8)  # Last 8 weeks
    recent_mean = recent_data[TARGET_COL].mean()
    
    # Simple trend calculation
    if len(recent_data) >= 4:
        first_half = recent_data[TARGET_COL].iloc[:len(recent_data)//2].mean()
        second_half = recent_data[TARGET_COL].iloc[len(recent_data)//2:].mean()
        trend = (second_half - first_half) / (len(recent_data)//2)
    else:
        trend = 0
    
    # Create future forecasts
    last_week = weekly_data["week"].max()
    future_weeks = [last_week + timedelta(weeks=i) for i in range(1, FORECAST_HORIZON_WEEKS + 1)]
    
    forecasts = []
    for i, week in enumerate(future_weeks):
        # Base forecast with trend
        base_forecast = recent_mean + (trend * (i + 1))
        
        # Add seasonal component (very simple)
        week_of_year = week.isocalendar().week
        seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * week_of_year / 52)
        
        forecast_value = max(0, base_forecast * seasonal_factor)
        forecasts.append(forecast_value)
    
    forecast_df = pd.DataFrame({
        "week": future_weeks,
        "forecast": forecasts,
        "method": "simple_trend_seasonal",
        "sheet_name": sheet_name
    })
    
    log_output(f"✅ {sheet_name}: Simple forecasting completed. Average forecast: {np.mean(forecasts):.2f}")
    return forecast_df

# ============================================================================
# ML-based Forecasting
# ============================================================================

def ml_forecast_for_sheet(df: pd.DataFrame, sheet_name: str) -> Dict[str, pd.DataFrame]:
    """
    Advanced ML-based forecasting for larger datasets (>=100 rows).
    """
    log_output(f"🤖 {sheet_name}: Applying advanced ML forecasting models...")
    
    # Prepare weekly data
    weekly_data = prepare_weekly_data(df, sheet_name)
    
    if len(weekly_data) < 20:
        log_output(f"⚠️  {sheet_name}: Insufficient weekly data for ML, using simple forecast")
        simple_forecast = simple_forecast_for_sheet(df, sheet_name)
        return {
            "rf": simple_forecast.copy(),
            "et": simple_forecast.copy(),
            "gb": simple_forecast.copy()
        }
    
    # Create advanced features
    weekly_model = create_features(weekly_data)
    
    # Train/Test split
    max_week = weekly_model["week"].max()
    cutoff_week = max_week - pd.Timedelta(weeks=min(FORECAST_HORIZON_WEEKS, len(weekly_model)//4))
    
    train = weekly_model[weekly_model["week"] <= cutoff_week].copy()
    test = weekly_model[weekly_model["week"] > cutoff_week].copy()
    
    if len(train) < 10:
        log_output(f"⚠️  {sheet_name}: Insufficient data for training")
        simple_forecast = simple_forecast_for_sheet(df, sheet_name)
        return {
            "rf": simple_forecast.copy(),
            "et": simple_forecast.copy(),
            "gb": simple_forecast.copy()
        }
    
    # Feature selection - use only available features
    available_features = [col for col in FEATURE_COLS if col in weekly_model.columns]
    X_train = train[available_features]
    y_train = train[TARGET_COL]
    
    # Train models and evaluate
    models = {}
    backtest_results = {}
    
    # RandomForest with optimized parameters
    log_output(f"🌲 {sheet_name}: Training Optimized RandomForest...")
    rf_model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42, 
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models["rf"] = rf_model
    
    # Evaluate on test set if available
    if len(test) > 0:
        X_test = test[available_features]
        y_test = test[TARGET_COL]
        rf_pred = rf_model.predict(X_test)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_mape = calculate_mape(y_test, rf_pred)
        rf_wape = calculate_wape(y_test, rf_pred)
        backtest_results["rf"] = {"MAE": rf_mae, "RMSE": rf_rmse, "MAPE": rf_mape, "WAPE": rf_wape}
        log_output(f"📊 {sheet_name}: RF Backtest - MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}, MAPE: {rf_mape:.1f}%, WAPE: {rf_wape:.1f}%")
    
    # ExtraTrees with optimized parameters
    log_output(f"🌳 {sheet_name}: Training Optimized ExtraTrees...")
    et_model = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42, 
        n_jobs=-1
    )
    et_model.fit(X_train, y_train)
    models["et"] = et_model
    
    # Evaluate on test set if available
    if len(test) > 0:
        et_pred = et_model.predict(X_test)
        et_mae = mean_absolute_error(y_test, et_pred)
        et_rmse = np.sqrt(mean_squared_error(y_test, et_pred))
        et_mape = calculate_mape(y_test, et_pred)
        et_wape = calculate_wape(y_test, et_pred)
        backtest_results["et"] = {"MAE": et_mae, "RMSE": et_rmse, "MAPE": et_mape, "WAPE": et_wape}
        log_output(f"📊 {sheet_name}: ET Backtest - MAE: {et_mae:.2f}, RMSE: {et_rmse:.2f}, MAPE: {et_mape:.1f}%, WAPE: {et_wape:.1f}%")
    
    # GradientBoosting with optimized parameters
    log_output(f"⚡ {sheet_name}: Training Optimized GradientBoosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    models["gb"] = gb_model
    
    # Evaluate on test set if available
    if len(test) > 0:
        gb_pred = gb_model.predict(X_test)
        gb_mae = mean_absolute_error(y_test, gb_pred)
        gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
        gb_mape = calculate_mape(y_test, gb_pred)
        gb_wape = calculate_wape(y_test, gb_pred)
        backtest_results["gb"] = {"MAE": gb_mae, "RMSE": gb_rmse, "MAPE": gb_mape, "WAPE": gb_wape}
        log_output(f"📊 {sheet_name}: GB Backtest - MAE: {gb_mae:.2f}, RMSE: {gb_rmse:.2f}, MAPE: {gb_mape:.1f}%, WAPE: {gb_wape:.1f}%")
    
    # XGBoost if available
    if LIBRARY_AVAILABILITY['xgboost']:
        log_output(f"🚀 {sheet_name}: Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        models["xgb"] = xgb_model
        
        if len(test) > 0:
            xgb_pred = xgb_model.predict(X_test)
            xgb_mae = mean_absolute_error(y_test, xgb_pred)
            xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
            xgb_mape = calculate_mape(y_test, xgb_pred)
            xgb_wape = calculate_wape(y_test, xgb_pred)
            backtest_results["xgb"] = {"MAE": xgb_mae, "RMSE": xgb_rmse, "MAPE": xgb_mape, "WAPE": xgb_wape}
            log_output(f"📊 {sheet_name}: XGB Backtest - MAE: {xgb_mae:.2f}, RMSE: {xgb_rmse:.2f}, MAPE: {xgb_mape:.1f}%, WAPE: {xgb_wape:.1f}%")
    
    # CatBoost if available
    if LIBRARY_AVAILABILITY['catboost']:
        log_output(f"🐱 {sheet_name}: Training CatBoost...")
        cat_model = cb.CatBoostRegressor(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            verbose=False,
            random_state=42
        )
        cat_model.fit(X_train, y_train)
        models["cat"] = cat_model
        
        if len(test) > 0:
            cat_pred = cat_model.predict(X_test)
            cat_mae = mean_absolute_error(y_test, cat_pred)
            cat_rmse = np.sqrt(mean_squared_error(y_test, cat_pred))
            cat_mape = calculate_mape(y_test, cat_pred)
            cat_wape = calculate_wape(y_test, cat_pred)
            backtest_results["cat"] = {"MAE": cat_mae, "RMSE": cat_rmse, "MAPE": cat_mape, "WAPE": cat_wape}
            log_output(f"📊 {sheet_name}: CAT Backtest - MAE: {cat_mae:.2f}, RMSE: {cat_rmse:.2f}, MAPE: {cat_mape:.1f}%, WAPE: {cat_wape:.1f}%")
    
    # Print best model
    if len(test) > 0 and backtest_results:
        best_model = min(backtest_results.keys(), key=lambda x: backtest_results[x]["MAE"])
        log_output(f"🏆 {sheet_name}: Best model: {best_model.upper()} (MAE: {backtest_results[best_model]['MAE']:.2f})")
    
    # Generate forecasts
    forecasts = _generate_ml_forecasts(models, weekly_model, available_features, sheet_name)
    
    # Add backtest results to forecasts
    for model_name in forecasts:
        if model_name in backtest_results:
            forecasts[model_name].attrs = backtest_results[model_name]
    
    return forecasts

def _generate_ml_forecasts(models: Dict, weekly_model: pd.DataFrame, available_features: list, sheet_name: str) -> Dict[str, pd.DataFrame]:
    """Helper function to generate ML forecasts for all models"""
    forecasts = {}
    last_week = weekly_model["week"].max()
    
    for model_name, model in models.items():
        future_weeks = []
        future_forecasts = []
        
        # Get the last row features for starting point
        last_features = weekly_model.iloc[-1][available_features].copy()
        
        # Store recent values for lag updates
        recent_values = weekly_model[TARGET_COL].tail(12).tolist()
        
        for i in range(1, FORECAST_HORIZON_WEEKS + 1):
            future_week = last_week + pd.Timedelta(weeks=i)
            
            # Create features for future week - only use available features
            future_features_dict = {}
            
            # Time features
            if "week_of_year" in available_features:
                future_features_dict["week_of_year"] = future_week.isocalendar().week
            if "month" in available_features:
                future_features_dict["month"] = future_week.month
            if "quarter" in available_features:
                future_features_dict["quarter"] = future_week.quarter
            if "day_of_year" in available_features:
                future_features_dict["day_of_year"] = future_week.dayofyear
            
            # Rolling features - use recent values
            for window in [4, 8, 12]:
                if f"rolling_mean_{window}" in available_features:
                    recent_window = recent_values[-window:] if len(recent_values) >= window else recent_values
                    future_features_dict[f"rolling_mean_{window}"] = np.mean(recent_window)
                
                if f"rolling_std_{window}" in available_features:
                    recent_window = recent_values[-window:] if len(recent_values) >= window else recent_values
                    future_features_dict[f"rolling_std_{window}"] = np.std(recent_window) if len(recent_window) > 1 else 0
                
                if f"rolling_median_{window}" in available_features:
                    recent_window = recent_values[-window:] if len(recent_values) >= window else recent_values
                    future_features_dict[f"rolling_median_{window}"] = np.median(recent_window)
            
            # Lag features
            for lag in [1, 2, 4, 8]:
                if f"lag_{lag}" in available_features:
                    if len(recent_values) >= lag:
                        future_features_dict[f"lag_{lag}"] = recent_values[-lag]
                    else:
                        future_features_dict[f"lag_{lag}"] = recent_values[-1] if recent_values else 0
            
            # Seasonal features
            if "weekly_cycle_sin" in available_features:
                future_features_dict["weekly_cycle_sin"] = np.sin(2 * np.pi * future_week.isocalendar().week / 52)
            if "weekly_cycle_cos" in available_features:
                future_features_dict["weekly_cycle_cos"] = np.cos(2 * np.pi * future_week.isocalendar().week / 52)
            if "monthly_cycle_sin" in available_features:
                future_features_dict["monthly_cycle_sin"] = np.sin(2 * np.pi * future_week.month / 12)
            if "monthly_cycle_cos" in available_features:
                future_features_dict["monthly_cycle_cos"] = np.cos(2 * np.pi * future_week.month / 12)
            
            # EMA features - estimate based on recent trend
            if "ema_3" in available_features:
                future_features_dict["ema_3"] = np.mean(recent_values[-3:]) if len(recent_values) >= 3 else np.mean(recent_values)
            if "ema_7" in available_features:
                future_features_dict["ema_7"] = np.mean(recent_values[-7:]) if len(recent_values) >= 7 else np.mean(recent_values)
            if "ema_14" in available_features:
                future_features_dict["ema_14"] = np.mean(recent_values[-14:]) if len(recent_values) >= 14 else np.mean(recent_values)
            
            # Fill missing features with 0
            for feature in available_features:
                if feature not in future_features_dict:
                    future_features_dict[feature] = 0
            
            # Create feature vector
            future_features = pd.Series([future_features_dict[f] for f in available_features], index=available_features)
            
            # Make prediction
            prediction = model.predict(future_features.values.reshape(1, -1))[0]
            prediction = max(0, prediction)  # Ensure non-negative
            
            future_weeks.append(future_week)
            future_forecasts.append(prediction)
            
            # Update recent values with prediction for next iteration
            recent_values.append(prediction)
            if len(recent_values) > 12:
                recent_values.pop(0)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            "week": future_weeks,
            "forecast": future_forecasts,
            "method": f"ml_{model_name}",
            "sheet_name": sheet_name
        })
        
        forecasts[model_name] = forecast_df
        
        log_output(f"✅ {sheet_name}: {model_name.upper()} forecast completed. Average: {np.mean(future_forecasts):.2f}")
    
    return forecasts
