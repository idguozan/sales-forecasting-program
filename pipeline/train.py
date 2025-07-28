# pipeline/train.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_model(
    weekly_model: pd.DataFrame,
    feature_cols,
    target_col: str = 'qty',
    forecast_horizon_weeks: int = 12,
):
    """
    Train/Test split + RF training + metrics as in the monolithic code.
    Returns: model, X_test, y_test, metrics(dict)
    """
    print("Performing Train/Test split (last 12 weeks as test)...")
    max_week = weekly_model['week'].max()
    cutoff_week = max_week - pd.Timedelta(weeks=forecast_horizon_weeks)

    train = weekly_model[weekly_model['week'] <= cutoff_week].copy()
    test  = weekly_model[weekly_model['week'] >  cutoff_week].copy()

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test  = test[feature_cols]
    y_test  = test[target_col]

    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("Calculating backtest predictions and metrics (last 12 weeks)...")
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print("\n--- Backtest Results ---")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    metrics = {"mae": mae, "rmse": rmse, "mape": mape}
    return model, X_test, y_test, metrics
