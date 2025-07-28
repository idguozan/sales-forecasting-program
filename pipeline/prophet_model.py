# pipeline/prophet_model.py
from prophet import Prophet
import pandas as pd


def forecast_product_sales(df: pd.DataFrame, product_code, weeks: int = 10) -> pd.DataFrame:
    """
    Generates weekly sales forecasts for the given product code using Prophet.
    Column names are flexible: ('week','qty') or ('Week','Quantity') are accepted.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least one of the following pairs of columns:
            - week + qty
            - Week + Quantity
        Also must contain 'StockCode' (product code).
    product_code :
        Product code to forecast (string or number).
    weeks : int
        Number of weeks to forecast ahead.

    Returns
    -------
    pd.DataFrame
        Columns: ds, yhat, yhat_lower, yhat_upper
        (Prophet forecast standard)
    """
    # Column mapping
    cols_lower = {c.lower(): c for c in df.columns}
    week_col = cols_lower.get('week')
    qty_col = cols_lower.get('qty')

    # Alternative uppercase names
    if week_col is None and 'Week' in df.columns:
        week_col = 'Week'
    if qty_col is None and 'Quantity' in df.columns:
        qty_col = 'Quantity'

    if week_col is None or qty_col is None:
        raise ValueError(
            "Could not find week (week/Week) and quantity (qty/Quantity) columns in DataFrame."
        )

    # Selected product
    product_df = df[df["StockCode"] == product_code].copy()
    if product_df.empty:
        # If no data, return empty forecast (do not break main flow)
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])

    # Weekly aggregation (if already weekly, this will aggregate again but it's harmless)
    weekly_sales = (
        product_df.groupby(week_col)[qty_col]
        .sum()
        .reset_index()
        .sort_values(week_col)
    )

    # Prophet format
    weekly_sales = weekly_sales.rename(columns={week_col: "ds", qty_col: "y"})

    # Convert date column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(weekly_sales['ds']):
        weekly_sales['ds'] = pd.to_datetime(weekly_sales['ds'])

    # Prophet model
    m = Prophet()
    m.fit(weekly_sales)

    # Future dataframe
    future = m.make_future_dataframe(periods=weeks, freq="W")
    forecast = m.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
