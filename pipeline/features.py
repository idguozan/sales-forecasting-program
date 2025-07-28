# pipeline/features.py
def create_features(data):
    """
    Feature engineering identical to the monolithic code.
    """
    data = data.copy()
    data['month'] = data['week'].dt.month
    data['year'] = data['week'].dt.year
    data['week_no'] = data['week'].dt.isocalendar().week.astype(int)

    # Lag features
    for lag in [1, 4, 8]:
        data[f'qty_lag_{lag}'] = data.groupby('StockCode')['qty'].shift(lag)

    # Price change + promo flag
    data['price_diff'] = data.groupby('StockCode')['avg_price'].diff()
    data['promo_flag'] = (data['price_diff'] < 0).astype(int)

    return data
