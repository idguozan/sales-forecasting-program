# pipeline/etl.py
import pandas as pd

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    print("Loading data...")
    df = pd.read_excel(filepath)

    print("Cleaning data...")
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Weekly period start
    df['week'] = df['InvoiceDate'].dt.to_period('W').dt.start_time

    # Row-level total
    df['line_total'] = df['Quantity'] * df['UnitPrice']
    return df
