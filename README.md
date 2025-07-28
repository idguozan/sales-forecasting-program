
# Sales Forecasting Program

This project is a Python-based sales forecasting pipeline. It generates future sales predictions per product using various machine learning models (Random Forest, Extra Trees, Gradient Boosting, Prophet) and visualizes the results.

## Features
- Read data from Excel, CSV, or SQLite files
- Flexible column mapping (Turkish/English supported)
- Data cleaning and feature engineering
- Forecasting with multiple regression models
- Product-level and total forecast reports (CSV, Excel, PDF)
- Automatic visualization (PNG charts, bar charts)
- FastAPI web interface support

## Installation
1. Python 3.9+ and pip must be installed.
2. Install required packages:
   ```sh
   pip install -r requierements.txt
   ```
3. Add your data file to the `uploads/` folder or upload via the admin panel.

## Usage
To run batch forecasting:
```sh
python scripts/run_batch.py uploads/your_data_file.xlsx
```

Forecasts and reports are saved in:
- `app/static/plots/` → Product charts
- `app/static/reports/` → PDF report, total forecasts
- Project root → future_forecast.csv

## FastAPI Service
To start the web interface:
```sh
python app/main.py
```

## File Structure
- `app/` : FastAPI backend and routers
- `pipeline/` : Data processing, modeling, and forecasting code
- `scripts/` : Batch run script
- `uploads/` : Uploaded data files

## Models
- RandomForestRegressor
- ExtraTreesRegressor
- GradientBoostingRegressor
- Prophet (optional)
- LightGBM (optional)

## Contributing
You can contribute by opening pull requests or issues.

## License
MIT

---
For detailed documentation and usage examples, please review the code.
