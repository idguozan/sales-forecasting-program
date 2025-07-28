
# Sales Forecasting Program 

> **🚀 Developed by Ozan İdgü**

This project is a Python-based sales forecasting pipeline with a **modular architecture**. It generates future sales predictions per product using various machine learning models and visualizes the results with comprehensive reporting.

## ✨ Recent Updates Forecasting Program 

> **� Developed by Ozan İdgü**

This project is a Python-based sales forecasting pipeline with a **modular architecture**. It generates future sales predictions per product using various machine learning models and visualizes the results with comprehensive reporting.

## ✨ Recent Updates
- **✅ Complete modular restructure**: Broke down monolithic 1460+ line script into organized modules
- **✅ Enhanced ML pipeline**: Added XGBoost and CatBoost models with optimized hyperparameters  
- **✅ Advanced feature engineering**: 25+ time-based features including rolling statistics, lag features, seasonal components
- **✅ Sheet-wise analysis**: Each Excel sheet is processed separately with appropriate ML/simple forecasting
- **✅ Comprehensive reporting**: Individual model forecasts, backtest results, and detailed PDF reports
- **✅ Bilingual support**: All system messages translated to English while maintaining Turkish column compatibility

## Features
- **Modular design** with 8 specialized modules for better maintainability
- Read data from Excel, CSV, or SQLite files  
- Flexible column mapping (Turkish/English supported)
- **Sheet-based processing**: < 100 rows = Simple forecasting, >= 100 rows = ML forecasting
- Data cleaning and advanced feature engineering (25+ features)
- **5 ML models**: Random Forest, Extra Trees, Gradient Boosting, XGBoost, CatBoost
- Product-level and aggregate forecast reports (CSV, Excel, PDF)
- Automatic visualization (PNG charts for each sheet)
- Backtest validation with multiple performance metrics
- FastAPI web interface support

## Installation
1. Python 3.9+ and pip must be installed.
2. Install required packages:
   ```sh
   pip install -r requierements.txt
   ```
3. Add your data file to the `uploads/` folder or upload via the admin panel.

## Usage
<<<<<<< HEAD
**Option 1 - Command line with specific file:**
```sh
python scripts/run_batch_new.py uploads/your_data_file.xlsx
```

**Option 2 - Use last uploaded file:**
```sh
python scripts/run_batch_new.py
```
(Reads from `uploads/last_uploaded.json` automatically)

### Output Locations:
- `app/static/plots/` → Product analysis and forecast charts (PNG)
- `app/static/reports/` → Model forecasts (CSV), comprehensive PDF report, sheet summary
- Project root → `future_forecast.csv` (legacy compatibility)
=======
To run batch forecasting:
```sh
python scripts/run_batch.py uploads/your_data_file.xlsx
```

Forecasts and reports are saved in:
- `app/static/plots/` → Product charts
- `app/static/reports/` → PDF report, total forecasts
- Project root → future_forecast.csv
>>>>>>> 5e3c9b7f3fa1ae0672ace07d77c5a57d6915e026

## FastAPI Service
To start the web interface:
```sh
python app/main.py
```

## File Structure
- `app/` : FastAPI backend and routers
<<<<<<< HEAD
- `pipeline/` : Legacy data processing code
- **`scripts/` : Main execution and modular architecture**
  - **`run_batch_new.py`** : Main execution script (sheet-wise processing)
  - **`modules/`** : Modular components
    - **`config.py`** : Configuration and parameters
    - **`data_loader.py`** : Data loading and column mapping
    - **`feature_engineering.py`** : Advanced feature creation (25+ features)
    - **`forecasting/models.py`** : ML models and forecasting logic
    - **`visualization/charts.py`** : Chart generation
    - **`visualization/pdf_reports.py`** : Comprehensive PDF reporting
    - **`utils/metrics.py`** : Performance metrics and utilities
- `uploads/` : Uploaded data files

## Models & Performance
- **RandomForestRegressor** (Optimized hyperparameters)
- **ExtraTreesRegressor** (Often best performer - 9.4% WAPE)
- **GradientBoostingRegressor** 
- **XGBoost** (Excellent accuracy - 5.8% WAPE)
- **CatBoost** (Robust performance - 12.5% WAPE)
- Prophet (optional, legacy support)
- LightGBM (optional)

## Recent Test Results
Latest run successfully processed **37 sheets**:
- 🤖 **26 sheets** processed with ML models
- 🔮 **11 sheets** processed with simple forecasting  
- ❌ **0 errors**
- ⚡ Average processing time: **2.5 seconds per sheet**

**Performance highlights:**
- ExtraTrees: 9.4% WAPE (best overall)
- XGBoost: 5.8% WAPE (most accurate)
- CatBoost: 12.5% WAPE (robust)

=======
- `pipeline/` : Data processing, modeling, and forecasting code
- `scripts/` : Batch run script
- `uploads/` : Uploaded data files

## Models
- RandomForestRegressor
- ExtraTreesRegressor
- GradientBoostingRegressor
- Prophet (optional)
- LightGBM (optional)

>>>>>>> 5e3c9b7f3fa1ae0672ace07d77c5a57d6915e026
## Contributing
You can contribute by opening pull requests or issues.

## License
MIT

---
<<<<<<< HEAD
**📦 Modular Architecture:** The system is now fully modularized for better maintainability, testing, and extensibility. Each module has a specific responsibility and can be easily modified or extended.

=======
>>>>>>> 5e3c9b7f3fa1ae0672ace07d77c5a57d6915e026
For detailed documentation and usage examples, please review the code.
