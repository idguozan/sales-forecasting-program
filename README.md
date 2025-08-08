
# Sales Forecasting Program

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-009688?style=flat-square)
![Machine Learning](https://img.shields.io/badge/ML-6%20Models-orange?style=flat-square)

### A comprehensive sales forecasting system that performs automatic sales prediction using 6 different machine learning algorithms with advanced feature engineering and statistical modeling.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models & Performance](#models--performance)
- [Results](#results)
- [Testing](#testing)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)

---

## Features

| **Machine Learning Models** | **Advanced Analytics** | **Rich Outputs** |
|---|---|---|
| **Random Forest** - Ensemble learning with optimized hyperparameters | **Advanced Feature Engineering** - 25+ time-based features including rolling statistics | **Interactive Dashboard** - FastAPI web interface for data upload and analysis |
| **Extra Trees** - Often best performer (9.4% WAPE) | **Sheet-wise Processing** - Automatic detection: <100 rows = Simple forecasting, >=100 rows = ML | **Comprehensive PDF Reports** - Detailed analysis with charts and model performance |
| **Gradient Boosting** - Robust performance with learning rate optimization | **Cross-Validation** - 5-fold stratified CV for reliable model evaluation | **Individual Model Forecasts** - CSV outputs for each model |
| **XGBoost** - Most accurate model (5.8% WAPE) | **Hyperparameter Optimization** - Adaptive parameters based on data size | **Product-level Charts** - PNG visualizations for each product category |
| **CatBoost** - Excellent categorical feature handling | **Conservative Optimization** - IQR outlier handling for robust predictions | **Performance Metrics** - WAPE, MAE, RMSE with detailed backtest results |
| **Improved Statistical Model** - Hybrid approach with weekly patterns | **Multiple Data Formats** - Excel, CSV, SQLite support with flexible column mapping | **Legacy Compatibility** - Maintains backward compatibility with existing systems |

---

## Quick Start

Get started in less than 5 minutes!

```bash
# Clone the repository
git clone https://github.com/idguozan/sales-prediction-program.git
cd sales-prediction-program

# Install dependencies
pip install -r requirements.txt

# Run analysis with your data
python scripts/run_batch_new.py uploads/your_data_file.xlsx
```

That's it! Your analysis results will be saved in the `app/static/reports/` directory.

---

## Installation

### Prerequisites

- Python 3.9+
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/idguozan/sales-prediction-program.git
cd sales-prediction-program
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

---

## Usage

### Command Line Interface

**Option 1 - With specific file:**
```bash
python scripts/run_batch_new.py uploads/your_data_file.xlsx
```

**Option 2 - Use last uploaded file:**
```bash
python scripts/run_batch_new.py
```
(Reads from `uploads/last_uploaded.json` automatically)

### FastAPI Web Interface

```bash
# Start the web server
python app/main.py

# Open browser and navigate to:
# http://localhost:8000
```

### Supported Data Formats

- **.xlsx / .xls** - Excel files with multiple sheets
- **.csv** - Comma-separated values
- **.sql** - SQLite script files
- **.sqlite / .db** - SQLite database files

---

## Project Structure

```
sales-prediction-program/
│
├── 📊 app/
│   ├── main.py                     # FastAPI application
│   ├── routers/                    # API endpoints
│   └── static/
│       ├── plots/                  # Generated charts (PNG)
│       └── reports/                # Analysis results (PDF, CSV)
│
├── 🧪 scripts/
│   ├── run_batch_new.py           # Main execution script
│   └── modules/
│       ├── config.py              # Configuration parameters
│       ├── data_loader.py         # Data processing and loading
│       ├── feature_engineering.py # Advanced feature creation
│       ├── forecasting/
│       │   └── models.py          # ML models and forecasting
│       ├── improved_forecasting/
│       │   └── models.py          # Statistical hybrid models
│       ├── utils/
│       │   └── metrics.py         # Performance evaluation
│       └── visualization/
│           ├── charts.py          # Chart generation
│           └── pdf_reports.py     # PDF report creation
│
├── 🔧 pipeline/                   # Legacy pipeline (maintained for compatibility)
├── 🧪 tests/                      # Comprehensive test suite
├── 📁 uploads/                    # Data upload directory
├── 📋 requirements.txt            # Python dependencies
└── 📖 README.md                   # This file
```

---

## Models & Performance

| Model | Type | Accuracy (WAPE) | Key Strengths |
|-------|------|----------------|---------------|
| **Extra Trees** | Ensemble | 9.4% | Most stable performance across products |
| **XGBoost** | Gradient Boosting | 5.8% | Highest accuracy, excellent for complex patterns |
| **CatBoost** | Gradient Boosting | 12.5% | Robust categorical handling, minimal preprocessing |
| **Random Forest** | Ensemble | Variable | Good baseline, feature importance insights |
| **Gradient Boosting** | Ensemble | Variable | Solid performance, learning rate optimization |
| **Improved Statistical** | Hybrid | 70.9% accuracy | Pattern-based approach with weekly variations |

### Recent Performance Highlights

- **Conservative Optimization**: 19.5% MAE improvement (176.2 → 141.85)
- **Processing Speed**: Average 2.5 seconds per sheet
- **Success Rate**: 100% (37 sheets processed, 0 errors)
- **Feature Engineering**: 25+ time-based features with rolling statistics

---

## Results

### Sample Performance Output

```
MODEL PERFORMANCE SUMMARY:
================================================
Model                Accuracy  WAPE    MAE      RMSE
Extra Trees         91.8%     9.4%    141.85   245.2
XGBoost            94.2%     5.8%    128.94   198.7
CatBoost           87.5%     12.5%   167.34   298.1
Random Forest      89.6%     10.4%   155.67   267.8
```

### Output Files

| File Type | Location | Description |
|-----------|----------|-------------|
| **PDF Report** | `app/static/reports/` | Comprehensive analysis with charts |
| **Model Forecasts** | `app/static/reports/` | Individual CSV files for each model |
| **Charts** | `app/static/plots/` | Product-level PNG visualizations |
| **Summary** | `app/static/reports/` | Sheet-wise processing summary |

---

## Testing

This project includes a comprehensive test suite with 32 automated tests.

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=scripts --cov-report=html

# Run specific test categories
python -m pytest tests/unit/ -v       # Unit tests only
python -m pytest tests/integration/ -v  # Integration tests only
```

### Test Coverage

- **32/32 tests passing** (100% success rate)
- **Unit Tests**: 25 tests for individual modules
- **Integration Tests**: 7 tests for complete pipeline
- **Coverage**: Comprehensive module coverage
- **Isolation**: Tests don't affect main system outputs

---

## API Documentation

The FastAPI web interface provides:

- **Data Upload**: Support for multiple file formats
- **Real-time Processing**: Progress tracking and results
- **Interactive Dashboards**: Model comparison and analysis
- **Download Results**: PDF reports and CSV forecasts

Access the interactive API documentation at `http://localhost:8000/docs` when running the web server.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

---

## License


This project is licensed under the GNU General Public License v3.0 (GPLv3). See the [LICENSE](LICENSE) file for details.

---

## Authors

**Developed by** - [Ozan İdgü](https://github.com/idguozan)

---

## Development Timeline

This project has been in active development with continuous improvements in machine learning accuracy, modular architecture, and comprehensive testing capabilities.

---

## Acknowledgments

- Open source Python ecosystem
- Scikit-learn and XGBoost communities
- FastAPI framework contributors
- Machine learning research community

---

### If you find this project useful, please consider giving it a star!

![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?style=for-the-badge&logo=python)
![Built with Love](https://img.shields.io/badge/Built%20with-Love-red?style=for-the-badge&logo=heart)

---

© 2024 Ozan İdgü. All rights reserved.

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### Alternative: Direct Installation
If you prefer not to use a virtual environment:
```bash
pip install -r requirements.txt
```

### Data Setup
Add your data file to the `uploads/` folder or upload via the admin panel.

## Usage

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

## 📊 Performance Metrics

### Conservative Optimization Results
- **Baseline Average MAE**: 176.2
- **Optimized Average MAE**: 141.85
- **Improvement**: **-19.5%** (Lower is better)
- **WAPE Target**: 20% (Updated from 10% for realistic evaluation)
- **Feature Count**: 25+ time-based engineered features
- **Optimization Method**: IQR outlier handling + adaptive hyperparameters

### Model Performance
- **Random Forest**: Adaptive n_estimators (100-200) based on data size
- **Extra Trees**: Optimized depth and sampling parameters
- **Gradient Boosting**: Learning rate and subsample optimization
- **Best Model Selection**: Automatic based on cross-validation MAE
- **Ensemble Ready**: Individual model predictions saved for future ensemble

## FastAPI Service
To start the web interface:
```sh
python app/main.py
```

## File Structure
- `app/` : FastAPI backend and routers
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
- **`tests/` : Comprehensive test suite**
  - **`unit/`** : Unit tests for individual modules (25 tests)
    - **`test_config.py`** : Configuration validation tests
    - **`test_data_loader.py`** : Data loading and column mapping tests
    - **`test_feature_engineering.py`** : Feature creation and data preparation tests
    - **`test_models.py`** : ML model functionality and forecasting tests
  - **`integration/`** : End-to-end pipeline tests (7 tests)
    - **`test_pipeline.py`** : Complete pipeline integration tests
  - **`conftest.py`** : Shared fixtures and test data
  - **`TEST_USAGE_GUIDE.md`** : Complete testing documentation
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

**Test System Validation:**
- 32/32 tests passing (100% success rate)
- Test isolation verified - no impact on main system outputs
- Comprehensive coverage of all core modules

## Testing

This project includes a comprehensive test suite to ensure code quality and reliability.

### Running Tests

#### Setup Test Environment
```bash
# Activate virtual environment
source venv/bin/activate

# Install test dependencies
pip install -r requirements-test.txt
```

#### Run All Tests
```bash
# Run all tests with coverage
python -m pytest tests/ --cov=scripts --cov-report=html

# Run tests with verbose output
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v       # Unit tests only
python -m pytest tests/integration/ -v  # Integration tests only
```

#### Test Structure
- `tests/unit/` - Unit tests for individual modules (25 tests)
  - `test_config.py` - Configuration and parameter validation
  - `test_data_loader.py` - Data loading, Excel/CSV processing, column mapping
  - `test_feature_engineering.py` - Weekly data preparation, feature creation
  - `test_models.py` - ML model functionality, forecasting methods
- `tests/integration/` - End-to-end pipeline tests (7 tests)
  - `test_pipeline.py` - Complete workflow integration testing
- `tests/conftest.py` - Shared fixtures and test data
- `tests/TEST_USAGE_GUIDE.md` - Complete testing documentation
- `pytest.ini` - Test configuration

#### Current Test Status
✅ **32/32 tests passing** (100% success rate)
- **Unit Tests**: 25 tests validating individual module functionality
- **Integration Tests**: 7 tests ensuring complete pipeline integrity
- **Coverage**: Comprehensive coverage of core modules
- **Isolation**: Tests do not affect main system outputs

### Coverage Reports
After running tests with coverage, open `htmlcov/index.html` in your browser to view detailed coverage reports.

For detailed testing examples and usage, see `tests/TEST_USAGE_GUIDE.md`.

## Contributing
You can contribute by opening pull requests or issues.


## License

This project is licensed under the GNU General Public License v3.0 (GPLv3). Commercial use is not permitted unless you comply with the terms of the GPLv3. See the [LICENSE](LICENSE) file for full details.

---

## User Interface (Planned)

An advanced user interface is planned for future releases. Currently, the system provides a FastAPI web interface for data upload and analysis. A more comprehensive and user-friendly UI is under development and will be announced in upcoming versions.

---
**📦 Modular Architecture:** The system is now fully modularized for better maintainability, testing, and extensibility. Each module has a specific responsibility and can be easily modified or extended.

**🧪 Quality Assurance:** Comprehensive test suite ensures code reliability and system stability with 32 automated tests covering all critical functionality.

For detailed documentation and usage examples, please review the code and `tests/TEST_USAGE_GUIDE.md`.
