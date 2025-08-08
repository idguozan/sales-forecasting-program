
# Sales Forecasting Program

[![GitHub stars](https://img.shields.io/github/stars/idguozan/sales-forecasting-program?style=flat-square)](https://github.com/idguozan/sales-forecasting-program/stargazers)
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
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)
- [Development Timeline](#development-timeline)
- [Acknowledgments](#acknowledgments)

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
git clone https://github.com/idguozan/sales-forecasting-program.git
cd sales-forecasting-program

# Install dependencies
pip install -r requirements.txt

# Run analysis with your data
python scripts/run_batch_new.py uploads/your_data_file.xlsx
```

**That's it!** Your analysis results will be saved in the `app/static/reports/` directory.

---

## Installation

### Prerequisites

**System Requirements:**
- Python 3.9+
- 4GB+ RAM (recommended for large datasets)
- 1GB free disk space
- pip package manager

**Operating System Support:**
- ✅ macOS 10.14+
- ✅ Windows 10+
- ✅ Linux (Ubuntu 18.04+)

### Step 1: Clone the Repository

```bash
git clone https://github.com/idguozan/sales-forecasting-program.git
cd sales-forecasting-program
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
sales-forecasting-program/
│
├── app/
│   ├── main.py                     # FastAPI application
│   ├── routers/                    # API endpoints
│   └── static/
│       ├── plots/                  # Generated charts (PNG)
│       └── reports/                # Analysis results (PDF, CSV)
│
├── scripts/
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
├── pipeline/                   # Legacy pipeline (maintained for compatibility)
├── tests/                      # Comprehensive test suite
├── uploads/                    # Data upload directory
├── requirements.txt            # Python dependencies
└── README.md                   # This file
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

## Troubleshooting

**Common Issues and Solutions:**

**Installation Issues:**
```bash
# If pip install fails, try upgrading pip first
pip install --upgrade pip

# For macOS users with M1/M2 chips
pip install --no-deps scikit-learn
```

**Memory Issues with Large Datasets:**
- Reduce batch size in `config.py`
- Process sheets individually
- Use data sampling for initial testing

**Import Errors:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**File Not Found Errors:**
- Ensure data files are in `uploads/` directory
- Check file permissions (read access required)
- Verify file format (.xlsx, .csv, .sql, .sqlite)

---

## Contributing

We welcome contributions! Here's how you can help:

**Getting Started:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`python -m pytest tests/`)
6. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
7. Push to the branch (`git push origin feature/AmazingFeature`)
8. Create a Pull Request

**Types of Contributions:**
-  Bug fixes
-  New features
-  Documentation improvements
-  Test coverage expansion
-  Performance optimizations

**Development Guidelines:**
- Follow PEP 8 style guidelines
- Add type hints where possible
- Include unit tests for new features
- Update documentation for API changes

---

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3). See the [LICENSE](LICENSE) file for details.

---

## Authors

**Developed by** - [Ozan İdgü](https://github.com/idguozan)

---

## Development Timeline

**Version 2.0** *(Current)*
- ✅ Modular architecture implementation
- ✅ Comprehensive test suite (32 tests)
- ✅ Advanced feature engineering (25+ features)
- ✅ FastAPI web interface
- ✅ GPL v3 license implementation

**Version 1.5**
- ✅ XGBoost and CatBoost integration
- ✅ Conservative optimization (19.5% MAE improvement)
- ✅ PDF report generation
- ✅ Multi-format data support

**Version 1.0**
- ✅ Initial ML models (Random Forest, Extra Trees)
- ✅ Basic forecasting pipeline
- ✅ Excel data processing

**Roadmap (Future Versions):**
- 🔄 Enhanced web UI with interactive dashboards
- 🔄 Real-time forecasting API
- 🔄 Docker containerization
- 🔄 Cloud deployment options

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

© 2025 Ozan İdgü. All rights reserved.
