# Testing Guide

## Running Tests

### Install Test Dependencies
```bash
pip install -r requirements-test.txt
```

### Run All Tests
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=scripts --cov-report=html

# Run specific test categories
python -m pytest tests/unit/ -m unit
python -m pytest tests/integration/ -m integration
```

### Run Specific Test Files
```bash
# Data loader tests
python -m pytest tests/unit/test_data_loader.py -v

# Feature engineering tests
python -m pytest tests/unit/test_feature_engineering.py -v

# ML model tests
python -m pytest tests/unit/test_models.py -v

# Integration tests
python -m pytest tests/integration/test_pipeline.py -v
```

### Test Options
```bash
# Verbose output
python -m pytest tests/ -v

# Stop on first failure
python -m pytest tests/ -x

# Run tests in parallel
python -m pytest tests/ -n 4

# Generate HTML report
python -m pytest tests/ --html=reports/test_report.html
```

## Test Structure

### Unit Tests (`tests/unit/`)
- `test_data_loader.py`: Tests for data loading and column mapping
- `test_feature_engineering.py`: Tests for feature creation and data preparation
- `test_models.py`: Tests for ML models and forecasting logic
- `test_config.py`: Tests for configuration management

### Integration Tests (`tests/integration/`)
- `test_pipeline.py`: End-to-end pipeline tests

### Test Fixtures (`tests/conftest.py`)
- Sample datasets for testing
- Mock configurations
- Temporary file helpers

## Writing New Tests

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Example Test
```python
def test_my_function():
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = my_function(input_data)
    
    # Assert
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert 'expected_column' in result.columns
```

### Using Fixtures
```python
def test_with_fixture(sample_sales_data):
    result = process_data(sample_sales_data)
    assert len(result) > 0
```

## Coverage Reports

After running tests with coverage:
- Open `htmlcov/index.html` in browser
- View line-by-line coverage
- Identify untested code areas

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
- All tests should pass
- Coverage should be >80%
- Tests should complete in <5 minutes
