# Test System Usage Guide

## 1. 🧪 Test Types and Use Cases

### Unit Tests
```bash
# Test a single module
python -m pytest tests/unit/test_config.py -v
python -m pytest tests/unit/test_data_loader.py -v

# Run a specific test function
python -m pytest tests/unit/test_config.py::test_forecast_horizon_reasonable -v
```

### Integration Tests
```bash
# Test entire system integration
python -m pytest tests/integration/ -v
```

### Coverage Analysis
```bash
# See which code parts are tested
python -m pytest tests/ --cov=scripts --cov-report=html
# Then open htmlcov/index.html to view details
```

## 2. 🔍 Using Tests During Development

### A) Before Code Changes
```bash
# Test current state
python -m pytest tests/ -v
```

### B) After Code Changes
```bash
# Check that changes didn't break the system
python -m pytest tests/ -v --tb=short
```

### C) Adding New Features
```bash
# Write tests first, then develop code (TDD)
python -m pytest tests/unit/test_new_feature.py -v
```

## 3. 🚨 Continuous Monitoring (CI/CD)

### Automated Test Execution
```bash
# Run before git push
python -m pytest tests/ --maxfail=5 --tb=line
```

### Performance Testing
```bash
# Measure test duration
python -m pytest tests/ --durations=10
```

## 4. 📊 Test Reporting

### HTML Report
```bash
python -m pytest tests/ --html=reports/test_report.html --self-contained-html
```

### JSON Report
```bash
python -m pytest tests/ --json-report --json-report-file=reports/test_results.json
```

## 5. 🎯 Debugging and Troubleshooting

### Single Test Debug
```bash
# Debug with verbose output
python -m pytest tests/unit/test_data_loader.py::test_column_mapping_basic -v -s
```

### Re-run Failed Tests
```bash
# Re-run only failed tests
python -m pytest --lf
```

### Find Test Coverage Gaps
```bash
python -m pytest tests/ --cov=scripts --cov-fail-under=80
```

## 6. 🔄 Test Driven Development (TDD) Example

### When Adding New Functions:
1. **Write Test** (Will fail initially)
2. **Write Minimum Code** (Just enough to pass test)
3. **Refactor** (Improve code quality)
4. **Repeat**

```python
# Example: New validation function
def test_validate_data_format():
    """Test data format validation"""
    valid_data = pd.DataFrame({'date': [1,2,3], 'qty': [100,200,300]})
    assert validate_data_format(valid_data) == True
    
    invalid_data = pd.DataFrame({'wrong': [1,2,3]})
    assert validate_data_format(invalid_data) == False
```

## 7. 📈 Test Metrics and KPIs

### Test Health Metrics
- **Coverage:** >80%
- **Test Count:** Increasing trend
- **Test Speed:** <5 minutes
- **Success Rate:** >95%

### Monitoring Commands
```bash
# Count tests
python -m pytest tests/ --collect-only -q | tail -1

# Get coverage percentage
python -m pytest tests/ --cov=scripts --cov-report=term | grep TOTAL
```

## 8. 💡 Best Practices

### Test Naming
```python
def test_[action]_[expected_result]_[condition]():
    """
    test_calculate_forecast_returns_positive_values_with_valid_data()
    test_load_data_raises_error_with_invalid_file()
    """
```

### Test Organization
```
tests/
├── unit/          # Fast, isolated tests
├── integration/   # Full system tests
├── fixtures/      # Test datasets
└── conftest.py    # Shared setups
```

### Mock Usage
```python
@pytest.fixture
def mock_api_call():
    with patch('requests.get') as mock:
        mock.return_value.json.return_value = {'status': 'ok'}
        yield mock
```

## 9. 🚀 Production Ready Testing

### Pre-deployment Checklist
```bash
# Full test suite
python -m pytest tests/ --cov=scripts --cov-fail-under=80

# Performance test
python -m pytest tests/ --benchmark-only

# Security test
python -m pytest tests/ -m security
```

### Staging Environment Test
```bash
# Test in production-like environment
ENVIRONMENT=staging python -m pytest tests/integration/
```

## 10. 📝 Test Documentation

### Test Results Tracking
```bash
# Save test results
python -m pytest tests/ --json-report --json-report-file=reports/test_$(date +%Y%m%d).json
```

By following this guide, you can use the test system with maximum efficiency, improve code quality, and catch bugs early! 🎯
