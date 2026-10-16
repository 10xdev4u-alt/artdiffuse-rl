# Testing Guide

## Overview

This document describes the testing strategy for ArtDiffuse-RL.

## Test Types

### Unit Tests

Test individual components in isolation.

```bash
pytest tests/test_model.py -v
```

### Integration Tests

Test how components work together.

```bash
pytest tests/test_integration.py -v
```

### End-to-End Tests

Test the complete workflow.

```bash
pytest tests/test_e2e.py -v
```

## Running Tests

### All Tests

```bash
pytest tests/ -v
```

### Specific Test File

```bash
pytest tests/test_model.py -v
```

### Specific Test

```bash
pytest tests/test_model.py::test_unet_forward -v
```

### With Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

## Test Structure

```
tests/
  test_model.py          # Model architecture tests
  test_dataset.py        # Dataset tests
  test_train.py          # Training tests
  test_evaluate.py       # Evaluation tests
  conftest.py            # Pytest fixtures
```

## Writing Tests

### Basic Test

```python
def test_something():
    """Test that something works."""
    result = do_something()
    assert result == expected
```

### Parametrized Test

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_multiply_by_two(input, expected):
    assert input * 2 == expected
```

### Fixture

```python
@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return UNet(ch=64, ch_mult=(1, 2), num_res_blocks=1)
```

## Test Coverage

### Current Coverage

- Model architecture: 100%
- Dataset: 80%
- Training: 60%
- Evaluation: 40%

### Goals

- Model architecture: 100%
- Dataset: 90%
- Training: 80%
- Evaluation: 70%

## Continuous Integration

Tests run automatically on:

- Every push to main
- Every pull request

### GitHub Actions

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
```

## Performance Tests

### Training Speed

```python
import time
start = time.time()
# Train for 100 steps
elapsed = time.time() - start
assert elapsed < 60  # Should take less than 60 seconds
```

### Memory Usage

```python
import torch
initial_memory = torch.cuda.memory_allocated()
# Run model
final_memory = torch.cuda.memory_allocated()
assert final_memory - initial_memory < 1000 * 1024 * 1024  # Less than 1GB
```

## Debugging Tests

### Run with PDB

```bash
pytest tests/test_model.py::test_unet_forward --pdb
```

### Print Debugging

```python
def test_something():
    result = do_something()
    print(f"Result: {result}")
    assert result == expected
```

### Verbose Output

```bash
pytest tests/ -vv
```

## Test Data

### Synthetic Data

```python
@pytest.fixture
def synthetic_images():
    """Create synthetic images for testing."""
    return torch.randn(10, 3, 64, 64)
```

### Real Data

```python
@pytest.fixture
def real_images():
    """Load real images for testing."""
    dataset = MidjourneyDataset('./data', max_samples=10)
    return [dataset[i]['image'] for i in range(len(dataset))]
```

## Best Practices

1. **Test one thing per test**: Keep tests focused
2. **Use descriptive names**: `test_unet_forward_with_valid_input`
3. **Use fixtures**: Avoid code duplication
4. **Test edge cases**: Empty input, large input, etc.
5. **Test error handling**: Invalid input should raise errors
6. **Keep tests fast**: Tests should run quickly
7. **Make tests independent**: Tests should not depend on each other

## Troubleshooting

### Test Fails

1. Check the error message
2. Run with `-vv` for verbose output
3. Use `--pdb` to debug
4. Check if test data is correct

### Test Hangs

1. Check for infinite loops
2. Add timeouts
3. Check for deadlocks

### Test is Flaky

1. Set random seeds
2. Use deterministic algorithms
3. Avoid timing-dependent tests
