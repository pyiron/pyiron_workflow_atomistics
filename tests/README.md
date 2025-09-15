# Test Suite for pyiron_workflow_atomistics

This directory contains comprehensive tests for the `pyiron_workflow_atomistics` module.

## Test Structure

```
tests/
├── conftest.py                    # Pytest configuration and fixtures
├── run_tests.py                   # Test runner script
├── README.md                      # This file
├── unit/                          # Unit tests
│   ├── test_calculator.py         # Tests for calculator module
│   ├── test_bulk.py              # Tests for bulk module
│   ├── test_utils.py             # Tests for utils module
│   ├── test_gb_analysis.py       # Tests for GB analysis module
│   ├── test_gb_cleavage.py       # Tests for GB cleavage module
│   ├── test_featurisers.py       # Tests for featurisers module
│   └── test_version.py           # Tests for version handling
├── integration/                   # Integration tests
│   └── test_workflow_integration.py  # Tests for workflow integration
└── benchmark/                     # Benchmark tests (if needed)
```

## Running Tests

### Using the Test Runner Script

The easiest way to run tests is using the provided test runner script:

```bash
# Run all tests
python tests/run_tests.py

# Run only unit tests
python tests/run_tests.py --type unit

# Run only integration tests
python tests/run_tests.py --type integration

# Run with verbose output
python tests/run_tests.py --verbose

# Run with coverage report
python tests/run_tests.py --coverage

# Run tests in parallel
python tests/run_tests.py --parallel

# Include slow tests
python tests/run_tests.py --slow
```

### Using pytest Directly

You can also run tests directly with pytest:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_calculator.py

# Run specific test function
pytest tests/unit/test_calculator.py::TestCalculatorFunctions::test_validate_calculation_inputs

# Run with verbose output
pytest -v tests/

# Run with coverage
pytest --cov=pyiron_workflow_atomistics tests/

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run tests matching a pattern
pytest -k "test_calculator" tests/

# Run tests excluding slow ones
pytest -m "not slow" tests/
```

### Using unittest

For compatibility, tests can also be run using unittest:

```bash
# Run all tests
python -m unittest discover tests/

# Run specific test file
python -m unittest tests.unit.test_calculator

# Run specific test class
python -m unittest tests.unit.test_calculator.TestCalculatorFunctions
```

## Test Categories

### Unit Tests

Unit tests test individual functions and classes in isolation:

- **test_calculator.py**: Tests for calculator module functions
- **test_bulk.py**: Tests for bulk structure generation and EOS fitting
- **test_utils.py**: Tests for utility functions
- **test_gb_analysis.py**: Tests for grain boundary analysis functions
- **test_gb_cleavage.py**: Tests for grain boundary cleavage functions
- **test_featurisers.py**: Tests for structure featurisation functions
- **test_version.py**: Tests for version handling

### Integration Tests

Integration tests test how different components work together:

- **test_workflow_integration.py**: Tests for complete workflow integration

## Test Fixtures

The `conftest.py` file provides common fixtures for testing:

- `simple_atoms`: Simple H2 molecule
- `fcc_al_atoms`: FCC aluminum structure
- `layered_atoms`: Layered structure for GB testing
- `mock_featuriser`: Mock featuriser function
- `mock_calculator`: Mock calculator
- `mock_engine`: Mock calculation engine
- `temp_dir`: Temporary directory for file operations
- `test_energies_volumes`: Test data for EOS fitting
- `mock_engine_output`: Mock engine output
- `test_dataclass`: Test dataclass for utility testing

## Test Markers

Tests are marked with different categories:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Slow tests (excluded by default)

## Coverage

To generate a coverage report:

```bash
python tests/run_tests.py --coverage
```

This will generate:
- Terminal coverage report
- HTML coverage report in `htmlcov/` directory

## Continuous Integration

The test suite is designed to work with CI/CD systems. Key features:

- Fast unit tests (exclude slow tests by default)
- Comprehensive integration tests
- Coverage reporting
- Parallel test execution support
- Clear test organization

## Adding New Tests

When adding new tests:

1. **Unit tests**: Add to appropriate file in `tests/unit/`
2. **Integration tests**: Add to `tests/integration/`
3. **Use fixtures**: Leverage existing fixtures from `conftest.py`
4. **Add markers**: Mark tests appropriately (`@pytest.mark.unit`, etc.)
5. **Mock external dependencies**: Use mocks for external libraries
6. **Test edge cases**: Include tests for error conditions and edge cases

## Test Dependencies

The test suite requires:

- `pytest`: Test framework
- `pytest-cov`: Coverage reporting
- `pytest-xdist`: Parallel test execution
- `unittest.mock`: Mocking (built-in)
- `numpy`: Numerical operations
- `ase`: Atomic structure handling
- `pymatgen`: Structure conversion
- `matplotlib`: Plotting (for plot tests)

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running tests from the project root
2. **Missing dependencies**: Install test dependencies with `pip install pytest pytest-cov pytest-xdist`
3. **Slow tests**: Use `--slow` flag to include slow tests
4. **Coverage issues**: Make sure the package is installed in development mode

### Debug Mode

To run tests in debug mode:

```bash
pytest --pdb tests/
```

This will drop into debugger on test failures.

## Performance

- Unit tests should run in < 1 second each
- Integration tests may take longer but should be < 10 seconds each
- Slow tests are marked and excluded by default
- Use parallel execution for large test suites
