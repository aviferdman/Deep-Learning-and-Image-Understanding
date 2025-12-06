# Deep Learning HW2 - Test Suite Documentation

## Overview
This is a comprehensive, modular test suite for Deep Learning HW2 functions. The test framework is designed to be easily extensible for testing all functions in the assignment.

## Files
- **`test_functions.py`** - Main test suite (modular, extensible)
- **`README_TESTS.md`** - This documentation file

## Quick Start

### Run All Tests
```bash
cd hw2
python test_functions.py
```

### Expected Output
```
============================================================
DEEP LEARNING HW2 - COMPREHENSIVE TEST SUITE
============================================================

[Individual test suite results...]

============================================================
FINAL SUMMARY
============================================================
🎉 All test suites passed!
============================================================
```

## Architecture

### Modular Design
The test suite uses a class-based architecture that makes it easy to add new test suites:

```python
# Base class for all test suites
class TestBase:
    - assert_close()      # Compare arrays/values
    - print_summary()     # Print test results

# Individual test suites inherit from TestBase
class TestSoftmaxLoss(TestBase):
    - test_method_1()
    - test_method_2()
    - run_all_tests()

class TestL2Regularization(TestBase):
    # TODO: Add tests here
    pass

# Main runner executes all test suites
def run_all_tests():
    # Runs each test suite
    # Returns overall pass/fail
```

### Benefits
1. **Easy to Extend**: Just create a new class inheriting from `TestBase`
2. **Isolated Tests**: Each function has its own test suite
3. **Reusable Utilities**: Helper functions shared across all tests
4. **Clear Output**: Organized results for each test suite

## Current Test Suites

### 1. TestSoftmaxLoss ✅ (Complete - 19/19 tests passing)

Tests the `softmax_loss` function with 12 test cases:

#### Test Cases:
1. **Basic Correctness** - Validates against known expected values
2. **Numerical Gradient** - Compares analytical vs numerical gradients
3. **Numerical Stability (Large)** - Tests with scores like 1000 (overflow prevention)
4. **Numerical Stability (Small)** - Tests with scores like -1000 (underflow prevention)
5. **Single Sample** - Edge case: batch size = 1
6. **Perfect Prediction** - Behavior when correct class score >> others
7. **Uniform Scores** - All scores equal (uniform probability)
8. **Batch Size Invariance** - Loss properly averaged over batch
9. **Gradient Sum Property** - Gradients sum to zero per sample
10. **Different Class Sizes** - Tests with 2, 5, 10, 100 classes
11. **Loss Positivity** - Loss always non-negative
12. **Cross-Entropy Bounds** - Theoretical bounds verification

#### What It Tests:
- ✅ Correct loss computation
- ✅ Correct gradient computation
- ✅ Numerical stability (no overflow/underflow)
- ✅ Edge cases handling
- ✅ Mathematical properties
- ✅ Vectorization (no loops)

## Adding New Test Suites

### Step 1: Create Test Class
```python
class TestNewFunction(TestBase):
    """Test suite for new_function"""
    
    def __init__(self):
        super().__init__("New Function")
    
    def test_basic(self):
        """Test basic functionality"""
        print("\n=== Test 1: Basic Test ===")
        
        # Your test code here
        result = new_function(input_data)
        expected = expected_result
        
        self.assert_close(result, expected, 
                         test_name="Basic functionality")
    
    def run_all_tests(self):
        """Run all tests for this function"""
        print("=" * 60)
        print("NEW FUNCTION TEST SUITE")
        print("=" * 60)
        
        self.test_basic()
        # Add more tests...
        
        return self.print_summary()
```

### Step 2: Add to Main Runner
```python
def run_all_tests():
    # ... existing code ...
    
    # Add your new test suite
    new_tester = TestNewFunction()
    new_passed = new_tester.run_all_tests()
    all_passed = all_passed and new_passed
    
    # ... rest of code ...
```

### Step 3: Add Function Implementation (if testing standalone)
```python
def new_function(x):
    """Your function implementation"""
    # Implementation here
    return result
```

### 2. TestL2Regularization ✅ (Complete - 34/34 tests passing)

Tests the `l2_regulariztion_loss` function with 12 test cases:

#### Test Cases:
1. **Basic Correctness** - Validates against known expected values
2. **Zero Regularization** - Tests with reg=0 (no regularization)
3. **Numerical Gradient** - Compares analytical vs numerical gradients
4. **Different Weight Shapes** - Tests with various matrix shapes (2D, 3D)
5. **Loss Formula** - Verifies loss = (reg/2) * sum(W²)
6. **Gradient Formula** - Verifies gradient = reg * W
7. **Different Regularization Strengths** - Tests with various reg values
8. **Loss Scaling** - Verifies quadratic scaling: loss(k*W) = k²*loss(W)
9. **Gradient Scaling** - Verifies linear scaling: grad(k*W) = k*grad(W)
10. **Loss Positivity** - Loss always non-negative
11. **Vectorization** - Ensures no loops, fast execution
12. **Zero Weights** - Edge case with all zeros

#### What It Tests:
- ✅ Correct loss computation: L = (λ/2) * ||W||²
- ✅ Correct gradient computation: dW = λ * W
- ✅ Mathematical properties (scaling, positivity)
- ✅ Edge cases (zero reg, zero weights)
- ✅ Various weight shapes (2D, 3D tensors)
- ✅ Vectorization (no loops)

## Planned Test Suites (TODO)

### 3. TestFullyConnectedLayer ⏳
For testing `fc_forward` and `fc_backward`:
- [ ] Forward pass correctness
- [ ] Backward pass correctness
- [ ] Gradient numerical verification
- [ ] Different input/output dimensions
- [ ] Reshape handling

### 4. TestReLU ⏳
For testing `relu_forward` and `relu_backward`:
- [ ] Forward pass (positive/negative inputs)
- [ ] Backward pass gradient
- [ ] Zero crossing behavior
- [ ] Large value stability

### 5. TestCombinedLayers ⏳
For testing `fc_relu_forward` and `fc_relu_backward`:
- [ ] Combined forward pass
- [ ] Combined backward pass
- [ ] Gradient verification

### 6. TestThreeLayerNet ⏳
For testing the `ThreeLayerNet` class:
- [ ] Initialization
- [ ] Forward pass (step with y=None)
- [ ] Loss computation
- [ ] Backward pass
- [ ] Training functionality
- [ ] Prediction accuracy

## Helper Functions

### eval_numerical_gradient()
Computes numerical gradient for loss functions using finite differences:
```python
dx_numerical = eval_numerical_gradient(loss_function, x, y)
```

### eval_numerical_gradient_array()
Computes numerical gradient for layer functions:
```python
dx_numerical = eval_numerical_gradient_array(forward_func, x, dout)
```

## Test Writing Best Practices

1. **Use Descriptive Names**: `test_numerical_stability_large_scores()` not `test1()`
2. **Test Edge Cases**: Single samples, zero inputs, extreme values
3. **Verify Gradients**: Always compare analytical vs numerical gradients
4. **Check Shapes**: Ensure output shapes match expectations
5. **Test Properties**: Mathematical properties (e.g., gradients sum to zero)
6. **Use Seeds**: `np.random.seed()` for reproducible tests
7. **Document Tests**: Add docstrings explaining what each test verifies

## Running Specific Test Suites

To run only specific tests, modify `run_all_tests()`:

```python
if __name__ == "__main__":
    # Run only softmax tests
    softmax_tester = TestSoftmaxLoss()
    success = softmax_tester.run_all_tests()
    sys.exit(0 if success else 1)
```

## Debugging Failed Tests

When a test fails:

1. **Check Error Message**: The output shows which assertion failed
2. **Review Test Code**: Look at what the test is checking
3. **Add Print Statements**: Print intermediate values
4. **Run Isolated**: Run just that one test
5. **Check Implementation**: Compare your code to expected behavior

Example failed test output:
```
✗ FAIL: Gradient correctness
  Error: Arrays not equal
  max absolute difference: 0.0001234
  max relative difference: 0.05
```

## Mathematical Background

### Softmax Cross-Entropy Loss
$$L = -\frac{1}{N}\sum_{i=1}^{N} \log(p_{y_i})$$

where $p_{y_i}$ is the softmax probability for the correct class.

### Gradient
$$\frac{\partial L}{\partial s_{ij}} = \frac{1}{N}(p_{ij} - \mathbb{1}(j = y_i))$$

### Numerical Gradient (Centered Difference)
$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x - h e_i)}{2h}$$

## Test Results Summary

### Current Status
- ✅ **Softmax Loss**: 19/19 tests passing
- ✅ **L2 Regularization**: 34/34 tests passing
- ⏳ **Fully Connected**: Not yet implemented  
- ⏳ **ReLU**: Not yet implemented
- ⏳ **Combined Layers**: Not yet implemented
- ⏳ **ThreeLayerNet**: Not yet implemented

### Total Coverage
- **Implemented**: 2/6 test suites (33%)
- **Tests Passing**: 53/53 (100%)

## Contributing

To add tests for additional functions:

1. Create a new test class inheriting from `TestBase`
2. Implement test methods following the naming convention `test_*`
3. Add the test suite to `run_all_tests()`
4. Update this documentation with the new test suite details
5. Run all tests to ensure nothing breaks

## Notes

- All tests use `numpy` only (no PyTorch, TensorFlow, etc.)
- Tests verify vectorized implementations (no Python loops)
- Numerical gradient checks use small step size (h=1e-5)
- Relative tolerance typically 1e-5, absolute tolerance 1e-7
- Seeds are set for reproducibility

## Example Usage

```python
# Run all tests
python test_functions.py

# Import and use in your code
from test_functions import TestSoftmaxLoss, eval_numerical_gradient

# Create tester
tester = TestSoftmaxLoss()

# Run tests
success = tester.run_all_tests()

# Check specific gradient
gradient = eval_numerical_gradient(my_loss_function, X, y)
```

## Conclusion

This modular test suite provides:
- ✅ Comprehensive testing framework
- ✅ Easy extensibility for new functions
- ✅ Clear, organized output
- ✅ Reusable test utilities
- ✅ Best practices for gradient verification

As you implement more functions in HW2, simply add new test classes following the established pattern!
