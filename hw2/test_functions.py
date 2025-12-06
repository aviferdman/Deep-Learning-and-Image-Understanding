"""
Test suite for Deep Learning HW2 functions
Tests for softmax_loss, L2 regularization, fully connected layers, ReLU, etc.
"""

import numpy as np
import sys

# ============================================================================
# Helper Functions
# ============================================================================

def eval_numerical_gradient(f, x, y, h=1e-5):
    """
    Compute numerical gradient using finite differences
    
    Args:
        f: Function that takes (x, y) and returns (loss, gradient)
        x: Input array
        y: Labels
        h: Step size for finite differences
    
    Returns:
        Numerical gradient array
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        
        x[ix] = oldval + h
        fxph, _ = f(x, y)
        
        x[ix] = oldval - h
        fxmh, _ = f(x, y)
        
        x[ix] = oldval
        
        grad[ix] = (fxph - fxmh) / (2 * h)
        it.iternext()
    
    return grad

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad

# ============================================================================
# Function Implementations (to be tested)
# ============================================================================

def softmax_loss(scores, y):
    """
    Computes the loss and gradient for softmax classification.
    
    Inputs:
    - scores: scores of shape (N, C) where scores[i, c] is the score for class c on input X[i].
    - y: Vector of labels
    
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to scores
    """
    N = scores.shape[0]
    
    # Numerical stability: subtract max from each row
    scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
    
    # Compute softmax probabilities
    exp_scores = np.exp(scores_shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Compute cross-entropy loss
    correct_logprobs = -np.log(probs[np.arange(N), y])
    loss = np.sum(correct_logprobs) / N
    
    # Compute gradient
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    
    return loss, dx

def l2_regulariztion_loss(W, reg=0):
    """
    L2 regulariztion loss function, vectorized version.
    - W: a layer's weights.
    - reg: (float) regularization strength
    """
    # L2 regularization loss: (reg/2) * sum(W^2)
    loss = 0.5 * reg * np.sum(W * W)
    
    # Gradient of L2 regularization: reg * W
    dW = reg * W
    
    return loss, dW

# ============================================================================
# Base Test Class
# ============================================================================

class TestBase:
    """Base class for test suites"""
    
    def __init__(self, name="Test Suite"):
        self.name = name
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []
    
    def assert_close(self, actual, expected, rtol=1e-5, atol=1e-8, test_name=""):
        """Assert that two arrays are close"""
        try:
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
            self.passed_tests += 1
            self.test_results.append(f"✓ PASS: {test_name}")
            return True
        except AssertionError as e:
            self.failed_tests += 1
            self.test_results.append(f"✗ FAIL: {test_name}")
            self.test_results.append(f"  Error: {str(e)}")
            return False
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print(f"{self.name} - TEST RESULTS")
        print("=" * 60)
        for result in self.test_results:
            print(result)
        
        print("\n" + "=" * 60)
        print(f"SUMMARY: {self.passed_tests} passed, {self.failed_tests} failed")
        print("=" * 60)
        
        return self.failed_tests == 0

# ============================================================================
# Softmax Loss Tests
# ============================================================================

class TestSoftmaxLoss(TestBase):
    """Test suite for softmax loss function"""
    
    def __init__(self):
        super().__init__("Softmax Loss")
    
    def test_basic_correctness(self):
        """Test 1: Basic correctness with known values"""
        print("\n=== Test 1: Basic Correctness ===")
        np.random.seed(42)
        
        num_instances = 5
        num_classes = 3
        y = np.random.randint(num_classes, size=num_instances)
        scores = np.random.randn(num_instances * num_classes).reshape(num_instances, num_classes)
        
        loss, dx = softmax_loss(scores, y)
        
        # Expected values from the notebook
        correct_grad = np.array([[ 0.0062,  0.1751, -0.1813],
                                 [-0.1463,  0.0561,  0.0901],
                                 [ 0.0404,  0.0771, -0.1174],
                                 [ 0.0223,  0.0855, -0.1078],
                                 [-0.1935,  0.1358,  0.0578]])
        correct_loss = 1.7544
        
        self.assert_close(dx.round(4), correct_grad, rtol=1e-3, 
                         test_name="Basic gradient correctness")
        self.assert_close(loss.round(4), correct_loss, rtol=1e-3,
                         test_name="Basic loss correctness")
    
    def test_numerical_gradient(self):
        """Test 2: Gradient check using numerical approximation"""
        print("\n=== Test 2: Numerical Gradient Check ===")
        np.random.seed(123)
        
        N, C = 10, 5
        scores = np.random.randn(N, C) * 0.01
        y = np.random.randint(C, size=N)
        
        loss, dx_analytic = softmax_loss(scores, y)
        dx_numerical = eval_numerical_gradient(softmax_loss, scores, y)
        
        self.assert_close(dx_analytic, dx_numerical, rtol=1e-5, atol=1e-7,
                         test_name="Analytical vs numerical gradient")
    
    def test_numerical_stability_large_scores(self):
        """Test 3: Numerical stability with large scores"""
        print("\n=== Test 3: Numerical Stability (Large Scores) ===")
        
        scores = np.array([[1000, 1001, 999],
                          [500, 501, 499],
                          [100, 105, 95]])
        y = np.array([1, 2, 1])
        
        try:
            loss, dx = softmax_loss(scores, y)
            
            assert np.isfinite(loss), "Loss should be finite"
            assert np.all(np.isfinite(dx)), "Gradients should be finite"
            
            self.passed_tests += 1
            self.test_results.append("✓ PASS: Numerical stability with large scores")
        except Exception as e:
            self.failed_tests += 1
            self.test_results.append(f"✗ FAIL: Numerical stability with large scores - {str(e)}")
    
    def test_numerical_stability_small_scores(self):
        """Test 4: Numerical stability with small scores"""
        print("\n=== Test 4: Numerical Stability (Small Scores) ===")
        
        scores = np.array([[-1000, -1001, -999],
                          [-500, -501, -499],
                          [-100, -105, -95]])
        y = np.array([0, 0, 2])
        
        try:
            loss, dx = softmax_loss(scores, y)
            
            assert np.isfinite(loss), "Loss should be finite"
            assert np.all(np.isfinite(dx)), "Gradients should be finite"
            
            self.passed_tests += 1
            self.test_results.append("✓ PASS: Numerical stability with small scores")
        except Exception as e:
            self.failed_tests += 1
            self.test_results.append(f"✗ FAIL: Numerical stability with small scores - {str(e)}")
    
    def test_single_sample(self):
        """Test 5: Single sample case"""
        print("\n=== Test 5: Single Sample ===")
        
        scores = np.array([[2.0, 1.0, 0.1]])
        y = np.array([0])
        
        loss, dx = softmax_loss(scores, y)
        
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)
        
        expected_dx = probs.copy()
        expected_dx[0, 0] -= 1
        
        self.assert_close(dx, expected_dx, test_name="Single sample gradient")
        
        expected_loss = -np.log(probs[0, 0])
        self.assert_close(loss, expected_loss, test_name="Single sample loss")
    
    def test_perfect_prediction(self):
        """Test 6: Perfect prediction"""
        print("\n=== Test 6: Perfect Prediction ===")
        
        scores = np.array([[100, 0, 0],
                          [0, 100, 0],
                          [0, 0, 100]])
        y = np.array([0, 1, 2])
        
        loss, dx = softmax_loss(scores, y)
        
        self.assert_close(loss, 0.0, atol=1e-5, test_name="Perfect prediction loss near zero")
        
        correct_class_grads = dx[np.arange(3), y]
        self.assert_close(correct_class_grads, np.zeros(3), atol=1e-5,
                         test_name="Perfect prediction gradient near zero")
    
    def test_uniform_scores(self):
        """Test 7: Uniform scores"""
        print("\n=== Test 7: Uniform Scores ===")
        
        scores = np.array([[1.0, 1.0, 1.0],
                          [5.0, 5.0, 5.0]])
        y = np.array([0, 2])
        
        loss, dx = softmax_loss(scores, y)
        
        expected_prob = 1.0 / 3.0
        expected_loss = -np.log(expected_prob)
        
        self.assert_close(loss, expected_loss, test_name="Uniform scores loss")
    
    def test_batch_size_invariance(self):
        """Test 8: Batch size invariance"""
        print("\n=== Test 8: Batch Size Invariance ===")
        np.random.seed(456)
        
        single_score = np.random.randn(1, 4)
        single_y = np.array([2])
        
        loss_single, _ = softmax_loss(single_score, single_y)
        
        double_scores = np.vstack([single_score, single_score])
        double_y = np.array([2, 2])
        
        loss_double, _ = softmax_loss(double_scores, double_y)
        
        self.assert_close(loss_single, loss_double, test_name="Loss is averaged over batch")
    
    def test_gradient_sum_property(self):
        """Test 9: Gradients sum to zero"""
        print("\n=== Test 9: Gradient Sum Property ===")
        np.random.seed(789)
        
        N, C = 8, 6
        scores = np.random.randn(N, C)
        y = np.random.randint(C, size=N)
        
        loss, dx = softmax_loss(scores, y)
        
        gradient_sums = dx.sum(axis=1)
        
        self.assert_close(gradient_sums, np.zeros(N), atol=1e-10,
                         test_name="Gradients sum to zero per sample")
    
    def test_different_class_sizes(self):
        """Test 10: Different class sizes"""
        print("\n=== Test 10: Different Class Sizes ===")
        
        for num_classes in [2, 5, 10, 100]:
            np.random.seed(999)
            N = 20
            scores = np.random.randn(N, num_classes) * 0.1
            y = np.random.randint(num_classes, size=N)
            
            loss, dx = softmax_loss(scores, y)
            
            assert dx.shape == (N, num_classes), f"Gradient shape mismatch for {num_classes} classes"
            assert isinstance(loss, (float, np.floating)), "Loss should be scalar"
            
            dx_numerical = eval_numerical_gradient(softmax_loss, scores, y)
            
            self.assert_close(dx, dx_numerical, rtol=1e-5, atol=1e-7,
                             test_name=f"Gradient correctness with {num_classes} classes")
    
    def test_loss_is_positive(self):
        """Test 11: Loss positivity"""
        print("\n=== Test 11: Loss Positivity ===")
        np.random.seed(111)
        
        for _ in range(10):
            N, C = np.random.randint(5, 20), np.random.randint(3, 10)
            scores = np.random.randn(N, C) * np.random.uniform(0.1, 10)
            y = np.random.randint(C, size=N)
            
            loss, _ = softmax_loss(scores, y)
            
            assert loss >= 0, f"Loss should be non-negative, got {loss}"
        
        self.passed_tests += 1
        self.test_results.append("✓ PASS: Loss is always non-negative")
    
    def test_cross_entropy_bounds(self):
        """Test 12: Cross-entropy loss bounds"""
        print("\n=== Test 12: Cross-Entropy Loss Bounds ===")
        
        N, C = 10, 5
        scores = np.random.randn(N, C)
        y = np.random.randint(C, size=N)
        
        loss, _ = softmax_loss(scores, y)
        
        assert loss >= 0, f"Loss should be non-negative, got {loss}"
        assert np.isfinite(loss), f"Loss should be finite, got {loss}"
        
        uniform_scores = np.ones((N, C))
        loss_uniform, _ = softmax_loss(uniform_scores, y)
        expected_uniform_loss = np.log(C)
        
        self.assert_close(loss_uniform, expected_uniform_loss, rtol=1e-5,
                         test_name="Uniform probability loss equals log(C)")
        
        self.passed_tests += 1
        self.test_results.append("✓ PASS: Loss bounds verified")
    
    def run_all_tests(self):
        """Run all tests"""
        print("=" * 60)
        print("SOFTMAX LOSS FUNCTION TEST SUITE")
        print("=" * 60)
        
        self.test_basic_correctness()
        self.test_numerical_gradient()
        self.test_numerical_stability_large_scores()
        self.test_numerical_stability_small_scores()
        self.test_single_sample()
        self.test_perfect_prediction()
        self.test_uniform_scores()
        self.test_batch_size_invariance()
        self.test_gradient_sum_property()
        self.test_different_class_sizes()
        self.test_loss_is_positive()
        self.test_cross_entropy_bounds()
        
        return self.print_summary()

# ============================================================================
# L2 Regularization Tests
# ============================================================================

class TestL2Regularization(TestBase):
    """Test suite for L2 regularization loss function"""
    
    def __init__(self):
        super().__init__("L2 Regularization")
    
    def test_basic_correctness(self):
        """Test 1: Basic correctness with known values"""
        print("\n=== Test 1: Basic Correctness ===")
        np.random.seed(42)
        
        W = np.array([[0.1, 0.2, 0.3],
                      [0.4, 0.5, 0.6]])
        reg = 0.5
        
        loss, dW = l2_regulariztion_loss(W, reg)
        
        # Expected: loss = 0.5 * reg * sum(W^2)
        expected_loss = 0.5 * reg * np.sum(W * W)
        # Expected: dW = reg * W
        expected_dW = reg * W
        
        self.assert_close(loss, expected_loss, test_name="Basic loss correctness")
        self.assert_close(dW, expected_dW, test_name="Basic gradient correctness")
    
    def test_zero_regularization(self):
        """Test 2: Zero regularization strength"""
        print("\n=== Test 2: Zero Regularization ===")
        
        W = np.random.randn(5, 10)
        reg = 0.0
        
        loss, dW = l2_regulariztion_loss(W, reg)
        
        self.assert_close(loss, 0.0, test_name="Zero regularization loss")
        self.assert_close(dW, np.zeros_like(W), test_name="Zero regularization gradient")
    
    def test_numerical_gradient(self):
        """Test 3: Gradient check using numerical approximation"""
        print("\n=== Test 3: Numerical Gradient Check ===")
        np.random.seed(123)
        
        W = np.random.randn(4, 6) * 0.1
        reg = 0.3
        
        # Compute analytical gradient
        loss, dW_analytic = l2_regulariztion_loss(W, reg)
        
        # Compute numerical gradient
        def loss_func(W_):
            loss_, _ = l2_regulariztion_loss(W_, reg)
            return loss_
        
        dW_numerical = np.zeros_like(W)
        h = 1e-5
        it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            oldval = W[ix]
            
            W[ix] = oldval + h
            loss_plus = loss_func(W)
            
            W[ix] = oldval - h
            loss_minus = loss_func(W)
            
            W[ix] = oldval
            dW_numerical[ix] = (loss_plus - loss_minus) / (2 * h)
            it.iternext()
        
        self.assert_close(dW_analytic, dW_numerical, rtol=1e-5, atol=1e-7,
                         test_name="Analytical vs numerical gradient")
    
    def test_different_weight_shapes(self):
        """Test 4: Different weight matrix shapes"""
        print("\n=== Test 4: Different Weight Shapes ===")
        
        shapes = [(10, 5), (1, 100), (50, 1), (20, 20), (3, 3, 3)]
        reg = 0.1
        
        for shape in shapes:
            W = np.random.randn(*shape) * 0.5
            loss, dW = l2_regulariztion_loss(W, reg)
            
            assert dW.shape == W.shape, f"Gradient shape mismatch for shape {shape}"
            assert isinstance(loss, (float, np.floating)), "Loss should be scalar"
            
            expected_loss = 0.5 * reg * np.sum(W * W)
            expected_dW = reg * W
            
            self.assert_close(loss, expected_loss, test_name=f"Loss correctness for shape {shape}")
            self.assert_close(dW, expected_dW, test_name=f"Gradient correctness for shape {shape}")
    
    def test_loss_formula(self):
        """Test 5: Verify loss formula"""
        print("\n=== Test 5: Loss Formula Verification ===")
        
        # Test that loss = (reg/2) * sum(W^2)
        W = np.array([[1.0, 2.0], [3.0, 4.0]])
        reg = 0.5
        
        loss, _ = l2_regulariztion_loss(W, reg)
        
        # Manual calculation: sum of squares = 1 + 4 + 9 + 16 = 30
        # loss = 0.5 * 0.5 * 30 = 7.5
        expected_loss = 0.5 * reg * (1.0 + 4.0 + 9.0 + 16.0)
        
        self.assert_close(loss, expected_loss, test_name="Loss formula verification")
    
    def test_gradient_formula(self):
        """Test 6: Verify gradient formula"""
        print("\n=== Test 6: Gradient Formula Verification ===")
        
        W = np.array([[1.0, -2.0], [3.0, -4.0]])
        reg = 2.0
        
        loss, dW = l2_regulariztion_loss(W, reg)
        
        # Gradient should be: reg * W = 2.0 * W
        expected_dW = np.array([[2.0, -4.0], [6.0, -8.0]])
        
        self.assert_close(dW, expected_dW, test_name="Gradient formula verification")
    
    def test_different_regularization_strengths(self):
        """Test 7: Different regularization strengths"""
        print("\n=== Test 7: Different Regularization Strengths ===")
        
        W = np.random.randn(5, 5)
        
        for reg in [0.001, 0.01, 0.1, 1.0, 10.0]:
            loss, dW = l2_regulariztion_loss(W, reg)
            
            expected_loss = 0.5 * reg * np.sum(W * W)
            expected_dW = reg * W
            
            self.assert_close(loss, expected_loss, 
                             test_name=f"Loss with reg={reg}")
            self.assert_close(dW, expected_dW, 
                             test_name=f"Gradient with reg={reg}")
    
    def test_loss_scaling(self):
        """Test 8: Loss scales quadratically with weights"""
        print("\n=== Test 8: Loss Scaling Property ===")
        
        W = np.random.randn(3, 4)
        reg = 0.5
        
        loss1, _ = l2_regulariztion_loss(W, reg)
        loss2, _ = l2_regulariztion_loss(2 * W, reg)
        loss3, _ = l2_regulariztion_loss(3 * W, reg)
        
        # Loss should scale quadratically: loss(k*W) = k^2 * loss(W)
        self.assert_close(loss2, 4 * loss1, rtol=1e-5,
                         test_name="Loss scales quadratically (2x)")
        self.assert_close(loss3, 9 * loss1, rtol=1e-5,
                         test_name="Loss scales quadratically (3x)")
    
    def test_gradient_scaling(self):
        """Test 9: Gradient scales linearly with weights"""
        print("\n=== Test 9: Gradient Scaling Property ===")
        
        W = np.random.randn(3, 4)
        reg = 0.5
        
        _, dW1 = l2_regulariztion_loss(W, reg)
        _, dW2 = l2_regulariztion_loss(2 * W, reg)
        
        # Gradient should scale linearly: dW(k*W) = k * dW(W)
        self.assert_close(dW2, 2 * dW1, test_name="Gradient scales linearly")
    
    def test_loss_positivity(self):
        """Test 10: Loss is always non-negative"""
        print("\n=== Test 10: Loss Positivity ===")
        np.random.seed(111)
        
        for _ in range(10):
            W = np.random.randn(np.random.randint(2, 10), 
                               np.random.randint(2, 10)) * np.random.uniform(0.1, 10)
            reg = np.random.uniform(0, 1)
            
            loss, _ = l2_regulariztion_loss(W, reg)
            
            assert loss >= 0, f"Loss should be non-negative, got {loss}"
        
        self.passed_tests += 1
        self.test_results.append("✓ PASS: Loss is always non-negative")
    
    def test_vectorization(self):
        """Test 11: Vectorized implementation (no loops)"""
        print("\n=== Test 11: Vectorization Check ===")
        
        # Test that the function uses vectorized operations
        W = np.random.randn(100, 100)
        reg = 0.5
        
        import time
        start = time.time()
        for _ in range(100):
            loss, dW = l2_regulariztion_loss(W, reg)
        elapsed = time.time() - start
        
        # Should be fast (< 0.1 seconds for 100 iterations)
        assert elapsed < 0.1, f"Function may not be vectorized (took {elapsed:.4f}s)"
        
        self.passed_tests += 1
        self.test_results.append(f"✓ PASS: Vectorization check ({elapsed:.4f}s for 100 iterations)")
    
    def test_gradient_with_zero_weights(self):
        """Test 12: Gradient when weights are zero"""
        print("\n=== Test 12: Zero Weights ===")
        
        W = np.zeros((5, 5))
        reg = 0.5
        
        loss, dW = l2_regulariztion_loss(W, reg)
        
        self.assert_close(loss, 0.0, test_name="Loss with zero weights")
        self.assert_close(dW, np.zeros_like(W), test_name="Gradient with zero weights")
    
    def run_all_tests(self):
        """Run all tests"""
        print("=" * 60)
        print("L2 REGULARIZATION FUNCTION TEST SUITE")
        print("=" * 60)
        
        self.test_basic_correctness()
        self.test_zero_regularization()
        self.test_numerical_gradient()
        self.test_different_weight_shapes()
        self.test_loss_formula()
        self.test_gradient_formula()
        self.test_different_regularization_strengths()
        self.test_loss_scaling()
        self.test_gradient_scaling()
        self.test_loss_positivity()
        self.test_vectorization()
        self.test_gradient_with_zero_weights()
        
        return self.print_summary()

# ============================================================================
# TODO: Add more test classes here for other functions
# ============================================================================

# class TestFullyConnectedLayer(TestBase):
#     """Test suite for fully connected layer forward/backward"""
#     pass

# class TestReLU(TestBase):
#     """Test suite for ReLU activation"""
#     pass

# class TestThreeLayerNet(TestBase):
#     """Test suite for ThreeLayerNet class"""
#     pass

# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all test suites"""
    print("\n" + "=" * 60)
    print("DEEP LEARNING HW2 - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    all_passed = True
    
    # Run Softmax Loss tests
    softmax_tester = TestSoftmaxLoss()
    softmax_passed = softmax_tester.run_all_tests()
    all_passed = all_passed and softmax_passed
    
    # Run L2 Regularization tests
    l2_tester = TestL2Regularization()
    l2_passed = l2_tester.run_all_tests()
    all_passed = all_passed and l2_passed
    
    # TODO: Add more test suites here
    
    # fc_tester = TestFullyConnectedLayer()
    # fc_passed = fc_tester.run_all_tests()
    # all_passed = all_passed and fc_passed
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    if all_passed:
        print("🎉 All test suites passed!")
    else:
        print("⚠️  Some tests failed. Please review the output above.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
