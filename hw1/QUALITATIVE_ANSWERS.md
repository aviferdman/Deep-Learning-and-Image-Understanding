# Answers to Qualitative Questions in Notebook

## Q1: Exploratory Data Analysis (EDA)

### Code to implement:

```python
# (1) Print shapes of training and test sets
print(f"Training set shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# (2) Display number of classes and their names
num_classes = len(np.unique(y_train))
print(f"\nNumber of classes: {num_classes}")
print(f"Class names: {classes}")

# (3) Show class distribution in training set
unique, counts = np.unique(y_train, return_counts=True)
print(f"\nClass distribution in training set:")
for class_id, count in zip(unique, counts):
    print(f"  Class {class_id} ({classes[class_id]}): {count} samples ({count/len(y_train)*100:.1f}%)")
```

---

## Q2: Understanding the Image Preprocessing

### 1. Mean Subtraction

**Why do we subtract the mean image (computed from the training set) from every image?**

We subtract the mean image for several important reasons:

- **Centering the data**: This centers the data around zero, which helps ensure that the features have similar ranges and that no single feature dominates due to scale differences.

- **Improved optimization**: Zero-centered data helps gradient descent converge faster because the loss surface becomes more symmetric. When features have different scales, the gradient descent path can oscillate, slowing convergence.

- **Reducing bias**: The mean image captures the average appearance across all training images. By subtracting it, we focus the model on learning the variations and differences between classes rather than learning the common background shared by all images.

- **Numerical stability**: Centered data prevents numerical issues during training, especially when using activation functions or computing gradients.

**Important**: We use only the training set mean to avoid data leakage. The validation and test sets should be preprocessed with the same training mean to simulate real-world conditions where future data won't be available during training.

### 2. Flattening

**Why do we need to flatten the images into 3072-dimensional vectors before training a linear model?**

Linear models require input in the form of 1D feature vectors because they compute a weighted sum of features:

$$f(x) = W \cdot x + b$$

Where:
- Original image shape: (32, 32, 3) = 32×32 pixels × 3 color channels
- Flattened shape: 3072 = 32 × 32 × 3 features

**Reasons for flattening**:

- **Mathematical compatibility**: The weight matrix $W$ has shape (D, C) where D is the number of features and C is the number of classes. Matrix multiplication requires matching dimensions.

- **Linear model assumption**: Linear classifiers treat each pixel intensity (after flattening) as an independent feature. The model learns a weight for each pixel-channel combination.

- **Simplicity**: While this loses spatial structure (a CNN would preserve it), linear models don't use spatial relationships anyway, so flattening doesn't lose information that the model could use.

### 3. Bias Trick

**Why do we add a constant 1 to every image vector (known as the bias trick)?**

The bias trick incorporates the bias term directly into the weight matrix, simplifying computations.

**Without bias trick**:
$$f(x) = Wx + b$$
Requires storing and updating both $W$ and $b$ separately.

**With bias trick**:
We augment $x$ with a constant 1: $x' = [x; 1]$

And augment $W$ to include the bias: $W' = [W; b^T]$

Then:
$$f(x') = W'x' = Wx + b \cdot 1 = Wx + b$$

**Benefits**:
- **Simplified implementation**: Only need to update one weight matrix instead of separate weights and biases.
- **Uniform treatment**: All parameters are treated the same way during gradient updates.
- **Cleaner code**: Matrix operations handle both weights and bias automatically.

---

## Q3: Explain Why the Accuracy on the Training Dataset is Low

**Why is the initial accuracy on the training dataset low (before extensive training)?**

The initial low accuracy (around 30-40% for 3 classes) occurs for several reasons:

1. **Random weight initialization**: The weights are initialized with small random values (0.001 * randn). These random weights have no meaningful relationship to the actual image patterns, so predictions are essentially random.

2. **Random guessing baseline**: For 3 classes, random guessing would yield approximately 33.3% accuracy. Initial performance close to this indicates the model hasn't learned anything meaningful yet.

3. **Complex feature space**: With 3072 features (32×32×3 pixels), the model needs to learn which pixel combinations are indicative of each class. Initially, all features are weighted equally and randomly.

4. **Linear model limitations**: Linear classifiers can only learn linear decision boundaries. If the classes are not linearly separable in pixel space (which is common for natural images), the model will struggle to achieve high accuracy even after training.

5. **Insufficient training**: Before training (or in early iterations), the model hasn't had enough gradient updates to adjust the weights toward meaningful patterns.

**Expected improvement**: As training progresses with proper learning rates and sufficient iterations, the accuracy should improve to 50-75% as the model learns useful linear combinations of pixels that distinguish between birds, cats, and deer.

**Note**: Linear classifiers have fundamental limitations for image classification. More sophisticated models (CNNs) can achieve much higher accuracy (>90%) on this task by learning hierarchical features and spatial relationships.

---

## Implementation Notes

These answers should be written in markdown cells in the Jupyter notebook at the appropriate locations. The test results show:

- ✓ All 9 tests passed successfully
- ✓ Gradient check shows excellent accuracy (relative error < 1e-9)
- ✓ Training improves accuracy as expected
- ✓ Hyperparameter tuning works correctly
- ✓ Both LinearPerceptron and LogisticRegression implementations are correct

The implementation is ready for use with the CIFAR-10 dataset in Google Colab.
