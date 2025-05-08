# PCA Implementation for High-Dimensional Data

This repository contains a Python implementation of Principal Component Analysis (PCA), with a focus on handling high-dimensional data efficiently.

## Overview

Principal Component Analysis is a dimensionality reduction technique widely used in data science and machine learning. This implementation provides two approaches:

1. **Standard PCA**: Suitable for datasets where the number of features is manageable.
2. **High-Dimensional PCA**: An optimized implementation for datasets with a large number of features, using matrix manipulation techniques to improve computational efficiency.

## Functions

- `normalize(X)`: Normalizes data by centering it (subtracting the mean)
- `eig(S)`: Computes and sorts eigenvalues and eigenvectors in descending order
- `projection_matrix(B)`: Creates a projection matrix for dimensionality reduction
- `PCA(X, num_components)`: Standard PCA implementation
- `PCA_high_dim(X, num_components)`: PCA implementation optimized for high-dimensional data

## Usage

```python
import numpy as np
from PCA.PCA import PCA, PCA_high_dim

# Create sample data
random = np.random.RandomState(0)
X = random.randn(5, 4)  # 5 samples, 4 features

# Apply standard PCA
reconstructed_data, mean, principal_values, principal_components = PCA(X, 2)

# Apply high-dimensional PCA
reconstructed_data_hd, mean_hd, principal_values_hd, principal_components_hd = PCA_high_dim(X, 2)
```

## Algorithm Details

Both implementations return:
- Reconstructed data after dimensionality reduction
- Mean of the original data
- Principal values (eigenvalues)
- Principal components (eigenvectors)

The high-dimensional implementation uses the trick of computing X·X^T/N instead of the full covariance matrix, making it more efficient when the number of features exceeds the number of samples.

## Requirements

- NumPy 
