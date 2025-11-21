# Missing Data Handling - Method Documentation

This document provides detailed information about the imputation methods implemented in this final project.

## Single Value Imputation Methods

### 1. Default Value Imputation

**Class**: `DefaultValueImputer`

**Purpose**: Replaces missing values with predetermined constant values.

**Use Cases**:
- When domain knowledge suggests specific default values
- For categorical variables where "unknown" or "other" is appropriate
- When simple, fast imputation is needed

**Parameters**:
- `default_value`: Single value or dictionary mapping column names to default values

**Example**:
```python
# Single default value for all columns
imputer = DefaultValueImputer(default_value=0)

# Different defaults for different columns
imputer = DefaultValueImputer(default_value={
    'age': 30,
    'category': 'unknown'
})
```

**Advantages**:
- Simple and fast
- Preserves original data type
- Domain knowledge can be incorporated

**Disadvantages**:
- Can introduce bias
- May not reflect true data distribution
- Can create artificial patterns

---

### 2. Mean Imputation

**Class**: `MeanImputer`

**Purpose**: Replaces missing values with the arithmetic mean of each column.

**Use Cases**:
- Normally distributed numerical data
- When maintaining central tendency is important
- Quick imputation for preliminary analysis

**Parameters**:
- `columns`: List of columns to apply mean imputation (default: all numerical columns)

**Example**:
```python
# Apply to all numerical columns
imputer = MeanImputer()

# Apply to specific columns only
imputer = MeanImputer(columns=['age', 'income'])
```

**Mathematical Formula**:
```
μ = (Σ xi) / n
where xi are non-missing values and n is count of non-missing values
```

**Advantages**:
- Maintains sample mean
- Simple to understand and implement
- Works well with normal distributions

**Disadvantages**:
- Reduces variance
- Sensitive to outliers
- Can distort distribution shape

---

### 3. Median Imputation

**Class**: `MedianImputer`

**Purpose**: Replaces missing values with the median value of each column.

**Use Cases**:
- Skewed numerical data
- Data with outliers
- When robust central tendency is needed

**Parameters**:
- `columns`: List of columns to apply median imputation (default: all numerical columns)

**Example**:
```python
# Apply to all numerical columns
imputer = MedianImputer()

# Apply to specific columns
imputer = MedianImputer(columns=['score', 'rating'])
```

**Mathematical Formula**:
```
For odd n: median = x[(n+1)/2]
For even n: median = (x[n/2] + x[n/2+1]) / 2
where x is sorted array of non-missing values
```

**Advantages**:
- Robust to outliers
- Maintains central tendency for skewed data
- Less affected by extreme values than mean

**Disadvantages**:
- Still reduces variance
- May not preserve distribution shape
- Less intuitive than mean for some users

---

### 4. Mode Imputation

**Class**: `ModeImputer`

**Purpose**: Replaces missing values with the most frequently occurring value in each column.

**Use Cases**:
- Categorical variables
- Nominal data
- When preserving most common category is important

**Parameters**:
- `columns`: List of columns to apply mode imputation (default: all columns)

**Example**:
```python
# Apply to all columns
imputer = ModeImputer()

# Apply to specific categorical columns
imputer = ModeImputer(columns=['category', 'region', 'status'])
```

**Behavior**:
- For multiple modes, uses the first one encountered
- Works with both categorical and numerical data
- Returns None if all values are missing

**Advantages**:
- Natural for categorical data
- Preserves most common patterns
- Simple and interpretable

**Disadvantages**:
- Can artificially increase frequency of common categories
- May not be appropriate for continuous variables
- Doesn't account for relationships between variables

---

## Advanced Imputation Methods

### 5. Forward/Backward Fill

**Class**: `ForwardBackwardFillImputer`

**Purpose**: Fills missing values by propagating previous (forward fill) or next (backward fill) valid observations.

**Use Cases**:
- Time series data
- Sequential data where order matters
- When temporal continuity is important

**Parameters**:
- `method`: 'ffill' (forward fill) or 'bfill' (backward fill)
- `limit`: Maximum number of consecutive missing values to fill

---

### 6. K-Nearest Neighbors (KNN) Imputation

**Class**: `KNNImputer`

**Purpose**: Imputes missing values using the K-nearest neighbors approach.

**Use Cases**:
- When relationships between variables are important
- Mixed-type data (after preprocessing)
- When higher accuracy is needed

**Parameters**:
- `n_neighbors`: Number of neighboring samples to use
- `weights`: 'uniform' or 'distance' weighting

---

### 7. Iterative Imputation

**Class**: `IterativeImputer`

**Purpose**: Models each feature with missing values as a function of other features in a round-robin fashion.

**Use Cases**:
- Complex multivariate relationships
- When high accuracy is critical
- Research or analytical contexts

**Parameters**:
- `max_iter`: Maximum number of imputation rounds
- `random_state`: Random seed for reproducibility

---

## Method Selection Guidelines

### Decision Tree for Method Selection

```
Is the variable categorical?
├── Yes → Use Mode Imputation
└── No → Is the data normally distributed?
    ├── Yes → Use Mean Imputation
    └── No → Are there outliers?
        ├── Yes → Use Median Imputation
        └── No → Consider advanced methods (KNN, Iterative)
```

### Performance vs Complexity Trade-off

| Method | Speed | Accuracy | Complexity | Best For |
|--------|-------|----------|------------|----------|
| Default | ★★★★★ | ★★☆☆☆ | ★☆☆☆☆ | Simple categorical |
| Mean | ★★★★★ | ★★★☆☆ | ★☆☆☆☆ | Normal distributions |
| Median | ★★★★★ | ★★★☆☆ | ★☆☆☆☆ | Skewed distributions |
| Mode | ★★★★★ | ★★★☆☆ | ★☆☆☆☆ | Categorical variables |
| KNN | ★★☆☆☆ | ★★★★☆ | ★★★☆☆ | Complex relationships |
| Iterative | ★☆☆☆☆ | ★★★★★ | ★★★★☆ | Research contexts |

### Considerations by Data Type

**Numerical Variables**:
- **Continuous**: Mean (normal) or Median (skewed)
- **Discrete**: Mode or advanced methods
- **With outliers**: Median or robust methods

**Categorical Variables**:
- **Nominal**: Mode imputation
- **Ordinal**: Mode or advanced methods considering order

**Time Series**:
- Forward/backward fill
- Interpolation methods
- Seasonal decomposition approaches

### Missing Data Mechanism Considerations

**MCAR (Missing Completely At Random)**:
- Any method can work
- Simple methods often sufficient

**MAR (Missing At Random)**:
- Advanced methods preferred (KNN, Iterative)
- Consider relationships with observed variables

**MNAR (Missing Not At Random)**:
- Domain-specific approaches needed
- May require modeling the missing mechanism

## Implementation Notes

### Error Handling

All imputers include:
- Input validation (DataFrame type, empty data checks)
- Fitting status verification
- Missing column handling

### Performance Optimization

- Single value methods are vectorized
- KNN uses sklearn's optimized implementation
- Memory usage is monitored for large datasets

### Extensibility

The base `BaseImputer` class allows easy extension:
```python
class CustomImputer(BaseImputer):
    def fit(self, X):
        # Custom fitting logic
        return self
    
    def transform(self, X):
        # Custom transformation logic
        return X_transformed
```

## Evaluation Metrics

### Numerical Variables
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **NRMSE**: Normalized RMSE

### Categorical Variables
- **Accuracy**: Proportion of correct classifications
- **Jensen-Shannon Divergence**: Distribution similarity

### Distribution Metrics
- Mean preservation
- Variance changes
- Skewness and kurtosis impact

## References and Further Reading

1. Little, R.J.A. and Rubin, D.B. (2019). Statistical Analysis with Missing Data, 3rd Edition.
2. Schafer, J.L. (1997). Analysis of Incomplete Multivariate Data.
3. Van Buuren, S. (2018). Flexible Imputation of Missing Data, 2nd Edition.
4. Scikit-learn documentation on imputation: https://scikit-learn.org/stable/modules/impute.html