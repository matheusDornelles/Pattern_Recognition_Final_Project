# Pattern Recognition Final Project
Bishop's University Master's in Computer Science.
This project demonstrates different methods for handling missing data, focusing on imputation-based approaches.

## Project Structure

```
finalproject_pattern/
├── src/
│   ├── imputation/           # Imputation methods
│   │   ├── __init__.py
│   │   ├── single_value.py   # Default, mean, median imputation
│   │   ├── advanced.py       # Advanced imputation methods
│   │   └── base.py          # Base classes
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   ├── data_generator.py # Generate test data with missing values
│   │   └── evaluation.py    # Evaluation metrics
│   └── __init__.py
├── notebooks/               # Jupyter notebooks with examples
├── data/                   # Sample datasets
├── tests/                  # Unit tests
├── requirements.txt
└── README.md
```

## Imputation Methods Implemented

### 1. Single Value Imputation
- **Default Value**: Replace missing values with a predefined constant
- **Mean Imputation**: Replace missing values with the mean of the column
- **Median Imputation**: Replace missing values with the median of the column
- **Mode Imputation**: Replace missing values with the most frequent value

### 2. Advanced Methods
- **Forward/Backward Fill**: Propagate last/next valid observations
- **Interpolation**: Linear, polynomial, and spline interpolation
- **K-Nearest Neighbors**: Standard KNN imputation
- **Iterative Imputation**: Multivariate approach using round-robin

### 3. Group-based and Mathematical Methods
- **Group Center Imputation**: Impute using group-specific means/medians/modes
- **Partial Mean Imputation**: Use subset-based means for targeted imputation
- **Enhanced K-NN**: Advanced KNN with weighted distances and feature selection
- **SVD Imputation**: Matrix completion using Singular Value Decomposition

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

See the notebooks in the `notebooks/` directory for detailed examples and demonstrations of each imputation method.

## Features

- Modular design for easy extension
- Comprehensive evaluation metrics
- Support for both numerical and categorical data
- Visualization tools for comparing methods
- Test data generation utilities

## Author

Final Project - Gabriel Fernandes, Matheus Dornelles, Jose Navarro
