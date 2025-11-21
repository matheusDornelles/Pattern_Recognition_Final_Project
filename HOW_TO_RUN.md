# How to Run the Missing Data Handling Final Project

## Prerequisites

1. **Python 3.7 or higher**
2. **Required packages** (install using the commands below)

## Installation Steps

### 1. Navigate to the Project Directory

```bash
cd c:\Users\rayss\OneDrive\Desktop\finalproject_pattern
```

### 2. Install Required Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter scipy
```

### 3. Set Up Python Path

The project uses relative imports. Make sure to run Python from the project root directory.

## Running the Project

### Option 1: Run the Jupyter Notebook (Recommended)

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Navigate to the `notebooks/` folder

3. Open `single_value_imputation_demo.ipynb`

4. Run all cells sequentially to see the complete demonstration

### Option 2: Run as Python Script

You can also run individual components as Python scripts:

```bash
# Run from project root directory
python -c "
import sys
sys.path.append('src')
from imputation.single_value import MeanImputer
from utils.data_generator import MissingDataGenerator
import pandas as pd
import numpy as np

# Quick test
generator = MissingDataGenerator(random_state=42)
data = generator.generate_sample_dataset(100, 3)
data_missing = generator.introduce_mcar(data, missing_rate=0.1)

imputer = MeanImputer()
result = imputer.fit_transform(data_missing.select_dtypes(include=[np.number]))
print('Imputation successful!')
print('Missing values before:', data_missing.isnull().sum().sum())
print('Missing values after:', result.isnull().sum().sum())
"
```

### Option 3: Run Tests

To verify everything works correctly:

```bash
python -m pytest tests/ -v
```

Or run the test file directly:
```bash
python tests/test_imputation.py
```

## Project Structure Guide

```
finalproject_pattern/
├── src/                     # Source code
│   ├── imputation/         # Imputation methods
│   │   ├── single_value.py # Basic imputation methods
│   │   ├── advanced.py     # Advanced methods
│   │   └── base.py         # Base classes
│   └── utils/              # Utility functions
│       ├── data_generator.py # Test data generation
│       └── evaluation.py    # Performance evaluation
├── notebooks/              # Jupyter demonstrations
│   └── single_value_imputation_demo.ipynb
├── tests/                  # Unit tests
├── data/                   # Sample datasets (empty initially)
└── requirements.txt        # Dependencies
```

## Usage Examples

### Basic Mean Imputation

```python
from src.imputation.single_value import MeanImputer
import pandas as pd
import numpy as np

# Create data with missing values
data = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [10, np.nan, 30, 40, 50]
})

# Apply mean imputation
imputer = MeanImputer()
result = imputer.fit_transform(data)
print(result)
```

### Generate Test Data with Missing Values

```python
from src.utils.data_generator import MissingDataGenerator

# Create generator
generator = MissingDataGenerator(random_state=42)

# Generate sample data
data = generator.generate_sample_dataset(n_samples=1000, n_features=5)

# Introduce missing values (MCAR pattern)
data_with_missing = generator.introduce_mcar(data, missing_rate=0.15)

print(f"Original shape: {data.shape}")
print(f"Missing values: {data_with_missing.isnull().sum().sum()}")
```

### Compare Multiple Imputation Methods

```python
from src.imputation.single_value import MeanImputer, MedianImputer, ModeImputer
from src.utils.evaluation import ImputationEvaluator

# Apply different imputation methods
mean_imputer = MeanImputer()
median_imputer = MedianImputer()

# Get numerical columns
numerical_cols = data_with_missing.select_dtypes(include=[np.number]).columns

# Apply imputations
mean_result = mean_imputer.fit_transform(data_with_missing[numerical_cols])
median_result = median_imputer.fit_transform(data_with_missing[numerical_cols])

# Evaluate performance
evaluator = ImputationEvaluator()
missing_mask = data_with_missing.isnull()

mean_metrics = evaluator.evaluate_numerical_imputation(
    original_data, mean_result, missing_mask
)
print("Mean Imputation Metrics:", mean_metrics)
```

## Troubleshooting

### Common Issues and Solutions

1. **ModuleNotFoundError**: Make sure you're running from the project root directory and that the `src` folder is in your Python path.

2. **Missing Dependencies**: Install all required packages using `pip install -r requirements.txt`

3. **Jupyter Kernel Issues**: If the notebook doesn't run properly, try restarting the kernel: Kernel → Restart & Clear Output

4. **Memory Issues**: If working with large datasets, try reducing the sample size in the data generator.

5. **Plotting Issues**: If plots don't show up, try running `%matplotlib inline` in Jupyter or ensure matplotlib backend is properly configured.

### Performance Tips

1. **For large datasets**: Use chunk processing or consider more efficient imputation methods
2. **For categorical data**: Mode imputation is typically fastest
3. **For numerical data**: Mean and median imputation are very fast; KNN is slower but more accurate

## Expected Outputs

When you run the complete notebook, you should see:

1. **Data Exploration**: Visualizations of missing data patterns
2. **Imputation Results**: Before/after comparisons for each method
3. **Performance Metrics**: Quantitative evaluation of different methods
4. **Distribution Analysis**: How imputation affects data distribution
5. **Recommendations**: Guidelines for choosing appropriate methods

## Next Steps

After running the basic demonstration, consider:

1. Experimenting with different missing data mechanisms (MAR, MNAR)
2. Testing on real datasets
3. Implementing custom imputation strategies
4. Exploring more advanced methods like iterative imputation
5. Building pipelines that combine multiple techniques

For questions or issues, refer to the comments in the source code or the detailed documentation in the Jupyter notebook.