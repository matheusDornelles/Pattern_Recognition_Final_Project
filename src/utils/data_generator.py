"""
Data generation utilities for creating test datasets with missing values.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import random


class MissingDataGenerator:
    """
    Generate datasets with missing values for testing imputation methods.
    
    This class provides various methods to introduce missing values into datasets
    following different missing data mechanisms (MCAR, MAR, MNAR).
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the missing data generator.
        
        Parameters
        ----------
        random_state : Optional[int], default=None
            Random state for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
    
    def generate_sample_dataset(self, 
                               n_samples: int = 1000,
                               n_features: int = 5,
                               feature_types: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Generate a sample dataset with various feature types.
        
        Parameters
        ----------
        n_samples : int, default=1000
            Number of samples to generate
        n_features : int, default=5
            Number of features to generate
        feature_types : Optional[Dict[str, str]], default=None
            Dictionary mapping feature names to types ('numerical', 'categorical')
            
        Returns
        -------
        pd.DataFrame
            Generated dataset without missing values
        """
        data = {}
        
        for i in range(n_features):
            col_name = f"feature_{i+1}"
            
            # Determine feature type
            if feature_types and col_name in feature_types:
                feat_type = feature_types[col_name]
            else:
                # Default: mostly numerical with some categorical
                feat_type = 'numerical' if i < n_features * 0.7 else 'categorical'
            
            if feat_type == 'numerical':
                # Generate numerical data with different distributions
                if i % 3 == 0:
                    # Normal distribution
                    data[col_name] = np.random.normal(50, 15, n_samples)
                elif i % 3 == 1:
                    # Exponential distribution
                    data[col_name] = np.random.exponential(2, n_samples)
                else:
                    # Uniform distribution
                    data[col_name] = np.random.uniform(0, 100, n_samples)
            else:
                # Generate categorical data
                categories = [f"cat_{j}" for j in range(3 + i % 3)]  # Variable number of categories
                data[col_name] = np.random.choice(categories, n_samples)
        
        return pd.DataFrame(data)
    
    def introduce_mcar(self, 
                       data: pd.DataFrame,
                       missing_rate: float = 0.1,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Introduce Missing Completely At Random (MCAR) values.
        
        Parameters
        ----------
        data : pd.DataFrame
            The original dataset
        missing_rate : float, default=0.1
            Proportion of values to make missing (0 to 1)
        columns : Optional[List[str]], default=None
            Columns to introduce missing values in. If None, all columns are used.
            
        Returns
        -------
        pd.DataFrame
            Dataset with MCAR missing values
        """
        data_copy = data.copy()
        
        if columns is None:
            columns = data.columns.tolist()
        
        for col in columns:
            if col in data_copy.columns:
                n_missing = int(len(data_copy) * missing_rate)
                missing_indices = np.random.choice(
                    data_copy.index, 
                    size=n_missing, 
                    replace=False
                )
                data_copy.loc[missing_indices, col] = np.nan
        
        return data_copy
    
    def introduce_mar(self,
                      data: pd.DataFrame,
                      target_column: str,
                      condition_column: str,
                      condition_threshold: Union[float, str],
                      missing_rate: float = 0.3) -> pd.DataFrame:
        """
        Introduce Missing At Random (MAR) values.
        
        The missingness in target_column depends on the values in condition_column.
        
        Parameters
        ----------
        data : pd.DataFrame
            The original dataset
        target_column : str
            Column to introduce missing values in
        condition_column : str
            Column that determines missingness probability
        condition_threshold : Union[float, str]
            Threshold value for the condition
        missing_rate : float, default=0.3
            Proportion of values to make missing in the condition group
            
        Returns
        -------
        pd.DataFrame
            Dataset with MAR missing values
        """
        data_copy = data.copy()
        
        if target_column not in data_copy.columns or condition_column not in data_copy.columns:
            raise ValueError("Target or condition column not found in data")
        
        # Determine condition based on column type
        if pd.api.types.is_numeric_dtype(data_copy[condition_column]):
            condition_mask = data_copy[condition_column] > condition_threshold
        else:
            condition_mask = data_copy[condition_column] == condition_threshold
        
        # Introduce missing values in the condition group
        condition_indices = data_copy[condition_mask].index
        n_missing = int(len(condition_indices) * missing_rate)
        
        if n_missing > 0:
            missing_indices = np.random.choice(
                condition_indices,
                size=n_missing,
                replace=False
            )
            data_copy.loc[missing_indices, target_column] = np.nan
        
        return data_copy
    
    def introduce_mnar(self,
                       data: pd.DataFrame,
                       target_column: str,
                       missing_mechanism: str = 'tail',
                       missing_rate: float = 0.2) -> pd.DataFrame:
        """
        Introduce Missing Not At Random (MNAR) values.
        
        The missingness depends on the unobserved values themselves.
        
        Parameters
        ----------
        data : pd.DataFrame
            The original dataset
        target_column : str
            Column to introduce missing values in
        missing_mechanism : str, default='tail'
            Type of MNAR mechanism:
            - 'tail': Missing values in the tail (high or low values)
            - 'middle': Missing values in the middle range
        missing_rate : float, default=0.2
            Proportion of values to make missing
            
        Returns
        -------
        pd.DataFrame
            Dataset with MNAR missing values
        """
        data_copy = data.copy()
        
        if target_column not in data_copy.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        if not pd.api.types.is_numeric_dtype(data_copy[target_column]):
            raise ValueError("MNAR mechanism requires numerical target column")
        
        if missing_mechanism == 'tail':
            # Missing values in the upper tail
            threshold = data_copy[target_column].quantile(1 - missing_rate)
            missing_mask = data_copy[target_column] > threshold
        elif missing_mechanism == 'middle':
            # Missing values in the middle range
            lower_bound = data_copy[target_column].quantile(0.5 - missing_rate/2)
            upper_bound = data_copy[target_column].quantile(0.5 + missing_rate/2)
            missing_mask = (data_copy[target_column] >= lower_bound) & \
                          (data_copy[target_column] <= upper_bound)
        else:
            raise ValueError("missing_mechanism must be 'tail' or 'middle'")
        
        data_copy.loc[missing_mask, target_column] = np.nan
        
        return data_copy
    
    def create_mixed_missing_pattern(self,
                                   data: pd.DataFrame,
                                   patterns: List[Dict]) -> pd.DataFrame:
        """
        Create complex missing patterns by combining different mechanisms.
        
        Parameters
        ----------
        data : pd.DataFrame
            The original dataset
        patterns : List[Dict]
            List of pattern configurations, each containing:
            - 'type': 'mcar', 'mar', or 'mnar'
            - 'columns': List of columns
            - 'rate': Missing rate
            - Additional parameters specific to each type
            
        Returns
        -------
        pd.DataFrame
            Dataset with complex missing patterns
        """
        data_copy = data.copy()
        
        for pattern in patterns:
            pattern_type = pattern['type']
            
            if pattern_type == 'mcar':
                data_copy = self.introduce_mcar(
                    data_copy,
                    missing_rate=pattern['rate'],
                    columns=pattern['columns']
                )
            elif pattern_type == 'mar':
                data_copy = self.introduce_mar(
                    data_copy,
                    target_column=pattern['target_column'],
                    condition_column=pattern['condition_column'],
                    condition_threshold=pattern['condition_threshold'],
                    missing_rate=pattern['rate']
                )
            elif pattern_type == 'mnar':
                data_copy = self.introduce_mnar(
                    data_copy,
                    target_column=pattern['target_column'],
                    missing_mechanism=pattern.get('mechanism', 'tail'),
                    missing_rate=pattern['rate']
                )
        
        return data_copy


def load_sample_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load pre-defined sample datasets for testing.
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing sample datasets
    """
    datasets = {}
    
    # Generate a simple numerical dataset
    generator = MissingDataGenerator(random_state=42)
    
    # Dataset 1: Pure numerical
    datasets['numerical'] = generator.generate_sample_dataset(
        n_samples=500,
        n_features=4,
        feature_types={
            'feature_1': 'numerical',
            'feature_2': 'numerical', 
            'feature_3': 'numerical',
            'feature_4': 'numerical'
        }
    )
    
    # Dataset 2: Mixed types
    datasets['mixed'] = generator.generate_sample_dataset(
        n_samples=800,
        n_features=6,
        feature_types={
            'feature_1': 'numerical',
            'feature_2': 'numerical',
            'feature_3': 'numerical',
            'feature_4': 'categorical',
            'feature_5': 'categorical',
            'feature_6': 'categorical'
        }
    )
    
    # Dataset 3: Time series-like data
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    datasets['timeseries'] = pd.DataFrame({
        'date': dates,
        'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(365) / 365) + \
                      np.random.normal(0, 2, 365),
        'humidity': 50 + 20 * np.cos(2 * np.pi * np.arange(365) / 365) + \
                   np.random.normal(0, 5, 365),
        'pressure': 1013 + np.random.normal(0, 10, 365)
    })
    
    return datasets