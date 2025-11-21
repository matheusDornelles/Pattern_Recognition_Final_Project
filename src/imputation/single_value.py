"""
Single Value Imputation Methods

This module implements basic imputation methods using single values:
- Default value imputation
- Mean imputation
- Median imputation
"""

import pandas as pd
import numpy as np
from typing import Union, Any, Dict, Optional
from .base import BaseImputer


class DefaultValueImputer(BaseImputer):
    """
    Impute missing values with a default/constant value.
    
    This imputer replaces missing values with a user-specified constant value.
    Different values can be specified for different columns.
    
    Parameters
    ----------
    default_value : Union[Any, Dict[str, Any]], default=0
        The default value to use for imputation. Can be:
        - A single value to use for all columns
        - A dictionary mapping column names to their default values
    """
    
    def __init__(self, default_value: Union[Any, Dict[str, Any]] = 0):
        super().__init__()
        self.default_value = default_value
    
    def fit(self, X: pd.DataFrame) -> 'DefaultValueImputer':
        """
        Fit the imputer (no fitting required for default value imputation).
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data
            
        Returns
        -------
        self : DefaultValueImputer
            The fitted imputer
        """
        self._validate_input(X)
        
        # Store the default values for each column
        if isinstance(self.default_value, dict):
            self.statistics_ = self.default_value.copy()
        else:
            self.statistics_ = {col: self.default_value for col in X.columns}
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by replacing missing values with default values.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values
            
        Returns
        -------
        pd.DataFrame
            The data with missing values replaced by default values
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        self._validate_input(X)
        X_imputed = X.copy()
        
        for col in X_imputed.columns:
            if col in self.statistics_:
                X_imputed[col] = X_imputed[col].fillna(self.statistics_[col])
        
        return X_imputed


class MeanImputer(BaseImputer):
    """
    Impute missing values with the mean of each column.
    
    This imputer replaces missing values with the mean value of the respective column.
    Only works with numerical columns.
    
    Parameters
    ----------
    columns : Optional[List[str]], default=None
        List of columns to apply mean imputation to. If None, applies to all numerical columns.
    """
    
    def __init__(self, columns: Optional[list] = None):
        super().__init__()
        self.columns = columns
    
    def fit(self, X: pd.DataFrame) -> 'MeanImputer':
        """
        Fit the imputer by computing the mean of each numerical column.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data
            
        Returns
        -------
        self : MeanImputer
            The fitted imputer
        """
        self._validate_input(X)
        
        # Determine which columns to impute
        if self.columns is None:
            # Use all numerical columns
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Use specified columns, but verify they're numerical
            numeric_columns = []
            for col in self.columns:
                if col in X.columns and X[col].dtype in [np.number, 'int64', 'float64', 'int32', 'float32']:
                    numeric_columns.append(col)
                else:
                    print(f"Warning: Column '{col}' is not numerical or doesn't exist. Skipping.")
        
        # Compute means
        self.statistics_ = {}
        for col in numeric_columns:
            if X[col].notna().any():  # Check if there are any non-null values
                self.statistics_[col] = X[col].mean()
            else:
                self.statistics_[col] = 0  # Default to 0 if all values are missing
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by replacing missing values with column means.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values
            
        Returns
        -------
        pd.DataFrame
            The data with missing values replaced by column means
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        self._validate_input(X)
        X_imputed = X.copy()
        
        for col, mean_value in self.statistics_.items():
            if col in X_imputed.columns:
                X_imputed[col] = X_imputed[col].fillna(mean_value)
        
        return X_imputed


class MedianImputer(BaseImputer):
    """
    Impute missing values with the median of each column.
    
    This imputer replaces missing values with the median value of the respective column.
    Only works with numerical columns. More robust to outliers than mean imputation.
    
    Parameters
    ----------
    columns : Optional[List[str]], default=None
        List of columns to apply median imputation to. If None, applies to all numerical columns.
    """
    
    def __init__(self, columns: Optional[list] = None):
        super().__init__()
        self.columns = columns
    
    def fit(self, X: pd.DataFrame) -> 'MedianImputer':
        """
        Fit the imputer by computing the median of each numerical column.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data
            
        Returns
        -------
        self : MedianImputer
            The fitted imputer
        """
        self._validate_input(X)
        
        # Determine which columns to impute
        if self.columns is None:
            # Use all numerical columns
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Use specified columns, but verify they're numerical
            numeric_columns = []
            for col in self.columns:
                if col in X.columns and X[col].dtype in [np.number, 'int64', 'float64', 'int32', 'float32']:
                    numeric_columns.append(col)
                else:
                    print(f"Warning: Column '{col}' is not numerical or doesn't exist. Skipping.")
        
        # Compute medians
        self.statistics_ = {}
        for col in numeric_columns:
            if X[col].notna().any():  # Check if there are any non-null values
                self.statistics_[col] = X[col].median()
            else:
                self.statistics_[col] = 0  # Default to 0 if all values are missing
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by replacing missing values with column medians.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values
            
        Returns
        -------
        pd.DataFrame
            The data with missing values replaced by column medians
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        self._validate_input(X)
        X_imputed = X.copy()
        
        for col, median_value in self.statistics_.items():
            if col in X_imputed.columns:
                X_imputed[col] = X_imputed[col].fillna(median_value)
        
        return X_imputed


class ModeImputer(BaseImputer):
    """
    Impute missing values with the mode (most frequent value) of each column.
    
    This imputer replaces missing values with the most frequent value in each column.
    Works with both numerical and categorical columns.
    
    Parameters
    ----------
    columns : Optional[List[str]], default=None
        List of columns to apply mode imputation to. If None, applies to all columns.
    """
    
    def __init__(self, columns: Optional[list] = None):
        super().__init__()
        self.columns = columns
    
    def fit(self, X: pd.DataFrame) -> 'ModeImputer':
        """
        Fit the imputer by computing the mode of each column.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data
            
        Returns
        -------
        self : ModeImputer
            The fitted imputer
        """
        self._validate_input(X)
        
        # Determine which columns to impute
        if self.columns is None:
            target_columns = X.columns.tolist()
        else:
            target_columns = [col for col in self.columns if col in X.columns]
        
        # Compute modes
        self.statistics_ = {}
        for col in target_columns:
            if X[col].notna().any():  # Check if there are any non-null values
                mode_values = X[col].mode()
                if len(mode_values) > 0:
                    self.statistics_[col] = mode_values.iloc[0]  # Take first mode if multiple
                else:
                    self.statistics_[col] = None  # No mode available
            else:
                self.statistics_[col] = None  # All values are missing
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by replacing missing values with column modes.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values
            
        Returns
        -------
        pd.DataFrame
            The data with missing values replaced by column modes
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        self._validate_input(X)
        X_imputed = X.copy()
        
        for col, mode_value in self.statistics_.items():
            if col in X_imputed.columns and mode_value is not None:
                X_imputed[col] = X_imputed[col].fillna(mode_value)
        
        return X_imputed