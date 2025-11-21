"""
Base classes for imputation methods
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Union, Any, Optional


class BaseImputer(ABC):
    """
    Abstract base class for all imputation methods.
    
    All imputation methods should inherit from this class and implement
    the fit and transform methods.
    """
    
    def __init__(self):
        self.is_fitted = False
        self.statistics_ = {}
    
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> 'BaseImputer':
        """
        Fit the imputer to the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with potential missing values
            
        Returns
        -------
        self : BaseImputer
            The fitted imputer instance
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by imputing missing values.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with potential missing values
            
        Returns
        -------
        pd.DataFrame
            The data with missing values imputed
        """
        pass
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the imputer and transform the data in one step.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with potential missing values
            
        Returns
        -------
        pd.DataFrame
            The data with missing values imputed
        """
        return self.fit(X).transform(X)
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """
        Validate input data.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data to validate
            
        Raises
        ------
        TypeError
            If X is not a pandas DataFrame
        ValueError
            If X is empty
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if X.empty:
            raise ValueError("Input DataFrame is empty")
    
    def get_missing_info(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get information about missing values in the dataset.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data
            
        Returns
        -------
        pd.DataFrame
            DataFrame with missing value statistics for each column
        """
        missing_info = pd.DataFrame({
            'column': X.columns,
            'missing_count': X.isnull().sum().values,
            'missing_percentage': (X.isnull().sum() / len(X) * 100).values,
            'dtype': X.dtypes.values
        })
        
        return missing_info.sort_values('missing_percentage', ascending=False)