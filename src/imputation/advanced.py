"""
Advanced Imputation Methods

This module implements more sophisticated imputation techniques:
- Forward fill and backward fill
- Linear and polynomial interpolation
- K-Nearest Neighbors imputation
- Iterative imputation (experimental)
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Literal
from sklearn.impute import KNNImputer as SklearnKNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer as SklearnIterativeImputer
from .base import BaseImputer


class ForwardBackwardFillImputer(BaseImputer):
    """
    Impute missing values using forward fill or backward fill.
    
    This imputer fills missing values by propagating the last valid observation
    forward (forward fill) or the next valid observation backward (backward fill).
    
    Parameters
    ----------
    method : {'ffill', 'bfill'}, default='ffill'
        The fill method to use:
        - 'ffill': Forward fill (propagate last valid observation forward)
        - 'bfill': Backward fill (propagate next valid observation backward)
    limit : Optional[int], default=None
        Maximum number of consecutive missing values to fill
    """
    
    def __init__(self, method: Literal['ffill', 'bfill'] = 'ffill', limit: Optional[int] = None):
        super().__init__()
        if method not in ['ffill', 'bfill']:
            raise ValueError("method must be either 'ffill' or 'bfill'")
        self.method = method
        self.limit = limit
    
    def fit(self, X: pd.DataFrame) -> 'ForwardBackwardFillImputer':
        """
        Fit the imputer (no fitting required for fill methods).
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data
            
        Returns
        -------
        self : ForwardBackwardFillImputer
            The fitted imputer
        """
        self._validate_input(X)
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using forward or backward fill.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values
            
        Returns
        -------
        pd.DataFrame
            The data with missing values filled
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        self._validate_input(X)
        
        if self.method == 'ffill':
            return X.fillna(method='ffill', limit=self.limit)
        else:  # bfill
            return X.fillna(method='bfill', limit=self.limit)


class InterpolationImputer(BaseImputer):
    """
    Impute missing values using interpolation methods.
    
    This imputer fills missing values using various interpolation techniques.
    
    Parameters
    ----------
    method : str, default='linear'
        The interpolation method to use. Options include:
        - 'linear': Linear interpolation
        - 'polynomial': Polynomial interpolation
        - 'spline': Spline interpolation
        - 'time': Time-based interpolation
    order : int, default=1
        Order for polynomial and spline interpolation
    """
    
    def __init__(self, method: str = 'linear', order: int = 1):
        super().__init__()
        self.method = method
        self.order = order
    
    def fit(self, X: pd.DataFrame) -> 'InterpolationImputer':
        """
        Fit the imputer (no fitting required for interpolation).
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data
            
        Returns
        -------
        self : InterpolationImputer
            The fitted imputer
        """
        self._validate_input(X)
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using interpolation.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values
            
        Returns
        -------
        pd.DataFrame
            The data with missing values interpolated
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        self._validate_input(X)
        X_imputed = X.copy()
        
        # Apply interpolation only to numerical columns
        numeric_columns = X_imputed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if self.method in ['polynomial', 'spline']:
                X_imputed[col] = X_imputed[col].interpolate(method=self.method, order=self.order)
            else:
                X_imputed[col] = X_imputed[col].interpolate(method=self.method)
        
        return X_imputed


class KNNImputer(BaseImputer):
    """
    K-Nearest Neighbors imputation.
    
    This imputer fills missing values using the K-Nearest Neighbors approach.
    Each missing value is imputed using values from n_neighbors nearest neighbors
    found in the training set.
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighboring samples to use for imputation
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction
    """
    
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform'):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._imputer = SklearnKNNImputer(n_neighbors=n_neighbors, weights=weights)
    
    def fit(self, X: pd.DataFrame) -> 'KNNImputer':
        """
        Fit the K-NN imputer.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data
            
        Returns
        -------
        self : KNNImputer
            The fitted imputer
        """
        self._validate_input(X)
        
        # KNN imputation only works with numerical data
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(self.numeric_columns) == 0:
            raise ValueError("No numerical columns found for KNN imputation")
        
        X_numeric = X[self.numeric_columns]
        self._imputer.fit(X_numeric)
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using K-NN imputation.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values
            
        Returns
        -------
        pd.DataFrame
            The data with missing values imputed using K-NN
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        self._validate_input(X)
        X_imputed = X.copy()
        
        # Apply KNN imputation only to numerical columns
        X_numeric = X_imputed[self.numeric_columns]
        X_imputed_numeric = self._imputer.transform(X_numeric)
        
        # Replace the numerical columns with imputed values
        X_imputed[self.numeric_columns] = X_imputed_numeric
        
        return X_imputed


class IterativeImputer(BaseImputer):
    """
    Multivariate iterative imputation.
    
    This imputer uses iterative imputation to estimate missing values.
    Each feature with missing values is modeled as a function of other features
    in a round-robin fashion.
    
    Parameters
    ----------
    max_iter : int, default=10
        Maximum number of imputation rounds to perform
    random_state : Optional[int], default=None
        Random state for reproducibility
    """
    
    def __init__(self, max_iter: int = 10, random_state: Optional[int] = None):
        super().__init__()
        self.max_iter = max_iter
        self.random_state = random_state
        self._imputer = SklearnIterativeImputer(
            max_iter=max_iter, 
            random_state=random_state
        )
    
    def fit(self, X: pd.DataFrame) -> 'IterativeImputer':
        """
        Fit the iterative imputer.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data
            
        Returns
        -------
        self : IterativeImputer
            The fitted imputer
        """
        self._validate_input(X)
        
        # Iterative imputation only works with numerical data
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(self.numeric_columns) == 0:
            raise ValueError("No numerical columns found for iterative imputation")
        
        X_numeric = X[self.numeric_columns]
        self._imputer.fit(X_numeric)
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using iterative imputation.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values
            
        Returns
        -------
        pd.DataFrame
            The data with missing values imputed iteratively
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        self._validate_input(X)
        X_imputed = X.copy()
        
        # Apply iterative imputation only to numerical columns
        X_numeric = X_imputed[self.numeric_columns]
        X_imputed_numeric = self._imputer.transform(X_numeric)
        
        # Replace the numerical columns with imputed values
        X_imputed[self.numeric_columns] = X_imputed_numeric
        
        return X_imputed