"""
Advanced Group-based and Mathematical Imputation Methods

This module implements sophisticated imputation techniques including:
- Group center imputation
- Partial mean imputation  
- SVD-based imputation
- Enhanced KNN variations
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Any
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from .base import BaseImputer


class GroupCenterImputer(BaseImputer):
    """
    Impute missing values using the center (mean/median/mode) of subgroups.
    
    This imputer divides data into groups based on one or more grouping variables
    and imputes missing values using the central tendency of each group.
    
    Parameters
    ----------
    group_columns : Union[str, List[str]]
        Column(s) to use for grouping
    strategy : str, default='mean'
        Strategy for computing group center ('mean', 'median', 'mode')
    fallback_strategy : str, default='overall'
        What to do when group has no valid values ('overall', 'drop')
    """
    
    def __init__(self, 
                 group_columns: Union[str, List[str]], 
                 strategy: str = 'mean',
                 fallback_strategy: str = 'overall'):
        super().__init__()
        self.group_columns = [group_columns] if isinstance(group_columns, str) else group_columns
        self.strategy = strategy
        self.fallback_strategy = fallback_strategy
        
        if strategy not in ['mean', 'median', 'mode']:
            raise ValueError("strategy must be 'mean', 'median', or 'mode'")
        if fallback_strategy not in ['overall', 'drop']:
            raise ValueError("fallback_strategy must be 'overall' or 'drop'")
    
    def fit(self, X: pd.DataFrame) -> 'GroupCenterImputer':
        """
        Fit the imputer by computing group centers.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data
            
        Returns
        -------
        self : GroupCenterImputer
            The fitted imputer
        """
        self._validate_input(X)
        
        # Verify group columns exist
        missing_cols = [col for col in self.group_columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Group columns not found: {missing_cols}")
        
        # Store group statistics for each target column
        self.group_stats_ = {}
        self.overall_stats_ = {}
        
        # Get columns to impute (exclude group columns)
        target_columns = [col for col in X.columns if col not in self.group_columns]
        
        for col in target_columns:
            if X[col].isnull().all():
                continue  # Skip columns with all missing values
                
            self.group_stats_[col] = {}
            
            # Calculate overall statistic as fallback
            if self.strategy == 'mean' and pd.api.types.is_numeric_dtype(X[col]):
                self.overall_stats_[col] = X[col].mean()
            elif self.strategy == 'median' and pd.api.types.is_numeric_dtype(X[col]):
                self.overall_stats_[col] = X[col].median()
            elif self.strategy == 'mode':
                mode_vals = X[col].mode()
                self.overall_stats_[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else None
            
            # Calculate group-wise statistics
            grouped = X.groupby(self.group_columns, dropna=False)[col]
            
            for group_key, group_data in grouped:
                if group_data.notna().any():  # If group has valid values
                    if self.strategy == 'mean' and pd.api.types.is_numeric_dtype(X[col]):
                        stat_val = group_data.mean()
                    elif self.strategy == 'median' and pd.api.types.is_numeric_dtype(X[col]):
                        stat_val = group_data.median()
                    elif self.strategy == 'mode':
                        mode_vals = group_data.mode()
                        stat_val = mode_vals.iloc[0] if len(mode_vals) > 0 else self.overall_stats_.get(col)
                    else:
                        stat_val = self.overall_stats_.get(col)
                    
                    self.group_stats_[col][group_key] = stat_val
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using group-based imputation.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values
            
        Returns
        -------
        pd.DataFrame
            The data with missing values imputed using group centers
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        self._validate_input(X)
        X_imputed = X.copy()
        
        for col in self.group_stats_.keys():
            if col not in X_imputed.columns:
                continue
                
            missing_mask = X_imputed[col].isnull()
            if not missing_mask.any():
                continue
            
            # Group the data
            grouped = X_imputed.groupby(self.group_columns, dropna=False)
            
            for group_key, group_indices in grouped.groups.items():
                group_missing = missing_mask.loc[group_indices]
                
                if group_missing.any():
                    # Get the statistic for this group
                    if group_key in self.group_stats_[col]:
                        fill_value = self.group_stats_[col][group_key]
                    elif self.fallback_strategy == 'overall':
                        fill_value = self.overall_stats_.get(col)
                    else:
                        continue  # Skip imputation for this group
                    
                    if fill_value is not None and not pd.isna(fill_value):
                        X_imputed.loc[group_indices[group_missing], col] = fill_value
        
        return X_imputed


class PartialMeanImputer(BaseImputer):
    """
    Impute missing values using partial means based on subsets of the data.
    
    This imputer calculates means using only subsets of observations that meet
    certain criteria, making it useful for handling non-random missing patterns.
    
    Parameters
    ----------
    subset_conditions : Dict[str, Any]
        Dictionary defining conditions for creating subsets
    min_subset_size : int, default=10
        Minimum number of observations required in subset
    fallback_to_overall : bool, default=True
        Whether to fallback to overall mean if subset is too small
    """
    
    def __init__(self, 
                 subset_conditions: Optional[Dict[str, Any]] = None,
                 min_subset_size: int = 10,
                 fallback_to_overall: bool = True):
        super().__init__()
        self.subset_conditions = subset_conditions or {}
        self.min_subset_size = min_subset_size
        self.fallback_to_overall = fallback_to_overall
    
    def fit(self, X: pd.DataFrame) -> 'PartialMeanImputer':
        """
        Fit the imputer by computing partial means.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data
            
        Returns
        -------
        self : PartialMeanImputer
            The fitted imputer
        """
        self._validate_input(X)
        
        self.partial_means_ = {}
        self.overall_means_ = {}
        
        # Get numerical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numerical_cols:
            if X[col].isnull().all():
                continue
                
            # Calculate overall mean as fallback
            self.overall_means_[col] = X[col].mean()
            
            # Calculate partial means based on conditions
            self.partial_means_[col] = {}
            
            if self.subset_conditions:
                for condition_name, condition in self.subset_conditions.items():
                    # Apply condition to create subset
                    if callable(condition):
                        subset_mask = condition(X)
                    elif isinstance(condition, dict):
                        subset_mask = pd.Series(True, index=X.index)
                        for cond_col, cond_val in condition.items():
                            if cond_col in X.columns:
                                if isinstance(cond_val, (list, tuple)):
                                    subset_mask &= X[cond_col].isin(cond_val)
                                else:
                                    subset_mask &= (X[cond_col] == cond_val)
                    else:
                        continue
                    
                    # Calculate partial mean if subset is large enough
                    subset_data = X.loc[subset_mask, col]
                    if len(subset_data) >= self.min_subset_size and subset_data.notna().any():
                        self.partial_means_[col][condition_name] = subset_data.mean()
            
            # If no conditions specified, use quantile-based partitioning
            if not self.subset_conditions:
                # Create quartile-based subsets
                quartiles = X[col].quantile([0.25, 0.5, 0.75])
                
                # Lower quartile mean
                lower_subset = X[col][X[col] <= quartiles[0.25]]
                if len(lower_subset) >= self.min_subset_size:
                    self.partial_means_[col]['lower_quartile'] = lower_subset.mean()
                
                # Middle mean
                middle_subset = X[col][(X[col] > quartiles[0.25]) & (X[col] <= quartiles[0.75])]
                if len(middle_subset) >= self.min_subset_size:
                    self.partial_means_[col]['middle'] = middle_subset.mean()
                
                # Upper quartile mean
                upper_subset = X[col][X[col] > quartiles[0.75]]
                if len(upper_subset) >= self.min_subset_size:
                    self.partial_means_[col]['upper_quartile'] = upper_subset.mean()
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using partial mean imputation.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values
            
        Returns
        -------
        pd.DataFrame
            The data with missing values imputed using partial means
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        self._validate_input(X)
        X_imputed = X.copy()
        
        for col in self.partial_means_.keys():
            if col not in X_imputed.columns:
                continue
                
            missing_mask = X_imputed[col].isnull()
            if not missing_mask.any():
                continue
            
            # For each missing value, determine which partial mean to use
            for idx in X_imputed.index[missing_mask]:
                fill_value = self.overall_means_.get(col)  # Default fallback
                
                # Try to find appropriate partial mean
                if self.subset_conditions:
                    for condition_name, condition in self.subset_conditions.items():
                        if condition_name in self.partial_means_[col]:
                            # Check if this row meets the condition
                            if callable(condition):
                                if condition(X_imputed.loc[[idx]]).iloc[0]:
                                    fill_value = self.partial_means_[col][condition_name]
                                    break
                            elif isinstance(condition, dict):
                                meets_condition = True
                                for cond_col, cond_val in condition.items():
                                    if cond_col in X_imputed.columns:
                                        row_val = X_imputed.loc[idx, cond_col]
                                        if isinstance(cond_val, (list, tuple)):
                                            meets_condition &= row_val in cond_val
                                        else:
                                            meets_condition &= (row_val == cond_val)
                                
                                if meets_condition:
                                    fill_value = self.partial_means_[col][condition_name]
                                    break
                else:
                    # Use quartile-based assignment based on other features
                    # For simplicity, use middle quartile mean as default
                    if 'middle' in self.partial_means_[col]:
                        fill_value = self.partial_means_[col]['middle']
                
                if fill_value is not None and not pd.isna(fill_value):
                    X_imputed.loc[idx, col] = fill_value
        
        return X_imputed


class SVDImputer(BaseImputer):
    """
    Impute missing values using Singular Value Decomposition (SVD).
    
    This imputer uses matrix factorization techniques to impute missing values
    by finding a low-rank approximation of the data matrix.
    
    Parameters
    ----------
    n_components : int, default=5
        Number of SVD components to use
    max_iter : int, default=100
        Maximum number of iterations for convergence
    tol : float, default=1e-4
        Tolerance for convergence
    """
    
    def __init__(self, n_components: int = 5, max_iter: int = 100, tol: float = 1e-4):
        super().__init__()
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X: pd.DataFrame) -> 'SVDImputer':
        """
        Fit the SVD imputer.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data
            
        Returns
        -------
        self : SVDImputer
            The fitted imputer
        """
        self._validate_input(X)
        
        # Only work with numerical columns
        self.numerical_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(self.numerical_columns_) == 0:
            raise ValueError("No numerical columns found for SVD imputation")
        
        # Store column means for initial imputation
        X_numeric = X[self.numerical_columns_]
        self.column_means_ = X_numeric.mean()
        
        # Adjust number of components if necessary
        self.n_components = min(
            self.n_components, 
            len(self.numerical_columns_) - 1, 
            len(X) - 1
        )
        
        if self.n_components < 1:
            raise ValueError("Insufficient data for SVD imputation")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using SVD-based imputation.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values
            
        Returns
        -------
        pd.DataFrame
            The data with missing values imputed using SVD
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        self._validate_input(X)
        X_imputed = X.copy()
        
        if len(self.numerical_columns_) == 0:
            return X_imputed
        
        X_numeric = X_imputed[self.numerical_columns_].copy()
        
        # Initial imputation with column means
        for col in self.numerical_columns_:
            X_numeric[col] = X_numeric[col].fillna(self.column_means_[col])
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
        
        # Iterative SVD imputation
        missing_mask = X[self.numerical_columns_].isnull().values
        
        if not missing_mask.any():
            return X_imputed  # No missing values to impute
        
        previous_imputed = X_scaled.copy()
        
        for iteration in range(self.max_iter):
            # Perform SVD
            svd = TruncatedSVD(n_components=self.n_components, random_state=42)
            
            try:
                X_transformed = svd.fit_transform(X_scaled)
                X_reconstructed = svd.inverse_transform(X_transformed)
            except:
                # Fallback to previous iteration if SVD fails
                X_reconstructed = previous_imputed
                break
            
            # Update only missing values
            X_scaled[missing_mask] = X_reconstructed[missing_mask]
            
            # Check for convergence
            diff = np.linalg.norm(X_scaled - previous_imputed, 'fro')
            if diff < self.tol:
                break
            
            previous_imputed = X_scaled.copy()
        
        # Inverse transform to original scale
        X_imputed_numeric = scaler.inverse_transform(X_scaled)
        
        # Update the imputed values in the original DataFrame
        X_imputed[self.numerical_columns_] = X_imputed_numeric
        
        return X_imputed


class EnhancedKNNImputer(BaseImputer):
    """
    Enhanced K-Nearest Neighbors imputation with additional features.
    
    This is an enhanced version of KNN imputation that includes:
    - Weighted distance calculations
    - Feature selection for distance computation
    - Categorical variable support through preprocessing
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use
    weights : str, default='distance'
        Weight function ('uniform' or 'distance')
    distance_features : Optional[List[str]], default=None
        Specific features to use for distance calculation
    categorical_strategy : str, default='mode'
        How to handle categorical variables ('mode', 'ignore')
    """
    
    def __init__(self, 
                 n_neighbors: int = 5,
                 weights: str = 'distance',
                 distance_features: Optional[List[str]] = None,
                 categorical_strategy: str = 'mode'):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.distance_features = distance_features
        self.categorical_strategy = categorical_strategy
    
    def fit(self, X: pd.DataFrame) -> 'EnhancedKNNImputer':
        """
        Fit the enhanced KNN imputer.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data
            
        Returns
        -------
        self : EnhancedKNNImputer
            The fitted imputer
        """
        self._validate_input(X)
        
        self.numerical_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Determine features for distance calculation
        if self.distance_features is None:
            self.distance_features_ = self.numerical_columns_
        else:
            self.distance_features_ = [col for col in self.distance_features 
                                     if col in X.columns and col in self.numerical_columns_]
        
        if len(self.distance_features_) == 0:
            raise ValueError("No valid numerical features for distance calculation")
        
        # Store column statistics for initial imputation
        self.column_means_ = X[self.numerical_columns_].mean()
        self.column_modes_ = {}
        
        for col in self.categorical_columns_:
            mode_vals = X[col].mode()
            self.column_modes_[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else None
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using enhanced KNN imputation.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values
            
        Returns
        -------
        pd.DataFrame
            The data with missing values imputed using enhanced KNN
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        self._validate_input(X)
        X_imputed = X.copy()
        
        # Initial imputation for distance calculation
        X_for_distance = X_imputed[self.distance_features_].copy()
        for col in self.distance_features_:
            X_for_distance[col] = X_for_distance[col].fillna(self.column_means_[col])
        
        # Standardize features for distance calculation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_for_distance)
        
        # Fit nearest neighbors model
        nn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, len(X)),  # +1 to exclude self
            metric='euclidean'
        )
        nn_model.fit(X_scaled)
        
        # Impute missing values
        for col in X_imputed.columns:
            missing_mask = X_imputed[col].isnull()
            if not missing_mask.any():
                continue
            
            for idx in X_imputed.index[missing_mask]:
                # Find nearest neighbors
                query_point = X_scaled[X_imputed.index.get_loc(idx)].reshape(1, -1)
                distances, neighbor_indices = nn_model.kneighbors(query_point)
                
                # Exclude self if present
                neighbor_indices = neighbor_indices[0]
                distances = distances[0]
                
                if neighbor_indices[0] == X_imputed.index.get_loc(idx):
                    neighbor_indices = neighbor_indices[1:]
                    distances = distances[1:]
                
                neighbor_indices = neighbor_indices[:self.n_neighbors]
                distances = distances[:self.n_neighbors]
                
                # Get neighbor values
                neighbor_values = X_imputed.iloc[neighbor_indices][col].dropna()
                
                if len(neighbor_values) == 0:
                    # Fallback to column mean/mode
                    if col in self.numerical_columns_:
                        fill_value = self.column_means_.get(col)
                    elif col in self.categorical_columns_:
                        fill_value = self.column_modes_.get(col)
                    else:
                        continue
                else:
                    # Calculate weighted value
                    if col in self.numerical_columns_:
                        if self.weights == 'distance' and len(distances) > 0:
                            # Use inverse distance weighting
                            weights = 1 / (distances[:len(neighbor_values)] + 1e-10)
                            weights = weights / weights.sum()
                            fill_value = np.average(neighbor_values, weights=weights)
                        else:
                            fill_value = neighbor_values.mean()
                    else:
                        # For categorical, use mode of neighbors
                        fill_value = neighbor_values.mode().iloc[0] if len(neighbor_values.mode()) > 0 else self.column_modes_.get(col)
                
                if fill_value is not None and not pd.isna(fill_value):
                    X_imputed.loc[idx, col] = fill_value
        
        return X_imputed