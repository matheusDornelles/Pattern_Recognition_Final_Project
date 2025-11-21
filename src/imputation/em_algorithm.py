"""
EM Algorithm-based Imputation Methods

This module implements the Expectation-Maximization algorithm approach for handling missing data.
The EM algorithm iteratively estimates missing values by maximizing the likelihood of the observed data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from scipy import stats
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from .base import BaseImputer


class EMImputer(BaseImputer):
    """
    Expectation-Maximization algorithm for missing data imputation.
    
    The EM algorithm alternates between two steps:
    1. E-step: Estimate the missing values given current parameters
    2. M-step: Update parameters given current complete data estimates
    
    This implementation assumes multivariate normal distribution.
    
    Parameters
    ----------
    max_iter : int, default=100
        Maximum number of EM iterations
    tol : float, default=1e-4
        Convergence tolerance for log-likelihood
    init_method : str, default='mean'
        Initial imputation method ('mean', 'median', 'zero')
    """
    
    def __init__(self, max_iter: int = 100, tol: float = 1e-4, init_method: str = 'mean'):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
        
        if init_method not in ['mean', 'median', 'zero']:
            raise ValueError("init_method must be 'mean', 'median', or 'zero'")
    
    def fit(self, X: pd.DataFrame) -> 'EMImputer':
        """
        Fit the EM imputer.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values
            
        Returns
        -------
        self : EMImputer
            The fitted imputer
        """
        self._validate_input(X)
        
        # Only work with numerical columns
        self.numerical_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(self.numerical_columns_) == 0:
            raise ValueError("No numerical columns found for EM imputation")
        
        X_numeric = X[self.numerical_columns_].copy()
        
        # Store original data and missing pattern
        self.X_original_ = X_numeric.copy()
        self.missing_mask_ = X_numeric.isnull()
        
        # Initial imputation
        X_complete = self._initial_imputation(X_numeric)
        
        # EM algorithm
        self.mu_, self.sigma_, self.log_likelihood_history_ = self._em_algorithm(X_complete)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using EM-based imputation.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values
            
        Returns
        -------
        pd.DataFrame
            The data with missing values imputed using EM algorithm
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        self._validate_input(X)
        X_imputed = X.copy()
        
        if len(self.numerical_columns_) == 0:
            return X_imputed
        
        X_numeric = X_imputed[self.numerical_columns_].copy()
        
        # Initial imputation
        X_complete = self._initial_imputation(X_numeric)
        
        # Apply final E-step to get best estimates
        X_final = self._expectation_step(X_complete, X_numeric.isnull())
        
        # Update the imputed values
        X_imputed[self.numerical_columns_] = X_final
        
        return X_imputed
    
    def _initial_imputation(self, X: pd.DataFrame) -> np.ndarray:
        """Perform initial imputation to start EM algorithm."""
        X_init = X.copy()
        
        for col in X.columns:
            missing_mask = X_init[col].isnull()
            if missing_mask.any():
                if self.init_method == 'mean':
                    fill_value = X_init[col].mean()
                elif self.init_method == 'median':
                    fill_value = X_init[col].median()
                else:  # zero
                    fill_value = 0.0
                
                X_init.loc[missing_mask, col] = fill_value
        
        return X_init.values
    
    def _em_algorithm(self, X_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Execute the EM algorithm.
        
        Returns
        -------
        mu : ndarray
            Final mean vector
        sigma : ndarray
            Final covariance matrix
        log_likelihood_history : list
            History of log-likelihood values
        """
        X_complete = X_init.copy()
        missing_mask = self.missing_mask_.values
        
        log_likelihood_history = []
        
        # Initialize parameters
        mu = np.mean(X_complete, axis=0)
        sigma = np.cov(X_complete, rowvar=False)
        
        # Add small regularization to ensure positive definiteness
        sigma += np.eye(sigma.shape[0]) * 1e-6
        
        for iteration in range(self.max_iter):
            # E-step: Estimate missing values
            X_complete = self._expectation_step(X_complete, missing_mask)
            
            # M-step: Update parameters
            mu_new = np.mean(X_complete, axis=0)
            sigma_new = np.cov(X_complete, rowvar=False)
            sigma_new += np.eye(sigma_new.shape[0]) * 1e-6
            
            # Calculate log-likelihood
            log_likelihood = self._calculate_log_likelihood(X_complete, mu_new, sigma_new)
            log_likelihood_history.append(log_likelihood)
            
            # Check convergence
            if iteration > 0:
                if abs(log_likelihood_history[-1] - log_likelihood_history[-2]) < self.tol:
                    break
            
            mu = mu_new
            sigma = sigma_new
        
        return mu, sigma, log_likelihood_history
    
    def _expectation_step(self, X_complete: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """
        E-step: Estimate missing values given current parameters.
        
        For each observation with missing values, we estimate the missing components
        using the conditional distribution given the observed components.
        """
        X_new = X_complete.copy()
        
        for i in range(X_complete.shape[0]):
            if missing_mask[i].any():
                # Indices of observed and missing variables
                obs_idx = ~missing_mask[i]
                miss_idx = missing_mask[i]
                
                if obs_idx.any():
                    # Conditional mean estimation using multivariate normal properties
                    # mu_miss|obs = mu_miss + Sigma_miss,obs * Sigma_obs^-1 * (x_obs - mu_obs)
                    
                    mu_obs = self.mu_[obs_idx]
                    mu_miss = self.mu_[miss_idx]
                    
                    sigma_obs = self.sigma_[np.ix_(obs_idx, obs_idx)]
                    sigma_miss_obs = self.sigma_[np.ix_(miss_idx, obs_idx)]
                    
                    x_obs = X_complete[i, obs_idx]
                    
                    # Handle singular covariance matrix
                    try:
                        sigma_obs_inv = np.linalg.inv(sigma_obs)
                        conditional_mean = mu_miss + sigma_miss_obs @ sigma_obs_inv @ (x_obs - mu_obs)
                        X_new[i, miss_idx] = conditional_mean
                    except np.linalg.LinAlgError:
                        # Fallback to marginal means if covariance is singular
                        X_new[i, miss_idx] = mu_miss
                else:
                    # If all values are missing, use marginal means
                    X_new[i, miss_idx] = self.mu_[miss_idx]
        
        return X_new
    
    def _calculate_log_likelihood(self, X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
        """Calculate log-likelihood of the data given parameters."""
        try:
            # Multivariate normal log-likelihood
            n, p = X.shape
            
            # Center the data
            X_centered = X - mu
            
            # Calculate log-likelihood
            sign, logdet = np.linalg.slogdet(sigma)
            if sign <= 0:
                return -np.inf
            
            sigma_inv = np.linalg.inv(sigma)
            
            # Vectorized computation of quadratic form
            quad_form = np.sum((X_centered @ sigma_inv) * X_centered, axis=1)
            
            log_likelihood = -0.5 * (n * p * np.log(2 * np.pi) + n * logdet + np.sum(quad_form))
            
            return log_likelihood
            
        except (np.linalg.LinAlgError, ValueError):
            return -np.inf


class EMGaussianMixtureImputer(BaseImputer):
    """
    EM algorithm for imputation using Gaussian Mixture Models.
    
    This extends the basic EM approach by modeling the data as a mixture of 
    Gaussian distributions, which can capture more complex data patterns.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of Gaussian components in the mixture
    max_iter : int, default=100
        Maximum number of EM iterations
    tol : float, default=1e-4
        Convergence tolerance
    init_method : str, default='mean'
        Initial imputation method
    """
    
    def __init__(self, n_components: int = 2, max_iter: int = 100, 
                 tol: float = 1e-4, init_method: str = 'mean'):
        super().__init__()
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
    
    def fit(self, X: pd.DataFrame) -> 'EMGaussianMixtureImputer':
        """
        Fit the Gaussian Mixture EM imputer.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values
            
        Returns
        -------
        self : EMGaussianMixtureImputer
            The fitted imputer
        """
        self._validate_input(X)
        
        # Only work with numerical columns
        self.numerical_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(self.numerical_columns_) == 0:
            raise ValueError("No numerical columns found for GMM EM imputation")
        
        X_numeric = X[self.numerical_columns_].copy()
        
        # Store missing pattern
        self.missing_mask_ = X_numeric.isnull()
        
        # Initial imputation
        X_complete = self._initial_imputation(X_numeric)
        
        # Fit Gaussian Mixture Model
        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=42
        )
        
        # Standardize data for better numerical stability
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_complete)
        
        self.gmm_.fit(X_scaled)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using GMM EM-based imputation.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values
            
        Returns
        -------
        pd.DataFrame
            The data with missing values imputed using GMM EM
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        self._validate_input(X)
        X_imputed = X.copy()
        
        if len(self.numerical_columns_) == 0:
            return X_imputed
        
        X_numeric = X_imputed[self.numerical_columns_].copy()
        
        # Initial imputation
        X_complete = self._initial_imputation(X_numeric)
        X_scaled = self.scaler_.transform(X_complete)
        
        # Iterative imputation using GMM
        missing_mask = X_numeric.isnull().values
        
        for iteration in range(self.max_iter):
            X_scaled_new = self._gmm_imputation_step(X_scaled, missing_mask)
            
            # Check convergence
            if np.allclose(X_scaled, X_scaled_new, atol=self.tol):
                break
            
            X_scaled = X_scaled_new
        
        # Transform back to original scale
        X_final = self.scaler_.inverse_transform(X_scaled)
        X_imputed[self.numerical_columns_] = X_final
        
        return X_imputed
    
    def _initial_imputation(self, X: pd.DataFrame) -> np.ndarray:
        """Perform initial imputation."""
        X_init = X.copy()
        
        for col in X.columns:
            missing_mask = X_init[col].isnull()
            if missing_mask.any():
                if self.init_method == 'mean':
                    fill_value = X_init[col].mean()
                elif self.init_method == 'median':
                    fill_value = X_init[col].median()
                else:  # zero
                    fill_value = 0.0
                
                X_init.loc[missing_mask, col] = fill_value
        
        return X_init.values
    
    def _gmm_imputation_step(self, X_scaled: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """Single step of GMM-based imputation."""
        X_new = X_scaled.copy()
        
        for i in range(X_scaled.shape[0]):
            if missing_mask[i].any():
                # Get component responsibilities
                responsibilities = self.gmm_.predict_proba(X_scaled[i:i+1])[0]
                
                # Estimate missing values as weighted sum of component expectations
                obs_idx = ~missing_mask[i]
                miss_idx = missing_mask[i]
                
                if obs_idx.any():
                    imputed_values = np.zeros(miss_idx.sum())
                    
                    for k in range(self.n_components):
                        # Conditional expectation for component k
                        mu_k = self.gmm_.means_[k]
                        sigma_k = self.gmm_.covariances_[k]
                        
                        # Conditional mean given observed values
                        mu_obs = mu_k[obs_idx]
                        mu_miss = mu_k[miss_idx]
                        sigma_obs = sigma_k[np.ix_(obs_idx, obs_idx)]
                        sigma_miss_obs = sigma_k[np.ix_(miss_idx, obs_idx)]
                        
                        x_obs = X_scaled[i, obs_idx]
                        
                        try:
                            sigma_obs_inv = np.linalg.inv(sigma_obs)
                            conditional_mean = mu_miss + sigma_miss_obs @ sigma_obs_inv @ (x_obs - mu_obs)
                            imputed_values += responsibilities[k] * conditional_mean
                        except np.linalg.LinAlgError:
                            # Fallback to component mean
                            imputed_values += responsibilities[k] * mu_miss
                    
                    X_new[i, miss_idx] = imputed_values
                else:
                    # Use weighted average of component means
                    weighted_mean = np.average(self.gmm_.means_[:, miss_idx], 
                                             weights=responsibilities, axis=0)
                    X_new[i, miss_idx] = weighted_mean
        
        return X_new


class MultipleImputationEM(BaseImputer):
    """
    Multiple Imputation using EM algorithm.
    
    Creates multiple plausible imputations to account for uncertainty in missing values.
    This follows Rubin's multiple imputation framework with EM-based imputation.
    
    Parameters
    ----------
    n_imputations : int, default=5
        Number of multiple imputations to create
    em_max_iter : int, default=50
        Maximum EM iterations for each imputation
    em_tol : float, default=1e-4
        EM convergence tolerance
    """
    
    def __init__(self, n_imputations: int = 5, em_max_iter: int = 50, em_tol: float = 1e-4):
        super().__init__()
        self.n_imputations = n_imputations
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
    
    def fit(self, X: pd.DataFrame) -> 'MultipleImputationEM':
        """
        Fit the Multiple Imputation EM algorithm.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values
            
        Returns
        -------
        self : MultipleImputationEM
            The fitted imputer
        """
        self._validate_input(X)
        
        # Only work with numerical columns
        self.numerical_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(self.numerical_columns_) == 0:
            raise ValueError("No numerical columns found for Multiple Imputation")
        
        # Store the training data for creating multiple imputations
        self.X_train_ = X[self.numerical_columns_].copy()
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Transform the data using Multiple Imputation EM.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values
            
        Returns
        -------
        Dict[int, pd.DataFrame]
            Dictionary containing multiple imputed datasets
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        self._validate_input(X)
        
        # Create multiple imputations
        imputations = {}
        
        for m in range(self.n_imputations):
            # Add random noise to create different starting points
            em_imputer = EMImputer(
                max_iter=self.em_max_iter,
                tol=self.em_tol,
                init_method='mean'
            )
            
            # Fit on training data with some noise for variation
            X_train_noisy = self._add_noise(self.X_train_)
            em_imputer.fit(pd.DataFrame(X_train_noisy, columns=self.numerical_columns_))
            
            # Transform the input data
            X_imputed = em_imputer.transform(X)
            imputations[m + 1] = X_imputed
        
        return imputations
    
    def _add_noise(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add small amount of noise to create variation in imputations."""
        X_noisy = X.copy()
        
        for col in X.columns:
            if X[col].notna().any():
                noise_std = X[col].std() * 0.01  # 1% of standard deviation
                noise = np.random.normal(0, noise_std, len(X))
                X_noisy[col] = X_noisy[col] + noise
        
        return X_noisy
    
    def pool_results(self, imputations: Dict[int, pd.DataFrame], 
                    statistic: str = 'mean') -> pd.DataFrame:
        """
        Pool results from multiple imputations.
        
        Parameters
        ----------
        imputations : Dict[int, pd.DataFrame]
            Dictionary of imputed datasets
        statistic : str, default='mean'
            Pooling statistic ('mean', 'median')
            
        Returns
        -------
        pd.DataFrame
            Pooled imputation result
        """
        if statistic == 'mean':
            # Average across all imputations
            pooled = sum(imputations.values()) / len(imputations)
        elif statistic == 'median':
            # Median across all imputations
            stacked = np.stack([df.values for df in imputations.values()], axis=2)
            pooled_values = np.median(stacked, axis=2)
            pooled = pd.DataFrame(pooled_values, 
                                columns=list(imputations.values())[0].columns,
                                index=list(imputations.values())[0].index)
        else:
            raise ValueError("statistic must be 'mean' or 'median'")
        
        return pooled