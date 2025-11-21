"""
Evaluation metrics for assessing imputation quality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report


class ImputationEvaluator:
    """
    Evaluate the quality of imputation methods.
    
    This class provides various metrics and visualization tools to assess
    how well different imputation methods perform.
    """
    
    def __init__(self):
        pass
    
    def evaluate_numerical_imputation(self,
                                    original: pd.DataFrame,
                                    imputed: pd.DataFrame,
                                    missing_mask: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate imputation quality for numerical columns.
        
        Parameters
        ----------
        original : pd.DataFrame
            Original data without missing values
        imputed : pd.DataFrame
            Data after imputation
        missing_mask : pd.DataFrame
            Boolean mask indicating where values were missing
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing evaluation metrics
        """
        metrics = {}
        
        # Get numerical columns
        numeric_columns = original.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in imputed.columns and col in missing_mask.columns:
                # Get only the imputed values (where data was originally missing)
                mask = missing_mask[col]
                
                if mask.any():  # If there were missing values in this column
                    true_values = original.loc[mask, col]
                    imputed_values = imputed.loc[mask, col]
                    
                    # Calculate metrics
                    mse = mean_squared_error(true_values, imputed_values)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(true_values, imputed_values)
                    
                    # Handle case where true values have no variance
                    if true_values.std() > 0:
                        r2 = r2_score(true_values, imputed_values)
                    else:
                        r2 = np.nan
                    
                    # Normalized metrics
                    true_range = true_values.max() - true_values.min()
                    if true_range > 0:
                        nrmse = rmse / true_range
                        nmae = mae / true_range
                    else:
                        nrmse = np.nan
                        nmae = np.nan
                    
                    metrics[f'{col}_MSE'] = mse
                    metrics[f'{col}_RMSE'] = rmse
                    metrics[f'{col}_MAE'] = mae
                    metrics[f'{col}_R2'] = r2
                    metrics[f'{col}_NRMSE'] = nrmse
                    metrics[f'{col}_NMAE'] = nmae
        
        # Overall metrics (average across columns)
        if metrics:
            rmse_values = [v for k, v in metrics.items() if 'RMSE' in k and not 'NRMSE' in k]
            mae_values = [v for k, v in metrics.items() if 'MAE' in k and not 'NMAE' in k]
            r2_values = [v for k, v in metrics.items() if 'R2' in k and not np.isnan(v)]
            
            if rmse_values:
                metrics['Overall_RMSE'] = np.mean(rmse_values)
            if mae_values:
                metrics['Overall_MAE'] = np.mean(mae_values)
            if r2_values:
                metrics['Overall_R2'] = np.mean(r2_values)
        
        return metrics
    
    def evaluate_categorical_imputation(self,
                                      original: pd.DataFrame,
                                      imputed: pd.DataFrame, 
                                      missing_mask: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate imputation quality for categorical columns.
        
        Parameters
        ----------
        original : pd.DataFrame
            Original data without missing values
        imputed : pd.DataFrame
            Data after imputation
        missing_mask : pd.DataFrame
            Boolean mask indicating where values were missing
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing evaluation metrics
        """
        metrics = {}
        
        # Get categorical columns
        categorical_columns = original.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if col in imputed.columns and col in missing_mask.columns:
                # Get only the imputed values (where data was originally missing)
                mask = missing_mask[col]
                
                if mask.any():  # If there were missing values in this column
                    true_values = original.loc[mask, col]
                    imputed_values = imputed.loc[mask, col]
                    
                    # Calculate accuracy
                    accuracy = accuracy_score(true_values, imputed_values)
                    metrics[f'{col}_Accuracy'] = accuracy
                    
                    # Calculate category distribution similarity
                    true_dist = true_values.value_counts(normalize=True)
                    imputed_dist = imputed_values.value_counts(normalize=True)
                    
                    # Ensure same categories
                    all_categories = set(true_dist.index) | set(imputed_dist.index)
                    true_dist = true_dist.reindex(all_categories, fill_value=0)
                    imputed_dist = imputed_dist.reindex(all_categories, fill_value=0)
                    
                    # Jensen-Shannon divergence
                    js_div = self._jensen_shannon_divergence(true_dist, imputed_dist)
                    metrics[f'{col}_JS_Divergence'] = js_div
        
        # Overall categorical accuracy
        accuracy_values = [v for k, v in metrics.items() if 'Accuracy' in k]
        if accuracy_values:
            metrics['Overall_Categorical_Accuracy'] = np.mean(accuracy_values)
        
        return metrics
    
    def _jensen_shannon_divergence(self, p: pd.Series, q: pd.Series) -> float:
        """
        Calculate Jensen-Shannon divergence between two probability distributions.
        
        Parameters
        ----------
        p, q : pd.Series
            Probability distributions
            
        Returns
        -------
        float
            Jensen-Shannon divergence
        """
        # Ensure probability distributions sum to 1
        p = p / p.sum()
        q = q / q.sum()
        
        # Calculate M = (P + Q) / 2
        m = (p + q) / 2
        
        # Calculate KL divergences
        kl_pm = self._kl_divergence(p, m)
        kl_qm = self._kl_divergence(q, m)
        
        # Jensen-Shannon divergence
        js = (kl_pm + kl_qm) / 2
        return js
    
    def _kl_divergence(self, p: pd.Series, q: pd.Series) -> float:
        """
        Calculate Kullback-Leibler divergence.
        
        Parameters
        ----------
        p, q : pd.Series
            Probability distributions
            
        Returns
        -------
        float
            KL divergence
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        return np.sum(p * np.log(p / q))
    
    def compare_imputation_methods(self,
                                 original: pd.DataFrame,
                                 imputation_results: Dict[str, pd.DataFrame],
                                 missing_mask: pd.DataFrame) -> pd.DataFrame:
        """
        Compare multiple imputation methods.
        
        Parameters
        ----------
        original : pd.DataFrame
            Original data without missing values
        imputation_results : Dict[str, pd.DataFrame]
            Dictionary mapping method names to imputed DataFrames
        missing_mask : pd.DataFrame
            Boolean mask indicating where values were missing
            
        Returns
        -------
        pd.DataFrame
            Comparison table with metrics for each method
        """
        comparison_results = {}
        
        for method_name, imputed_data in imputation_results.items():
            # Evaluate numerical columns
            num_metrics = self.evaluate_numerical_imputation(
                original, imputed_data, missing_mask
            )
            
            # Evaluate categorical columns  
            cat_metrics = self.evaluate_categorical_imputation(
                original, imputed_data, missing_mask
            )
            
            # Combine metrics
            all_metrics = {**num_metrics, **cat_metrics}
            comparison_results[method_name] = all_metrics
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_results).T
        
        return comparison_df
    
    def plot_imputation_comparison(self,
                                 original: pd.DataFrame,
                                 imputation_results: Dict[str, pd.DataFrame],
                                 missing_mask: pd.DataFrame,
                                 column: str,
                                 figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Create visualizations comparing imputation methods.
        
        Parameters
        ----------
        original : pd.DataFrame
            Original data without missing values
        imputation_results : Dict[str, pd.DataFrame]
            Dictionary mapping method names to imputed DataFrames
        missing_mask : pd.DataFrame
            Boolean mask indicating where values were missing
        column : str
            Column to visualize
        figsize : Tuple[int, int], default=(12, 8)
            Figure size
        """
        if column not in original.columns:
            raise ValueError(f"Column '{column}' not found in original data")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Imputation Comparison for {column}', fontsize=16)
        
        # Get the missing mask for this column
        mask = missing_mask[column]
        
        if not mask.any():
            print(f"No missing values in column '{column}' to visualize")
            return
        
        # Original vs Imputed scatter plots
        ax1 = axes[0, 0]
        for i, (method_name, imputed_data) in enumerate(imputation_results.items()):
            if column in imputed_data.columns:
                true_vals = original.loc[mask, column]
                imputed_vals = imputed_data.loc[mask, column]
                
                if pd.api.types.is_numeric_dtype(original[column]):
                    ax1.scatter(true_vals, imputed_vals, alpha=0.6, 
                              label=method_name, s=20)
        
        if pd.api.types.is_numeric_dtype(original[column]):
            # Perfect prediction line
            min_val = original[column].min()
            max_val = original[column].max()
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            ax1.set_xlabel('True Values')
            ax1.set_ylabel('Imputed Values')
            ax1.set_title('True vs Imputed Values')
            ax1.legend()
        
        # Distribution comparison
        ax2 = axes[0, 1]
        if pd.api.types.is_numeric_dtype(original[column]):
            # Plot distributions
            ax2.hist(original[column], bins=30, alpha=0.5, label='Original', density=True)
            
            for method_name, imputed_data in imputation_results.items():
                if column in imputed_data.columns:
                    ax2.hist(imputed_data[column], bins=30, alpha=0.5, 
                           label=f'{method_name}', density=True)
            
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Density')
            ax2.set_title('Distribution Comparison')
            ax2.legend()
        
        # Error distribution
        ax3 = axes[1, 0]
        if pd.api.types.is_numeric_dtype(original[column]):
            for method_name, imputed_data in imputation_results.items():
                if column in imputed_data.columns:
                    true_vals = original.loc[mask, column]
                    imputed_vals = imputed_data.loc[mask, column]
                    errors = imputed_vals - true_vals
                    
                    ax3.hist(errors, bins=20, alpha=0.6, label=method_name, density=True)
            
            ax3.set_xlabel('Error (Imputed - True)')
            ax3.set_ylabel('Density')
            ax3.set_title('Error Distribution')
            ax3.legend()
        
        # Box plot of errors
        ax4 = axes[1, 1]
        if pd.api.types.is_numeric_dtype(original[column]):
            error_data = []
            method_labels = []
            
            for method_name, imputed_data in imputation_results.items():
                if column in imputed_data.columns:
                    true_vals = original.loc[mask, column]
                    imputed_vals = imputed_data.loc[mask, column]
                    errors = np.abs(imputed_vals - true_vals)
                    
                    error_data.extend(errors.tolist())
                    method_labels.extend([method_name] * len(errors))
            
            if error_data:
                error_df = pd.DataFrame({
                    'Method': method_labels,
                    'Absolute_Error': error_data
                })
                
                sns.boxplot(data=error_df, x='Method', y='Absolute_Error', ax=ax4)
                ax4.set_title('Absolute Error by Method')
                ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_imputation(self,
                          original: pd.DataFrame,
                          imputed: pd.DataFrame,
                          original_with_missing: pd.DataFrame) -> Dict[str, float]:
        """
        Simplified evaluation method for easy use.
        
        Parameters
        ----------
        original : pd.DataFrame
            Original complete data
        imputed : pd.DataFrame
            Data after imputation
        original_with_missing : pd.DataFrame
            Original data with missing values
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing key evaluation metrics
        """
        # Create missing mask
        missing_mask = original_with_missing.isnull()
        
        # Evaluate numerical columns
        num_metrics = self.evaluate_numerical_imputation(original, imputed, missing_mask)
        cat_metrics = self.evaluate_categorical_imputation(original, imputed, missing_mask)
        
        # Combine and simplify metrics
        simplified_metrics = {}
        
        # Extract overall metrics
        if 'Overall_RMSE' in num_metrics:
            simplified_metrics['rmse'] = num_metrics['Overall_RMSE']
        if 'Overall_MAE' in num_metrics:
            simplified_metrics['mae'] = num_metrics['Overall_MAE']
        if 'Overall_R2' in num_metrics:
            simplified_metrics['r2_score'] = num_metrics['Overall_R2']
        if 'Overall_Categorical_Accuracy' in cat_metrics:
            simplified_metrics['categorical_accuracy'] = cat_metrics['Overall_Categorical_Accuracy']
        
        # Calculate mean absolute bias across numerical columns
        numeric_columns = original.select_dtypes(include=[np.number]).columns
        bias_values = []
        
        for col in numeric_columns:
            if col in imputed.columns and missing_mask[col].any():
                mask = missing_mask[col]
                true_vals = original.loc[mask, col]
                imp_vals = imputed.loc[mask, col]
                bias = np.mean(imp_vals - true_vals)
                bias_values.append(abs(bias))
        
        if bias_values:
            simplified_metrics['mean_abs_bias'] = np.mean(bias_values)
        
        # Fill missing metrics with default values
        default_metrics = {'rmse': 0.0, 'mae': 0.0, 'r2_score': 0.0, 'categorical_accuracy': 0.0, 'mean_abs_bias': 0.0}
        for key, default_val in default_metrics.items():
            if key not in simplified_metrics:
                simplified_metrics[key] = default_val
        
        return simplified_metrics

    def generate_evaluation_report(self,
                                 original: pd.DataFrame,
                                 imputation_results: Dict[str, pd.DataFrame],
                                 missing_mask: pd.DataFrame) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Parameters
        ----------
        original : pd.DataFrame
            Original data without missing values
        imputation_results : Dict[str, pd.DataFrame]
            Dictionary mapping method names to imputed DataFrames
        missing_mask : pd.DataFrame
            Boolean mask indicating where values were missing
            
        Returns
        -------
        str
            Formatted evaluation report
        """
        comparison_df = self.compare_imputation_methods(
            original, imputation_results, missing_mask
        )
        
        report = "IMPUTATION EVALUATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Missing data summary
        total_values = missing_mask.size
        missing_values = missing_mask.sum().sum()
        missing_percentage = (missing_values / total_values) * 100
        
        report += f"Missing Data Summary:\n"
        report += f"- Total values: {total_values:,}\n"
        report += f"- Missing values: {missing_values:,}\n"
        report += f"- Missing percentage: {missing_percentage:.2f}%\n\n"
        
        # Column-wise missing data
        report += "Missing Values by Column:\n"
        column_missing = missing_mask.sum()
        for col, count in column_missing.items():
            if count > 0:
                pct = (count / len(missing_mask)) * 100
                report += f"- {col}: {count} ({pct:.2f}%)\n"
        report += "\n"
        
        # Method comparison
        report += "Method Comparison:\n"
        report += "-" * 20 + "\n"
        
        # Overall metrics
        overall_metrics = ['Overall_RMSE', 'Overall_MAE', 'Overall_R2', 'Overall_Categorical_Accuracy']
        
        for metric in overall_metrics:
            if metric in comparison_df.columns:
                report += f"\n{metric}:\n"
                values = comparison_df[metric].dropna().sort_values()
                for method, value in values.items():
                    report += f"  {method}: {value:.4f}\n"
        
        # Best performing methods
        report += "\nBest Performing Methods:\n"
        report += "-" * 25 + "\n"
        
        if 'Overall_RMSE' in comparison_df.columns:
            best_rmse = comparison_df['Overall_RMSE'].idxmin()
            report += f"Lowest RMSE: {best_rmse} ({comparison_df.loc[best_rmse, 'Overall_RMSE']:.4f})\n"
        
        if 'Overall_R2' in comparison_df.columns:
            best_r2 = comparison_df['Overall_R2'].idxmax()
            report += f"Highest R²: {best_r2} ({comparison_df.loc[best_r2, 'Overall_R2']:.4f})\n"
        
        if 'Overall_Categorical_Accuracy' in comparison_df.columns:
            best_acc = comparison_df['Overall_Categorical_Accuracy'].idxmax()
            report += f"Highest Categorical Accuracy: {best_acc} ({comparison_df.loc[best_acc, 'Overall_Categorical_Accuracy']:.4f})\n"
        
        return report