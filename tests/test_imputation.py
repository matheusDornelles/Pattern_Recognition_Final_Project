"""
Unit tests for missing data imputation methods
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from imputation.single_value import (
    DefaultValueImputer, 
    MeanImputer, 
    MedianImputer, 
    ModeImputer
)
from utils.data_generator import MissingDataGenerator


class TestSingleValueImputation(unittest.TestCase):
    """Test cases for single value imputation methods"""
    
    def setUp(self):
        """Set up test data"""
        self.generator = MissingDataGenerator(random_state=42)
        
        # Create test data
        self.test_data = pd.DataFrame({
            'numerical1': [1, 2, 3, np.nan, 5, 6, np.nan, 8, 9, 10],
            'numerical2': [10, 20, np.nan, 40, 50, np.nan, 70, 80, 90, 100],
            'categorical1': ['A', 'B', 'A', np.nan, 'C', 'B', 'A', np.nan, 'C', 'A'],
            'categorical2': ['X', np.nan, 'Y', 'X', 'Y', 'Z', np.nan, 'X', 'Y', 'Z']
        })
    
    def test_default_value_imputer(self):
        """Test default value imputation"""
        imputer = DefaultValueImputer(default_value={'numerical1': 0, 'categorical1': 'unknown'})
        result = imputer.fit_transform(self.test_data[['numerical1', 'categorical1']])
        
        # Check that missing values are replaced with defaults
        self.assertEqual(result['numerical1'].isnull().sum(), 0)
        self.assertEqual(result['categorical1'].isnull().sum(), 0)
        self.assertTrue((result['numerical1'] == 0).any())
        self.assertTrue((result['categorical1'] == 'unknown').any())
    
    def test_mean_imputer(self):
        """Test mean imputation"""
        numerical_data = self.test_data[['numerical1', 'numerical2']]
        imputer = MeanImputer()
        result = imputer.fit_transform(numerical_data)
        
        # Check that missing values are filled
        self.assertEqual(result.isnull().sum().sum(), 0)
        
        # Check that means are correct
        original_mean1 = numerical_data['numerical1'].mean()
        original_mean2 = numerical_data['numerical2'].mean()
        
        # The imputed values should equal the original means
        missing_indices1 = numerical_data['numerical1'].isnull()
        missing_indices2 = numerical_data['numerical2'].isnull()
        
        self.assertTrue(all(result.loc[missing_indices1, 'numerical1'] == original_mean1))
        self.assertTrue(all(result.loc[missing_indices2, 'numerical2'] == original_mean2))
    
    def test_median_imputer(self):
        """Test median imputation"""
        numerical_data = self.test_data[['numerical1', 'numerical2']]
        imputer = MedianImputer()
        result = imputer.fit_transform(numerical_data)
        
        # Check that missing values are filled
        self.assertEqual(result.isnull().sum().sum(), 0)
        
        # Check that medians are correct
        original_median1 = numerical_data['numerical1'].median()
        original_median2 = numerical_data['numerical2'].median()
        
        missing_indices1 = numerical_data['numerical1'].isnull()
        missing_indices2 = numerical_data['numerical2'].isnull()
        
        self.assertTrue(all(result.loc[missing_indices1, 'numerical1'] == original_median1))
        self.assertTrue(all(result.loc[missing_indices2, 'numerical2'] == original_median2))
    
    def test_mode_imputer(self):
        """Test mode imputation"""
        categorical_data = self.test_data[['categorical1', 'categorical2']]
        imputer = ModeImputer()
        result = imputer.fit_transform(categorical_data)
        
        # Check that missing values are filled
        self.assertEqual(result.isnull().sum().sum(), 0)
        
        # Check that modes are correct
        original_mode1 = categorical_data['categorical1'].mode().iloc[0]
        original_mode2 = categorical_data['categorical2'].mode().iloc[0]
        
        missing_indices1 = categorical_data['categorical1'].isnull()
        missing_indices2 = categorical_data['categorical2'].isnull()
        
        self.assertTrue(all(result.loc[missing_indices1, 'categorical1'] == original_mode1))
        self.assertTrue(all(result.loc[missing_indices2, 'categorical2'] == original_mode2))
    
    def test_data_generator(self):
        """Test missing data generator"""
        # Test sample dataset generation
        data = self.generator.generate_sample_dataset(n_samples=100, n_features=3)
        self.assertEqual(data.shape, (100, 3))
        self.assertEqual(data.isnull().sum().sum(), 0)  # No missing values initially
        
        # Test MCAR introduction
        data_mcar = self.generator.introduce_mcar(data, missing_rate=0.1)
        missing_count = data_mcar.isnull().sum().sum()
        self.assertTrue(missing_count > 0)  # Should have missing values
        
        # Test MAR introduction
        numeric_col = data.select_dtypes(include=[np.number]).columns[0]
        data_mar = self.generator.introduce_mar(
            data, 
            target_column=numeric_col,
            condition_column=numeric_col,
            condition_threshold=data[numeric_col].median()
        )
        self.assertTrue(data_mar.isnull().sum().sum() > 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow"""
    
    def setUp(self):
        """Set up integration test data"""
        self.generator = MissingDataGenerator(random_state=42)
        
        # Create larger test dataset
        self.original_data = self.generator.generate_sample_dataset(
            n_samples=200,
            n_features=4
        )
        
        # Introduce missing values
        self.data_with_missing = self.generator.introduce_mcar(
            self.original_data.copy(),
            missing_rate=0.15
        )
    
    def test_complete_imputation_workflow(self):
        """Test complete workflow with multiple imputation methods"""
        numerical_cols = self.data_with_missing.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data_with_missing.select_dtypes(include=['object']).columns
        
        # Test mean imputation on numerical columns
        if len(numerical_cols) > 0:
            mean_imputer = MeanImputer()
            mean_result = mean_imputer.fit_transform(self.data_with_missing[numerical_cols])
            self.assertEqual(mean_result.isnull().sum().sum(), 0)
        
        # Test mode imputation on categorical columns  
        if len(categorical_cols) > 0:
            mode_imputer = ModeImputer()
            mode_result = mode_imputer.fit_transform(self.data_with_missing[categorical_cols])
            self.assertEqual(mode_result.isnull().sum().sum(), 0)
    
    def test_imputer_fit_transform_consistency(self):
        """Test that fit_transform gives same result as fit then transform"""
        numerical_data = self.data_with_missing.select_dtypes(include=[np.number])
        
        if len(numerical_data.columns) > 0:
            mean_imputer = MeanImputer()
            
            # Method 1: fit_transform
            result1 = mean_imputer.fit_transform(numerical_data.copy())
            
            # Method 2: fit then transform
            mean_imputer2 = MeanImputer()
            mean_imputer2.fit(numerical_data.copy())
            result2 = mean_imputer2.transform(numerical_data.copy())
            
            # Results should be identical
            pd.testing.assert_frame_equal(result1, result2)


if __name__ == '__main__':
    # Run the tests
    unittest.main()