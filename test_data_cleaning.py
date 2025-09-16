"""
Lightweight test suite for data cleaning functionality.
"""

import unittest
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risk_recognition.data.dataset_cleaner import DatasetCleaner, create_sample_datasets


class TestDatasetCleaner(unittest.TestCase):
    """Test the dataset cleaning functionality."""
    
    def setUp(self):
        self.cleaner = DatasetCleaner()
        
    def test_standardize_columns(self):
        """Test column standardization."""
        # Create test DataFrame with various column names
        test_data = pd.DataFrame({
            'text': ['Sample accident text'],
            'risk': ['high'],
            'cause': ['human error']
        })
        
        standardized = self.cleaner.standardize_columns(test_data)
        
        # Check that columns are renamed correctly
        self.assertIn('scenario_text', standardized.columns)
        self.assertIn('risk_level', standardized.columns)
        self.assertIn('accident_cause', standardized.columns)
        
    def test_normalize_risk_levels(self):
        """Test risk level normalization."""
        # Test various risk level inputs
        test_cases = [
            ('low', 'low'),
            ('high', 'high'),
            ('critical', 'critical'),
            ('major', 'high'),
            ('1', 'low'),
            ('unknown_value', 'unknown')
        ]
        
        for input_val, expected in test_cases:
            result = self.cleaner.normalize_risk_levels(input_val)
            self.assertEqual(result, expected)
    
    def test_normalize_accident_causes(self):
        """Test accident cause normalization."""
        test_cases = [
            ('human error', 'human_error'),
            ('equipment failure', 'equipment_failure'),
            ('environmental', 'environmental'),
            ('unknown_cause', 'other')
        ]
        
        for input_val, expected in test_cases:
            result = self.cleaner.normalize_accident_causes(input_val)
            self.assertEqual(result, expected)
    
    def test_create_sample_datasets(self):
        """Test sample dataset creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = create_sample_datasets(temp_dir)
            
            # Check that files were created
            self.assertEqual(len(file_paths), 3)
            for path in file_paths:
                self.assertTrue(os.path.exists(path))
                
                # Check that files contain data
                df = pd.read_csv(path)
                self.assertGreater(len(df), 0)
    
    def test_combine_datasets(self):
        """Test dataset combination."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample datasets
            file_paths = create_sample_datasets(temp_dir)
            
            # Combine datasets
            combined_df = self.cleaner.combine_datasets(file_paths)
            
            # Check result
            self.assertGreater(len(combined_df), 0)
            self.assertIn('scenario_text', combined_df.columns)
            self.assertIn('risk_level', combined_df.columns)
            self.assertIn('accident_cause', combined_df.columns)
            
            # Check that all text entries are non-empty
            self.assertTrue(all(len(text.strip()) > 0 for text in combined_df['scenario_text']))
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        test_cases = [
            ('  Text with   extra   spaces  ', 'Text with extra spaces'),
            ('Text with\n\tnewlines\r', 'Text with newlines'),
            ('Text with @#$ special chars!', 'Text with special chars!'),
            ('', ''),
        ]
        
        for input_text, expected in test_cases:
            result = self.cleaner.clean_text(input_text)
            self.assertEqual(result, expected)
    
    def test_dataset_statistics(self):
        """Test dataset statistics generation."""
        test_df = pd.DataFrame({
            'scenario_text': ['Text 1', 'Text 2', 'Text 3'],
            'risk_level': ['high', 'low', 'high'],
            'accident_cause': ['human_error', 'equipment_failure', 'human_error']
        })
        
        stats = self.cleaner.get_dataset_statistics(test_df)
        
        # Check statistics structure
        self.assertIn('total_rows', stats)
        self.assertIn('risk_level_distribution', stats)
        self.assertIn('accident_cause_distribution', stats)
        self.assertIn('avg_text_length', stats)
        
        # Check values
        self.assertEqual(stats['total_rows'], 3)
        self.assertEqual(stats['risk_level_distribution']['high'], 2)
        self.assertEqual(stats['accident_cause_distribution']['human_error'], 2)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Running lightweight data cleaning tests...")
    unittest.main(verbosity=2)