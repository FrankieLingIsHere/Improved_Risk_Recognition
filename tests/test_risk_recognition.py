"""
Test suite for the risk recognition system.
"""

import unittest
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risk_recognition.data.dataset_cleaner import DatasetCleaner, create_sample_datasets
from risk_recognition.models.risk_classifier import RiskClassifier


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


class TestRiskClassifier(unittest.TestCase):
    """Test the risk classifier functionality."""
    
    def setUp(self):
        self.classifier = RiskClassifier(base_model='all-MiniLM-L6-v2')
        
        # Create a small test dataset
        self.test_data = pd.DataFrame({
            'scenario_text': [
                'Worker fell from height without safety equipment',
                'Chemical spill due to container failure',
                'Machine malfunction caused injury',
                'Slip and fall on wet surface',
                'Electrical shock from faulty wiring'
            ],
            'risk_level': ['high', 'medium', 'high', 'low', 'medium'],
            'accident_cause': ['human_error', 'equipment_failure', 'equipment_failure', 'environmental', 'equipment_failure']
        })
    
    def test_prepare_data(self):
        """Test data preparation."""
        train_data, test_data = self.classifier.prepare_data(self.test_data, test_size=0.2)
        
        # Check that data was split
        self.assertGreater(len(train_data['texts']), 0)
        self.assertGreater(len(test_data['texts']), 0)
        
        # Check that labels were encoded
        self.assertTrue(all(isinstance(label, (int, np.integer)) for label in train_data['risk_labels']))
        self.assertTrue(all(isinstance(label, (int, np.integer)) for label in train_data['cause_labels']))
    
    def test_sentence_model_creation(self):
        """Test sentence model creation."""
        model = self.classifier._create_sentence_model()
        self.assertIsNotNone(model)
        
        # Test encoding
        test_texts = ['Test sentence for encoding']
        embeddings = model.encode(test_texts)
        self.assertEqual(len(embeddings), 1)
        self.assertGreater(len(embeddings[0]), 0)
    
    def test_classifier_head_creation(self):
        """Test classifier head creation."""
        input_dim = 384  # Typical dimension for MiniLM
        num_classes = 4
        
        classifier_head = self.classifier._create_classifier_head(input_dim, num_classes)
        self.assertIsNotNone(classifier_head)
        
        # Test forward pass
        import torch
        test_input = torch.randn(1, input_dim)
        output = classifier_head(test_input)
        self.assertEqual(output.shape, (1, num_classes))


class TestEndToEndWorkflow(unittest.TestCase):
    """Test the complete end-to-end workflow."""
    
    def test_complete_workflow(self):
        """Test the complete workflow with minimal settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Create sample data
                cleaner = DatasetCleaner()
                file_paths = create_sample_datasets(temp_dir)
                
                # Clean and combine data
                combined_df = cleaner.combine_datasets(file_paths)
                self.assertGreater(len(combined_df), 0)
                
                # Initialize classifier with minimal model for testing
                classifier = RiskClassifier(base_model='all-MiniLM-L6-v2')
                
                # Prepare data (using all data for training in this test)
                train_data, test_data = classifier.prepare_data(combined_df, test_size=0.3)
                
                # Test that we can create embeddings
                model = classifier._create_sentence_model()
                embeddings = model.encode(train_data['texts'][:2])  # Just test 2 samples
                self.assertEqual(len(embeddings), 2)
                
                print("End-to-end workflow test completed successfully")
                
            except Exception as e:
                self.fail(f"End-to-end workflow failed: {e}")


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)