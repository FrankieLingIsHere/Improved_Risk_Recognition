"""
Dataset cleaning and combination utilities for accident risk data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
import logging

logger = logging.getLogger(__name__)


class DatasetCleaner:
    """
    Handles cleaning and combining multiple CSV files containing accident scenario data.
    """
    
    def __init__(self, required_columns: List[str] = None):
        """
        Initialize the dataset cleaner.
        
        Args:
            required_columns: List of required column names for validation
        """
        self.required_columns = required_columns or [
            'scenario_text', 'risk_level', 'accident_cause'
        ]
        
    def load_csv_files(self, file_paths: List[str]) -> List[pd.DataFrame]:
        """
        Load multiple CSV files and return as list of DataFrames.
        
        Args:
            file_paths: List of paths to CSV files
            
        Returns:
            List of loaded DataFrames
        """
        dataframes = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {len(df)} rows from {file_path}")
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                raise
        
        return dataframes
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names across different data sources.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        # Common column name mappings
        column_mapping = {
            'text': 'scenario_text',
            'description': 'scenario_text',
            'incident_text': 'scenario_text',
            'scenario': 'scenario_text',
            'risk': 'risk_level',
            'severity': 'risk_level',
            'level': 'risk_level',
            'cause': 'accident_cause',
            'category': 'accident_cause',
            'type': 'accident_cause',
            'accident_type': 'accident_cause'
        }
        
        # Convert column names to lowercase for mapping
        df_copy = df.copy()
        df_copy.columns = df_copy.columns.str.lower().str.strip()
        
        # Apply mapping
        df_copy = df_copy.rename(columns=column_mapping)
        
        return df_copy
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:\-()]', '', text)
        
        # Remove extra whitespace (after special char removal)
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def normalize_risk_levels(self, risk_level: str) -> str:
        """
        Normalize risk level values to standard categories.
        
        Args:
            risk_level: Input risk level
            
        Returns:
            Normalized risk level
        """
        if pd.isna(risk_level) or not isinstance(risk_level, str):
            return "unknown"
        
        risk_level = risk_level.lower().strip()
        
        # Map various risk level representations to standard categories
        if risk_level in ['low', 'minor', '1', 'minimal']:
            return 'low'
        elif risk_level in ['medium', 'moderate', '2', 'med']:
            return 'medium'
        elif risk_level in ['high', 'major', '3', 'severe']:
            return 'high'
        elif risk_level in ['critical', 'extreme', '4', 'catastrophic']:
            return 'critical'
        else:
            return 'unknown'
    
    def normalize_accident_causes(self, cause: str) -> str:
        """
        Normalize accident cause values to standard categories.
        
        Args:
            cause: Input accident cause
            
        Returns:
            Normalized accident cause
        """
        if pd.isna(cause) or not isinstance(cause, str):
            return "other"
        
        cause = cause.lower().strip()
        
        # Map various cause representations to standard categories
        if any(word in cause for word in ['human', 'operator', 'worker', 'personnel']):
            return 'human_error'
        elif any(word in cause for word in ['equipment', 'machine', 'mechanical', 'technical']):
            return 'equipment_failure'
        elif any(word in cause for word in ['environment', 'weather', 'natural']):
            return 'environmental'
        elif any(word in cause for word in ['process', 'procedure', 'protocol']):
            return 'process_failure'
        elif any(word in cause for word in ['communication', 'coordination']):
            return 'communication_failure'
        else:
            return 'other'
    
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required columns and data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check required columns exist
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for empty data
        if df.empty:
            logger.error("DataFrame is empty")
            return False
        
        # Check for completely null required columns
        for col in self.required_columns:
            if df[col].isna().all():
                logger.error(f"Column {col} is completely null")
                return False
        
        return True
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cleaning operations to a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Standardize columns
        df_clean = self.standardize_columns(df)
        
        # Check if we have the required columns after standardization
        if not self.validate_dataframe(df_clean):
            raise ValueError("DataFrame does not have required columns after standardization")
        
        # Clean text data
        if 'scenario_text' in df_clean.columns:
            df_clean['scenario_text'] = df_clean['scenario_text'].apply(self.clean_text)
        
        # Normalize categorical data
        if 'risk_level' in df_clean.columns:
            df_clean['risk_level'] = df_clean['risk_level'].apply(self.normalize_risk_levels)
        
        if 'accident_cause' in df_clean.columns:
            df_clean['accident_cause'] = df_clean['accident_cause'].apply(self.normalize_accident_causes)
        
        # Remove rows with empty scenario text
        df_clean = df_clean[df_clean['scenario_text'].str.strip() != '']
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates(subset=['scenario_text'])
        
        return df_clean
    
    def combine_datasets(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Load, clean, and combine multiple CSV files.
        
        Args:
            file_paths: List of paths to CSV files
            
        Returns:
            Combined and cleaned DataFrame
        """
        # Load all files
        dataframes = self.load_csv_files(file_paths)
        
        # Clean each dataframe
        cleaned_dfs = []
        for i, df in enumerate(dataframes):
            try:
                cleaned_df = self.clean_dataframe(df)
                logger.info(f"Cleaned dataset {i+1}: {len(cleaned_df)} rows remaining")
                cleaned_dfs.append(cleaned_df)
            except Exception as e:
                logger.warning(f"Skipping dataset {i+1} due to error: {e}")
        
        if not cleaned_dfs:
            raise ValueError("No valid datasets found after cleaning")
        
        # Combine all cleaned dataframes
        combined_df = pd.concat(cleaned_dfs, ignore_index=True)
        
        # Final deduplication across all sources
        combined_df = combined_df.drop_duplicates(subset=['scenario_text'])
        
        logger.info(f"Combined dataset: {len(combined_df)} total rows")
        
        return combined_df
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about the combined dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            'total_rows': len(df),
            'risk_level_distribution': df['risk_level'].value_counts().to_dict(),
            'accident_cause_distribution': df['accident_cause'].value_counts().to_dict(),
            'avg_text_length': df['scenario_text'].str.len().mean(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return stats


def create_sample_datasets(output_dir: str = "/tmp/sample_data"):
    """
    Create sample CSV files for testing the cleaning functionality.
    
    Args:
        output_dir: Directory to save sample files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample data 1 - Construction accidents
    data1 = {
        'incident_text': [
            'Worker fell from scaffold due to missing safety harness',
            'Heavy machinery tipped over on uneven ground causing injury',
            'Chemical spill occurred during material handling',
            'Electrical shock from faulty wiring in temporary installation'
        ],
        'severity': ['high', 'critical', 'medium', 'high'],
        'type': ['human error', 'equipment failure', 'process failure', 'equipment failure']
    }
    
    # Sample data 2 - Manufacturing incidents
    data2 = {
        'description': [
            'Machine operator caught hand in unguarded equipment',
            'Fire started due to overheated electrical panel',
            'Worker slipped on wet floor near production line',
            'Conveyor belt malfunction caused product backup'
        ],
        'risk': ['Major', 'Critical', 'Minor', 'Moderate'],
        'cause': ['Human', 'Technical', 'Environmental', 'Mechanical']
    }
    
    # Sample data 3 - Transportation incidents
    data3 = {
        'scenario': [
            'Vehicle collision at intersection due to poor visibility',
            'Cargo shifted during transport causing vehicle instability',
            'Driver fatigue led to lane departure incident',
            'Mechanical brake failure on steep grade'
        ],
        'level': ['3', '2', '2', '4'],
        'category': ['environmental', 'process failure', 'human error', 'equipment failure']
    }
    
    # Save sample files
    pd.DataFrame(data1).to_csv(f"{output_dir}/construction_incidents.csv", index=False)
    pd.DataFrame(data2).to_csv(f"{output_dir}/manufacturing_incidents.csv", index=False)
    pd.DataFrame(data3).to_csv(f"{output_dir}/transportation_incidents.csv", index=False)
    
    return [
        f"{output_dir}/construction_incidents.csv",
        f"{output_dir}/manufacturing_incidents.csv", 
        f"{output_dir}/transportation_incidents.csv"
    ]