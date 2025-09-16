#!/usr/bin/env python3
"""
Demo script showing the data cleaning functionality without ML dependencies.
"""

import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from risk_recognition.data.dataset_cleaner import DatasetCleaner, create_sample_datasets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Demonstrate the data cleaning functionality."""
    
    print("="*60)
    print("RISK RECOGNITION SYSTEM - DATA CLEANING DEMO")
    print("="*60)
    
    try:
        # Step 1: Create sample datasets
        print("\nStep 1: Creating sample datasets...")
        sample_dir = "/tmp/sample_risk_data"
        file_paths = create_sample_datasets(sample_dir)
        
        print(f"Created {len(file_paths)} sample CSV files:")
        for path in file_paths:
            print(f"  - {path}")
        
        # Step 2: Initialize cleaner and show raw data
        print("\nStep 2: Examining raw data...")
        cleaner = DatasetCleaner()
        
        import pandas as pd
        for i, path in enumerate(file_paths, 1):
            df = pd.read_csv(path)
            print(f"\nDataset {i} ({path.split('/')[-1]}):")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Rows: {len(df)}")
            print("  Sample data:")
            for _, row in df.head(2).iterrows():
                print(f"    {dict(row)}")
        
        # Step 3: Clean and combine datasets
        print("\nStep 3: Cleaning and combining datasets...")
        combined_df = cleaner.combine_datasets(file_paths)
        
        print(f"Combined dataset:")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Columns: {list(combined_df.columns)}")
        
        # Step 4: Show statistics
        print("\nStep 4: Dataset statistics...")
        stats = cleaner.get_dataset_statistics(combined_df)
        
        print(f"Risk level distribution:")
        for level, count in stats['risk_level_distribution'].items():
            print(f"  {level}: {count}")
        
        print(f"Accident cause distribution:")
        for cause, count in stats['accident_cause_distribution'].items():
            print(f"  {cause}: {count}")
        
        print(f"Average text length: {stats['avg_text_length']:.1f} characters")
        
        # Step 5: Show sample cleaned data
        print("\nStep 5: Sample cleaned data...")
        print("Scenario examples:")
        for i, row in combined_df.head(3).iterrows():
            print(f"\n  Text: {row['scenario_text']}")
            print(f"  Risk Level: {row['risk_level']}")
            print(f"  Accident Cause: {row['accident_cause']}")
        
        # Step 6: Save cleaned data
        output_path = "/tmp/cleaned_risk_data.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to: {output_path}")
        
        print("\n" + "="*60)
        print("DATA CLEANING DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Install PyTorch and sentence-transformers:")
        print("   pip install torch sentence-transformers transformers")
        print("2. Run the full system:")
        print("   python main.py --use-sample-data --epochs 5")
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)