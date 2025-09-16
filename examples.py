"""
Example usage script for the Risk Recognition System.
This demonstrates the complete workflow with sample data.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def example_data_cleaning():
    """Example of data cleaning workflow."""
    print("\n" + "="*50)
    print("EXAMPLE: DATA CLEANING WORKFLOW")
    print("="*50)
    
    from risk_recognition.data.dataset_cleaner import DatasetCleaner, create_sample_datasets
    
    # Step 1: Create sample data
    print("\n1. Creating sample datasets...")
    file_paths = create_sample_datasets("/tmp/example_data")
    print(f"   Created {len(file_paths)} CSV files")
    
    # Step 2: Clean and combine
    print("\n2. Cleaning and combining datasets...")
    cleaner = DatasetCleaner()
    combined_df = cleaner.combine_datasets(file_paths)
    print(f"   Combined dataset: {len(combined_df)} rows")
    
    # Step 3: Show results
    print("\n3. Results:")
    stats = cleaner.get_dataset_statistics(combined_df)
    print(f"   Risk levels: {stats['risk_level_distribution']}")
    print(f"   Causes: {stats['accident_cause_distribution']}")
    
    return combined_df


def example_model_training(df):
    """Example of model training workflow (requires ML dependencies)."""
    print("\n" + "="*50)
    print("EXAMPLE: MODEL TRAINING WORKFLOW")
    print("="*50)
    
    try:
        from risk_recognition.models.risk_classifier import RiskClassifier
        
        print("\n1. Initializing BERT-based classifier...")
        classifier = RiskClassifier(base_model='all-MiniLM-L6-v2')
        
        print("\n2. Training the model...")
        print("   Note: This may take several minutes...")
        evaluation = classifier.train(
            df,
            test_size=0.3,
            fine_tune_epochs=2,  # Reduced for demo
            classifier_epochs=5   # Reduced for demo
        )
        
        print("\n3. Model training completed!")
        print(f"   Risk classification accuracy: {evaluation['risk_classification']['accuracy']:.3f}")
        print(f"   Cause classification accuracy: {evaluation['cause_classification']['accuracy']:.3f}")
        
        print("\n4. Making predictions on sample texts...")
        demo_texts = [
            "Worker fell from scaffold without safety equipment",
            "Chemical leak from damaged container",
            "Vehicle accident due to mechanical brake failure"
        ]
        
        predictions = classifier.predict(demo_texts)
        
        for i, text in enumerate(demo_texts):
            print(f"\n   Text: {text}")
            print(f"   Predicted Risk: {predictions['risk_levels'][i]}")
            print(f"   Predicted Cause: {predictions['accident_causes'][i]}")
        
        return classifier
        
    except ImportError as e:
        print(f"\n   ML dependencies not installed: {e}")
        print("\n   To run the full model training, install:")
        print("   pip install torch sentence-transformers transformers")
        return None


def example_visualization(df, evaluation=None):
    """Example of visualization workflow."""
    print("\n" + "="*50)
    print("EXAMPLE: VISUALIZATION WORKFLOW")
    print("="*50)
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        from risk_recognition.utils.visualization import (
            plot_data_distribution, 
            create_visualization_report
        )
        
        print("\n1. Creating data distribution plot...")
        fig = plot_data_distribution(df, "/tmp/data_distribution.png")
        print("   Plot saved to: /tmp/data_distribution.png")
        
        if evaluation:
            print("\n2. Creating comprehensive visualization report...")
            viz_dir = create_visualization_report(
                df, evaluation, {}, None, "/tmp/visualizations"
            )
            print(f"   Visualization report created in: {viz_dir}")
        else:
            print("\n2. Skipping model evaluation plots (no evaluation data)")
        
    except ImportError as e:
        print(f"\n   Visualization dependencies not installed: {e}")
        print("\n   To create visualizations, install:")
        print("   pip install matplotlib seaborn")


def main():
    """Run all examples."""
    print("RISK RECOGNITION SYSTEM - USAGE EXAMPLES")
    print("=" * 60)
    
    # Example 1: Data cleaning (always works)
    df = example_data_cleaning()
    
    # Example 2: Model training (requires ML dependencies)
    classifier = example_model_training(df)
    evaluation = None
    if classifier:
        # Get evaluation results if training succeeded
        train_data, test_data = classifier.prepare_data(df, 0.3)
        evaluation = classifier.evaluate(test_data)
    
    # Example 3: Visualization (requires matplotlib)
    example_visualization(df, evaluation)
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETED")
    print("="*60)
    print("\nTo run the complete system:")
    print("1. Install all dependencies: pip install -r requirements.txt")
    print("2. Run main script: python main.py --use-sample-data --epochs 5")
    print("3. Or run data cleaning demo: python demo_data_cleaning.py")


if __name__ == "__main__":
    main()