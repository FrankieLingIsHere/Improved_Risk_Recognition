#!/usr/bin/env python3
"""
Main script demonstrating the complete risk recognition workflow.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from risk_recognition.data.dataset_cleaner import DatasetCleaner, create_sample_datasets
from risk_recognition.models.risk_classifier import RiskClassifier
from risk_recognition.utils.visualization import create_visualization_report, print_evaluation_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('risk_recognition.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Risk Recognition System')
    parser.add_argument('--data-paths', nargs='+', help='Paths to CSV files')
    parser.add_argument('--use-sample-data', action='store_true', 
                       help='Use generated sample data for demonstration')
    parser.add_argument('--output-dir', default='./outputs', 
                       help='Output directory for results')
    parser.add_argument('--model-dir', default='./models', 
                       help='Directory to save/load models')
    parser.add_argument('--epochs', type=int, default=5, 
                       help='Number of training epochs')
    parser.add_argument('--test-size', type=float, default=0.2, 
                       help='Test set size (0-1)')
    parser.add_argument('--load-model', 
                       help='Path to load existing model')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    logger.info("Starting Risk Recognition System")
    
    try:
        # Step 1: Data preparation
        logger.info("Step 1: Data Preparation")
        
        if args.use_sample_data or not args.data_paths:
            logger.info("Creating sample datasets for demonstration...")
            data_paths = create_sample_datasets()
        else:
            data_paths = args.data_paths
        
        # Clean and combine datasets
        cleaner = DatasetCleaner()
        combined_df = cleaner.combine_datasets(data_paths)
        
        # Save cleaned dataset
        cleaned_data_path = os.path.join(args.output_dir, 'cleaned_combined_data.csv')
        combined_df.to_csv(cleaned_data_path, index=False)
        logger.info(f"Cleaned dataset saved to {cleaned_data_path}")
        
        # Print dataset statistics
        stats = cleaner.get_dataset_statistics(combined_df)
        logger.info(f"Dataset statistics: {stats}")
        
        # Step 2: Model training or loading
        logger.info("Step 2: Model Training/Loading")
        
        classifier = RiskClassifier()
        
        if args.load_model and os.path.exists(args.load_model):
            logger.info(f"Loading existing model from {args.load_model}")
            classifier.load_model(args.load_model)
            
            # Just evaluate on test split
            train_data, test_data = classifier.prepare_data(combined_df, args.test_size)
            evaluation_results = classifier.evaluate(test_data)
            
        else:
            logger.info("Training new model...")
            evaluation_results = classifier.train(
                combined_df, 
                test_size=args.test_size,
                fine_tune_epochs=2,  # Reduced for demo
                classifier_epochs=args.epochs
            )
            
            # Save the trained model
            model_save_path = os.path.join(args.model_dir, 'risk_classifier')
            classifier.save_model(model_save_path)
            logger.info(f"Model saved to {model_save_path}")
        
        # Step 3: Evaluation and visualization
        logger.info("Step 3: Evaluation and Visualization")
        
        # Print evaluation summary
        print_evaluation_summary(evaluation_results)
        
        # Step 4: Demo predictions
        logger.info("Step 4: Demo Predictions")
        
        demo_texts = [
            "Worker fell from scaffold while not wearing safety harness during construction",
            "Chemical leak occurred due to corroded pipe in manufacturing facility",
            "Vehicle accident happened due to driver fatigue on highway",
            "Equipment malfunction caused by inadequate maintenance procedures",
            "Fire started from overloaded electrical circuit in office building"
        ]
        
        predictions = classifier.predict(demo_texts)
        
        # Create prediction report
        from risk_recognition.utils.visualization import generate_prediction_report
        report_df = generate_prediction_report(
            demo_texts, 
            predictions, 
            os.path.join(args.output_dir, 'demo_predictions.csv')
        )
        
        print("\n" + "="*80)
        print("DEMO PREDICTIONS")
        print("="*80)
        for i, text in enumerate(demo_texts):
            print(f"\nText: {text}")
            print(f"Predicted Risk Level: {predictions['risk_levels'][i]}")
            print(f"Predicted Cause: {predictions['accident_causes'][i]}")
            print(f"Risk Confidence: {max(predictions['risk_probabilities'][i]):.3f}")
            print(f"Cause Confidence: {max(predictions['cause_probabilities'][i]):.3f}")
        
        # Step 5: Create visualization report
        logger.info("Step 5: Creating Visualization Report")
        
        viz_dir = create_visualization_report(
            combined_df,
            evaluation_results,
            classifier.training_history,
            predictions,
            os.path.join(args.output_dir, 'visualizations')
        )
        
        logger.info("Risk Recognition System completed successfully!")
        print(f"\nResults saved to: {args.output_dir}")
        print(f"Visualizations saved to: {viz_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main workflow: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)