"""
Utility functions for visualization and evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


def plot_data_distribution(df: pd.DataFrame, save_path: str = None):
    """
    Plot the distribution of risk levels and accident causes.
    
    Args:
        df: DataFrame with risk data
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Risk level distribution
    risk_counts = df['risk_level'].value_counts()
    axes[0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
    axes[0].set_title('Risk Level Distribution')
    
    # Accident cause distribution
    cause_counts = df['accident_cause'].value_counts()
    axes[1].pie(cause_counts.values, labels=cause_counts.index, autopct='%1.1f%%')
    axes[1].set_title('Accident Cause Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Distribution plot saved to {save_path}")
    
    return fig


def plot_confusion_matrices(evaluation_results: Dict, save_path: str = None):
    """
    Plot confusion matrices for risk and cause classification.
    
    Args:
        evaluation_results: Results from model evaluation
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Risk classification confusion matrix
    risk_cm = evaluation_results['risk_confusion_matrix']
    risk_classes = list(evaluation_results['risk_classification'].keys())[:-3]  # Remove avg metrics
    
    sns.heatmap(risk_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=risk_classes, yticklabels=risk_classes, ax=axes[0])
    axes[0].set_title('Risk Level Classification\nConfusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Cause classification confusion matrix
    cause_cm = evaluation_results['cause_confusion_matrix']
    cause_classes = list(evaluation_results['cause_classification'].keys())[:-3]  # Remove avg metrics
    
    sns.heatmap(cause_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=cause_classes, yticklabels=cause_classes, ax=axes[1])
    axes[1].set_title('Accident Cause Classification\nConfusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrices saved to {save_path}")
    
    return fig


def plot_training_history(training_history: Dict, save_path: str = None):
    """
    Plot training loss and validation accuracy curves.
    
    Args:
        training_history: Training history from the model
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data
    risk_history = training_history['risk']
    cause_history = training_history['cause']
    
    epochs = [h['epoch'] for h in risk_history]
    risk_losses = [h['loss'] for h in risk_history]
    risk_val_accs = [h['val_acc'] for h in risk_history]
    
    cause_losses = [h['loss'] for h in cause_history]
    cause_val_accs = [h['val_acc'] for h in cause_history]
    
    # Risk loss
    axes[0, 0].plot(epochs, risk_losses, 'b-', label='Training Loss')
    axes[0, 0].set_title('Risk Classification - Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Risk validation accuracy
    axes[0, 1].plot(epochs, risk_val_accs, 'b-', label='Validation Accuracy')
    axes[0, 1].set_title('Risk Classification - Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True)
    
    # Cause loss
    axes[1, 0].plot(epochs, cause_losses, 'g-', label='Training Loss')
    axes[1, 0].set_title('Cause Classification - Training Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    # Cause validation accuracy
    axes[1, 1].plot(epochs, cause_val_accs, 'g-', label='Validation Accuracy')
    axes[1, 1].set_title('Cause Classification - Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    return fig


def plot_classification_metrics(evaluation_results: Dict, save_path: str = None):
    """
    Plot precision, recall, and F1-score for each class.
    
    Args:
        evaluation_results: Results from model evaluation
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Risk classification metrics
    risk_data = evaluation_results['risk_classification']
    risk_classes = [k for k in risk_data.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    risk_precision = [risk_data[cls]['precision'] for cls in risk_classes]
    risk_recall = [risk_data[cls]['recall'] for cls in risk_classes]
    risk_f1 = [risk_data[cls]['f1-score'] for cls in risk_classes]
    
    # Risk metrics plots
    axes[0, 0].bar(risk_classes, risk_precision, color='skyblue')
    axes[0, 0].set_title('Risk Classification - Precision')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[0, 1].bar(risk_classes, risk_recall, color='lightcoral')
    axes[0, 1].set_title('Risk Classification - Recall')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    axes[0, 2].bar(risk_classes, risk_f1, color='lightgreen')
    axes[0, 2].set_title('Risk Classification - F1-Score')
    axes[0, 2].set_ylabel('F1-Score')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Cause classification metrics
    cause_data = evaluation_results['cause_classification']
    cause_classes = [k for k in cause_data.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    cause_precision = [cause_data[cls]['precision'] for cls in cause_classes]
    cause_recall = [cause_data[cls]['recall'] for cls in cause_classes]
    cause_f1 = [cause_data[cls]['f1-score'] for cls in cause_classes]
    
    # Cause metrics plots
    axes[1, 0].bar(cause_classes, cause_precision, color='skyblue')
    axes[1, 0].set_title('Cause Classification - Precision')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    axes[1, 1].bar(cause_classes, cause_recall, color='lightcoral')
    axes[1, 1].set_title('Cause Classification - Recall')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    axes[1, 2].bar(cause_classes, cause_f1, color='lightgreen')
    axes[1, 2].set_title('Cause Classification - F1-Score')
    axes[1, 2].set_ylabel('F1-Score')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Classification metrics plot saved to {save_path}")
    
    return fig


def print_evaluation_summary(evaluation_results: Dict):
    """
    Print a summary of evaluation results.
    
    Args:
        evaluation_results: Results from model evaluation
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    # Risk classification summary
    risk_data = evaluation_results['risk_classification']
    print(f"\nRisk Level Classification:")
    print(f"  Overall Accuracy: {risk_data['accuracy']:.3f}")
    print(f"  Macro Avg F1:    {risk_data['macro avg']['f1-score']:.3f}")
    print(f"  Weighted Avg F1: {risk_data['weighted avg']['f1-score']:.3f}")
    
    # Cause classification summary
    cause_data = evaluation_results['cause_classification']
    print(f"\nAccident Cause Classification:")
    print(f"  Overall Accuracy: {cause_data['accuracy']:.3f}")
    print(f"  Macro Avg F1:    {cause_data['macro avg']['f1-score']:.3f}")
    print(f"  Weighted Avg F1: {cause_data['weighted avg']['f1-score']:.3f}")
    
    print("\n" + "="*60)


def generate_prediction_report(texts: List[str], predictions: Dict, save_path: str = None) -> pd.DataFrame:
    """
    Generate a detailed prediction report.
    
    Args:
        texts: Input texts
        predictions: Model predictions
        save_path: Optional path to save the report as CSV
        
    Returns:
        DataFrame with prediction results
    """
    # Create detailed report
    report_data = []
    
    for i, text in enumerate(texts):
        risk_probs = predictions['risk_probabilities'][i]
        cause_probs = predictions['cause_probabilities'][i]
        
        report_data.append({
            'text': text,
            'predicted_risk': predictions['risk_levels'][i],
            'predicted_cause': predictions['accident_causes'][i],
            'risk_confidence': np.max(risk_probs),
            'cause_confidence': np.max(cause_probs),
            'text_length': len(text),
            'word_count': len(text.split())
        })
    
    report_df = pd.DataFrame(report_data)
    
    if save_path:
        report_df.to_csv(save_path, index=False)
        logger.info(f"Prediction report saved to {save_path}")
    
    return report_df


def plot_prediction_confidence(predictions: Dict, save_path: str = None):
    """
    Plot confidence distributions for predictions.
    
    Args:
        predictions: Model predictions with probabilities
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Risk prediction confidence
    risk_confidences = [np.max(probs) for probs in predictions['risk_probabilities']]
    axes[0].hist(risk_confidences, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('Risk Prediction Confidence Distribution')
    axes[0].set_xlabel('Confidence Score')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(np.mean(risk_confidences), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(risk_confidences):.3f}')
    axes[0].legend()
    
    # Cause prediction confidence
    cause_confidences = [np.max(probs) for probs in predictions['cause_probabilities']]
    axes[1].hist(cause_confidences, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_title('Cause Prediction Confidence Distribution')
    axes[1].set_xlabel('Confidence Score')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(np.mean(cause_confidences), color='red', linestyle='--',
                    label=f'Mean: {np.mean(cause_confidences):.3f}')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confidence distribution plot saved to {save_path}")
    
    return fig


def create_visualization_report(df: pd.DataFrame, 
                              evaluation_results: Dict,
                              training_history: Dict,
                              predictions: Dict = None,
                              output_dir: str = "/tmp/visualizations"):
    """
    Create a comprehensive visualization report.
    
    Args:
        df: Original dataset
        evaluation_results: Model evaluation results
        training_history: Training history
        predictions: Optional prediction results
        output_dir: Directory to save visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Data distribution
    plot_data_distribution(df, f"{output_dir}/data_distribution.png")
    
    # Confusion matrices
    plot_confusion_matrices(evaluation_results, f"{output_dir}/confusion_matrices.png")
    
    # Training history
    plot_training_history(training_history, f"{output_dir}/training_history.png")
    
    # Classification metrics
    plot_classification_metrics(evaluation_results, f"{output_dir}/classification_metrics.png")
    
    # Prediction confidence if provided
    if predictions:
        plot_prediction_confidence(predictions, f"{output_dir}/prediction_confidence.png")
    
    logger.info(f"Visualization report created in {output_dir}")
    
    return output_dir