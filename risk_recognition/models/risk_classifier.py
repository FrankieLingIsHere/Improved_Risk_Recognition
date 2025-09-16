"""
BERT-based risk classification model using sentence transformers.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Dict, Optional
import logging
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)


class RiskClassifier:
    """
    BERT-based classifier for risk level and accident cause prediction.
    """
    
    def __init__(self, 
                 base_model: str = 'all-MiniLM-L6-v2',
                 device: str = None):
        """
        Initialize the risk classifier.
        
        Args:
            base_model: Base sentence transformer model name
            device: Device to use for training ('cuda', 'cpu', or None for auto)
        """
        self.base_model = base_model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.sentence_model = None
        self.risk_classifier = None
        self.cause_classifier = None
        self.risk_label_encoder = LabelEncoder()
        self.cause_label_encoder = LabelEncoder()
        
        # Training history
        self.training_history = {'risk': [], 'cause': []}
        
    def _create_sentence_model(self):
        """Create and return the sentence transformer model."""
        if self.sentence_model is None:
            self.sentence_model = SentenceTransformer(self.base_model)
        return self.sentence_model
    
    def _create_classifier_head(self, input_dim: int, num_classes: int) -> nn.Module:
        """
        Create a classification head.
        
        Args:
            input_dim: Input dimension (sentence embedding size)
            num_classes: Number of output classes
            
        Returns:
            Classification head model
        """
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[Dict, Dict]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with 'scenario_text', 'risk_level', 'accident_cause'
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        # Encode labels
        risk_labels = self.risk_label_encoder.fit_transform(df['risk_level'])
        cause_labels = self.cause_label_encoder.fit_transform(df['accident_cause'])
        
        # Split data
        X_train, X_test, y_risk_train, y_risk_test, y_cause_train, y_cause_test = train_test_split(
            df['scenario_text'].tolist(),
            risk_labels,
            cause_labels,
            test_size=test_size,
            random_state=42,
            stratify=risk_labels
        )
        
        train_data = {
            'texts': X_train,
            'risk_labels': y_risk_train,
            'cause_labels': y_cause_train
        }
        
        test_data = {
            'texts': X_test,
            'risk_labels': y_risk_test,
            'cause_labels': y_cause_test
        }
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Risk classes: {len(self.risk_label_encoder.classes_)}")
        logger.info(f"Cause classes: {len(self.cause_label_encoder.classes_)}")
        
        return train_data, test_data
    
    def fine_tune_sentence_model(self, train_texts: List[str], epochs: int = 3):
        """
        Fine-tune the sentence transformer model using contrastive learning.
        
        Args:
            train_texts: List of training texts
            epochs: Number of training epochs
        """
        logger.info("Fine-tuning sentence transformer model...")
        
        # Create the sentence model
        model = self._create_sentence_model()
        
        # Create training examples for contrastive learning
        # We'll use a simple approach where similar texts should have similar embeddings
        train_examples = []
        for i, text in enumerate(train_texts):
            # Create positive pairs (same text with slight variations)
            train_examples.append(InputExample(texts=[text, text], label=1.0))
            
            # Create negative pairs (different texts)
            if i < len(train_texts) - 1:
                train_examples.append(InputExample(texts=[text, train_texts[i+1]], label=0.0))
        
        # Create data loader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        
        # Define loss function
        train_loss = losses.CosineSimilarityLoss(model)
        
        # Fine-tune the model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=int(len(train_dataloader) * 0.1),
            show_progress_bar=True
        )
        
        self.sentence_model = model
        logger.info("Sentence model fine-tuning completed")
    
    def train_classifiers(self, 
                         train_data: Dict, 
                         val_data: Dict = None,
                         epochs: int = 10,
                         learning_rate: float = 0.001,
                         batch_size: int = 32):
        """
        Train the risk and cause classifiers.
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary (optional)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
        """
        logger.info("Training classification heads...")
        
        # Get sentence embeddings
        model = self._create_sentence_model()
        train_embeddings = model.encode(train_data['texts'], show_progress_bar=True)
        
        embedding_dim = train_embeddings.shape[1]
        num_risk_classes = len(self.risk_label_encoder.classes_)
        num_cause_classes = len(self.cause_label_encoder.classes_)
        
        # Create classification heads
        self.risk_classifier = self._create_classifier_head(embedding_dim, num_risk_classes)
        self.cause_classifier = self._create_classifier_head(embedding_dim, num_cause_classes)
        
        # Move to device
        self.risk_classifier.to(self.device)
        self.cause_classifier.to(self.device)
        
        # Create optimizers
        risk_optimizer = torch.optim.Adam(self.risk_classifier.parameters(), lr=learning_rate)
        cause_optimizer = torch.optim.Adam(self.cause_classifier.parameters(), lr=learning_rate)
        
        # Loss functions
        criterion = nn.CrossEntropyLoss()
        
        # Convert to tensors
        X_train = torch.FloatTensor(train_embeddings).to(self.device)
        y_risk_train = torch.LongTensor(train_data['risk_labels']).to(self.device)
        y_cause_train = torch.LongTensor(train_data['cause_labels']).to(self.device)
        
        # Validation data if provided
        if val_data:
            val_embeddings = model.encode(val_data['texts'], show_progress_bar=True)
            X_val = torch.FloatTensor(val_embeddings).to(self.device)
            y_risk_val = torch.LongTensor(val_data['risk_labels']).to(self.device)
            y_cause_val = torch.LongTensor(val_data['cause_labels']).to(self.device)
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.risk_classifier.train()
            self.cause_classifier.train()
            
            # Create batches
            num_samples = len(X_train)
            indices = torch.randperm(num_samples)
            
            total_risk_loss = 0
            total_cause_loss = 0
            num_batches = 0
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X_train[batch_indices]
                batch_y_risk = y_risk_train[batch_indices]
                batch_y_cause = y_cause_train[batch_indices]
                
                # Train risk classifier
                risk_optimizer.zero_grad()
                risk_outputs = self.risk_classifier(batch_X)
                risk_loss = criterion(risk_outputs, batch_y_risk)
                risk_loss.backward()
                risk_optimizer.step()
                
                # Train cause classifier
                cause_optimizer.zero_grad()
                cause_outputs = self.cause_classifier(batch_X)
                cause_loss = criterion(cause_outputs, batch_y_cause)
                cause_loss.backward()
                cause_optimizer.step()
                
                total_risk_loss += risk_loss.item()
                total_cause_loss += cause_loss.item()
                num_batches += 1
            
            avg_risk_loss = total_risk_loss / num_batches
            avg_cause_loss = total_cause_loss / num_batches
            
            # Validation phase
            val_risk_acc = val_cause_acc = 0
            if val_data:
                self.risk_classifier.eval()
                self.cause_classifier.eval()
                
                with torch.no_grad():
                    risk_val_outputs = self.risk_classifier(X_val)
                    cause_val_outputs = self.cause_classifier(X_val)
                    
                    risk_val_pred = torch.argmax(risk_val_outputs, dim=1)
                    cause_val_pred = torch.argmax(cause_val_outputs, dim=1)
                    
                    val_risk_acc = (risk_val_pred == y_risk_val).float().mean().item()
                    val_cause_acc = (cause_val_pred == y_cause_val).float().mean().item()
            
            # Log progress
            if val_data:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Risk Loss: {avg_risk_loss:.4f}, Risk Val Acc: {val_risk_acc:.4f} - "
                          f"Cause Loss: {avg_cause_loss:.4f}, Cause Val Acc: {val_cause_acc:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Risk Loss: {avg_risk_loss:.4f} - "
                          f"Cause Loss: {avg_cause_loss:.4f}")
            
            # Store training history
            self.training_history['risk'].append({
                'epoch': epoch + 1,
                'loss': avg_risk_loss,
                'val_acc': val_risk_acc
            })
            self.training_history['cause'].append({
                'epoch': epoch + 1,
                'loss': avg_cause_loss,
                'val_acc': val_cause_acc
            })
        
        logger.info("Classification training completed")
    
    def predict(self, texts: List[str]) -> Dict[str, List]:
        """
        Predict risk levels and accident causes for given texts.
        
        Args:
            texts: List of scenario texts
            
        Returns:
            Dictionary with 'risk_levels', 'accident_causes', and 'probabilities'
        """
        if self.sentence_model is None or self.risk_classifier is None or self.cause_classifier is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get embeddings
        embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
        X = torch.FloatTensor(embeddings).to(self.device)
        
        # Make predictions
        self.risk_classifier.eval()
        self.cause_classifier.eval()
        
        with torch.no_grad():
            risk_outputs = self.risk_classifier(X)
            cause_outputs = self.cause_classifier(X)
            
            risk_probs = torch.softmax(risk_outputs, dim=1)
            cause_probs = torch.softmax(cause_outputs, dim=1)
            
            risk_predictions = torch.argmax(risk_outputs, dim=1)
            cause_predictions = torch.argmax(cause_outputs, dim=1)
        
        # Decode predictions
        risk_labels = self.risk_label_encoder.inverse_transform(risk_predictions.cpu().numpy())
        cause_labels = self.cause_label_encoder.inverse_transform(cause_predictions.cpu().numpy())
        
        return {
            'risk_levels': risk_labels.tolist(),
            'accident_causes': cause_labels.tolist(),
            'risk_probabilities': risk_probs.cpu().numpy(),
            'cause_probabilities': cause_probs.cpu().numpy()
        }
    
    def evaluate(self, test_data: Dict) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data dictionary
            
        Returns:
            Evaluation metrics
        """
        predictions = self.predict(test_data['texts'])
        
        # Convert predictions back to encoded labels for comparison
        risk_pred_encoded = self.risk_label_encoder.transform(predictions['risk_levels'])
        cause_pred_encoded = self.cause_label_encoder.transform(predictions['accident_causes'])
        
        # Calculate metrics
        risk_report = classification_report(
            test_data['risk_labels'], 
            risk_pred_encoded, 
            target_names=self.risk_label_encoder.classes_,
            output_dict=True
        )
        
        cause_report = classification_report(
            test_data['cause_labels'], 
            cause_pred_encoded, 
            target_names=self.cause_label_encoder.classes_,
            output_dict=True
        )
        
        return {
            'risk_classification': risk_report,
            'cause_classification': cause_report,
            'risk_confusion_matrix': confusion_matrix(test_data['risk_labels'], risk_pred_encoded),
            'cause_confusion_matrix': confusion_matrix(test_data['cause_labels'], cause_pred_encoded)
        }
    
    def save_model(self, save_path: str):
        """
        Save the trained model.
        
        Args:
            save_path: Directory to save the model
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save sentence transformer
        if self.sentence_model:
            self.sentence_model.save(os.path.join(save_path, 'sentence_model'))
        
        # Save classification heads
        if self.risk_classifier:
            torch.save(self.risk_classifier.state_dict(), 
                      os.path.join(save_path, 'risk_classifier.pth'))
        
        if self.cause_classifier:
            torch.save(self.cause_classifier.state_dict(), 
                      os.path.join(save_path, 'cause_classifier.pth'))
        
        # Save label encoders
        import joblib
        joblib.dump(self.risk_label_encoder, os.path.join(save_path, 'risk_label_encoder.pkl'))
        joblib.dump(self.cause_label_encoder, os.path.join(save_path, 'cause_label_encoder.pkl'))
        
        # Save training history
        import json
        with open(os.path.join(save_path, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """
        Load a trained model.
        
        Args:
            load_path: Directory containing the saved model
        """
        # Load sentence transformer
        sentence_model_path = os.path.join(load_path, 'sentence_model')
        if os.path.exists(sentence_model_path):
            self.sentence_model = SentenceTransformer(sentence_model_path)
        
        # Load label encoders
        import joblib
        self.risk_label_encoder = joblib.load(os.path.join(load_path, 'risk_label_encoder.pkl'))
        self.cause_label_encoder = joblib.load(os.path.join(load_path, 'cause_label_encoder.pkl'))
        
        # Load classification heads
        embedding_dim = self.sentence_model.get_sentence_embedding_dimension()
        num_risk_classes = len(self.risk_label_encoder.classes_)
        num_cause_classes = len(self.cause_label_encoder.classes_)
        
        self.risk_classifier = self._create_classifier_head(embedding_dim, num_risk_classes)
        self.cause_classifier = self._create_classifier_head(embedding_dim, num_cause_classes)
        
        self.risk_classifier.load_state_dict(
            torch.load(os.path.join(load_path, 'risk_classifier.pth'), map_location=self.device)
        )
        self.cause_classifier.load_state_dict(
            torch.load(os.path.join(load_path, 'cause_classifier.pth'), map_location=self.device)
        )
        
        self.risk_classifier.to(self.device)
        self.cause_classifier.to(self.device)
        
        # Load training history
        import json
        history_path = os.path.join(load_path, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
        
        logger.info(f"Model loaded from {load_path}")
    
    def train(self, df: pd.DataFrame, 
              test_size: float = 0.2,
              fine_tune_epochs: int = 3,
              classifier_epochs: int = 10,
              learning_rate: float = 0.001,
              batch_size: int = 32):
        """
        Complete training pipeline.
        
        Args:
            df: Training DataFrame
            test_size: Fraction for test split
            fine_tune_epochs: Epochs for sentence model fine-tuning
            classifier_epochs: Epochs for classifier training
            learning_rate: Learning rate
            batch_size: Batch size
        """
        logger.info("Starting complete training pipeline...")
        
        # Prepare data
        train_data, test_data = self.prepare_data(df, test_size)
        
        # Fine-tune sentence model
        self.fine_tune_sentence_model(train_data['texts'], fine_tune_epochs)
        
        # Train classifiers
        self.train_classifiers(
            train_data, 
            test_data, 
            classifier_epochs, 
            learning_rate, 
            batch_size
        )
        
        # Evaluate
        evaluation = self.evaluate(test_data)
        logger.info("Training pipeline completed")
        
        return evaluation