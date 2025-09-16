# Improved Risk Recognition System

A comprehensive system for processing accident scenario data and classifying risk levels and accident causes using BERT sentence transformers.

## Features

- **Dataset Cleaning**: Combines and cleans multiple CSV files with different formats
- **BERT Fine-tuning**: Uses sentence transformers for text encoding and classification
- **Multi-class Classification**: Predicts both risk levels and accident causes
- **Visualization**: Comprehensive charts and analysis of results
- **Extensible**: Easy to add new data sources and modify classification categories

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/FrankieLingIsHere/Improved_Risk_Recognition.git
cd Improved_Risk_Recognition

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with sample data (recommended for first time)
python main.py --use-sample-data --epochs 5

# Run with your own CSV files
python main.py --data-paths file1.csv file2.csv file3.csv --epochs 10

# Load existing model for inference
python main.py --load-model ./models/risk_classifier --use-sample-data
```

## System Architecture

### Data Processing Pipeline

1. **CSV File Loading**: Supports multiple CSV formats
2. **Column Standardization**: Maps various column names to standard format
3. **Text Cleaning**: Normalizes and cleans scenario text
4. **Label Normalization**: Standardizes risk levels and accident causes
5. **Data Combination**: Merges multiple sources with deduplication

### Model Architecture

1. **Sentence Transformer**: BERT-based text encoding (default: all-MiniLM-L6-v2)
2. **Fine-tuning**: Contrastive learning on domain-specific data
3. **Classification Heads**: Separate neural networks for risk and cause prediction
4. **Multi-task Learning**: Joint training for both classification tasks

### Expected Data Format

Your CSV files should contain at least these columns (various names accepted):

- **Text column**: `scenario_text`, `text`, `description`, `incident_text`, `scenario`
- **Risk level**: `risk_level`, `risk`, `severity`, `level` (values: low/medium/high/critical)
- **Accident cause**: `accident_cause`, `cause`, `category`, `type`

## Supported Categories

### Risk Levels
- **Low**: Minor incidents, minimal impact
- **Medium**: Moderate incidents, some impact  
- **High**: Major incidents, significant impact
- **Critical**: Severe incidents, catastrophic impact

### Accident Causes
- **Human Error**: Operator mistakes, training issues
- **Equipment Failure**: Mechanical/technical failures
- **Environmental**: Weather, natural conditions
- **Process Failure**: Procedural breakdowns
- **Communication Failure**: Coordination issues
- **Other**: Miscellaneous or unclassified causes

## Advanced Usage

### Custom Training

```python
from risk_recognition.data.dataset_cleaner import DatasetCleaner
from risk_recognition.models.risk_classifier import RiskClassifier

# Clean your data
cleaner = DatasetCleaner()
combined_df = cleaner.combine_datasets(['file1.csv', 'file2.csv', 'file3.csv'])

# Train model
classifier = RiskClassifier(base_model='all-MiniLM-L6-v2')
evaluation = classifier.train(
    combined_df, 
    test_size=0.2,
    fine_tune_epochs=3,
    classifier_epochs=10
)

# Make predictions
predictions = classifier.predict([
    "Worker fell from scaffold without safety harness",
    "Chemical spill due to corroded pipeline"
])
```

### Inference Only

```python
from risk_recognition.models.risk_classifier import RiskClassifier

# Load trained model
classifier = RiskClassifier()
classifier.load_model('./models/risk_classifier')

# Predict on new data
results = classifier.predict([
    "Equipment malfunction in manufacturing line",
    "Vehicle accident due to poor weather conditions"
])

print("Risk levels:", results['risk_levels'])
print("Accident causes:", results['accident_causes'])
```

## Output Files

The system generates several output files:

- `cleaned_combined_data.csv`: Processed and combined dataset
- `demo_predictions.csv`: Predictions on sample texts
- `visualizations/`: Directory containing charts and plots
  - `data_distribution.png`: Dataset distribution charts
  - `confusion_matrices.png`: Classification performance
  - `training_history.png`: Loss and accuracy curves
  - `classification_metrics.png`: Precision, recall, F1-score
  - `prediction_confidence.png`: Confidence distributions

## Testing

Run the test suite to verify installation:

```bash
python -m pytest tests/ -v
# or
python tests/test_risk_recognition.py
```

## Model Performance

The system typically achieves:
- **Risk Level Classification**: 80-90% accuracy
- **Accident Cause Classification**: 75-85% accuracy

Performance depends on:
- Quality and quantity of training data
- Diversity of scenarios
- Consistency of labeling

## Customization

### Adding New Risk Categories

Modify the `normalize_risk_levels()` method in `dataset_cleaner.py`:

```python
def normalize_risk_levels(self, risk_level: str) -> str:
    # Add your custom mappings
    if risk_level in ['catastrophic', 'extreme']:
        return 'catastrophic'
    # ... existing code
```

### Using Different BERT Models

```python
classifier = RiskClassifier(base_model='bert-base-uncased')
# or
classifier = RiskClassifier(base_model='distilbert-base-uncased')
```

### Custom Classification Heads

Modify the `_create_classifier_head()` method for different architectures.

## Troubleshooting

### Common Issues

1. **Memory Errors**: Use smaller batch sizes or lighter BERT models
2. **CUDA Issues**: Set device explicitly: `RiskClassifier(device='cpu')`
3. **Data Format Errors**: Check CSV column names and ensure text data is present

### Performance Optimization

- Use GPU if available: `RiskClassifier(device='cuda')`
- Reduce model size: Use `all-MiniLM-L6-v2` instead of `bert-base-uncased`
- Limit training epochs for faster iteration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```
@software{risk_recognition_system,
  title={Improved Risk Recognition System},
  author={Frankie Ling},
  year={2024},
  url={https://github.com/FrankieLingIsHere/Improved_Risk_Recognition}
}
```