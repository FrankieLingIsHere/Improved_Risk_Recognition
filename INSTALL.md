# Installation Guide

## Quick Setup

### 1. Basic Installation (Data Cleaning Only)
```bash
pip install pandas numpy scikit-learn
```

### 2. Full Installation (ML Training & Visualization)
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
# Test data cleaning
python demo_data_cleaning.py

# Test full workflow (requires ML dependencies)
python main.py --use-sample-data --epochs 5

# Run examples
python examples.py
```

## System Requirements

- Python 3.8+
- RAM: 4GB minimum, 8GB recommended
- Storage: 2GB for models and data
- GPU: Optional but recommended for training

## Dependencies

### Core Dependencies
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - ML utilities

### ML Dependencies
- `torch` - PyTorch framework
- `transformers` - Hugging Face transformers
- `sentence-transformers` - BERT sentence encoders

### Visualization Dependencies
- `matplotlib` - Plotting
- `seaborn` - Statistical plots

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'torch'
   ```
   **Solution**: Install ML dependencies
   ```bash
   pip install torch sentence-transformers transformers
   ```

2. **Memory Errors**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Use CPU or reduce batch size
   ```python
   classifier = RiskClassifier(device='cpu')
   ```

3. **Permission Errors**
   ```
   PermissionError: [Errno 13] Permission denied
   ```
   **Solution**: Run with appropriate permissions or change output directory

### Performance Tips

1. **Use GPU for training**
   ```python
   classifier = RiskClassifier(device='cuda')
   ```

2. **Reduce model size for faster inference**
   ```python
   classifier = RiskClassifier(base_model='all-MiniLM-L6-v2')
   ```

3. **Limit training epochs for quick testing**
   ```bash
   python main.py --use-sample-data --epochs 3
   ```

## Development Setup

### For Contributors

1. **Clone repository**
   ```bash
   git clone https://github.com/FrankieLingIsHere/Improved_Risk_Recognition.git
   cd Improved_Risk_Recognition
   ```

2. **Install in development mode**
   ```bash
   pip install -e .
   pip install -e .[dev]
   ```

3. **Run tests**
   ```bash
   python test_data_cleaning.py
   pytest tests/ -v  # If pytest is installed
   ```

4. **Code formatting**
   ```bash
   black risk_recognition/
   flake8 risk_recognition/
   ```