# Transaction Anomaly Detection

A machine learning system for detecting fraudulent transactions using ensemble methods.

## About This Project

I built this project to explore different anomaly detection techniques for identifying suspicious financial transactions. After experimenting with various approaches, I settled on an ensemble method that combines multiple algorithms to reduce false positives while maintaining good detection rates.

## What It Does

The system uses three different detection methods:
- **Isolation Forest**: Tree-based algorithm that isolates anomalies
- **PCA Reconstruction**: Detects anomalies based on reconstruction error
- **Statistical Methods**: Z-score based outlier detection

The final prediction uses a voting mechanism - if 2 or more methods flag a transaction as suspicious, it's marked as an anomaly.

## Quick Start

### Installation

```bash
git clone https://github.com/Jaimin-prajapati-ds/transaction-anomaly-detection.git
cd transaction-anomaly-detection
pip install -r requirements.txt
```

### Basic Usage

```python
from src.data_generator import TransactionDataGenerator
from src.anomaly_detector import AnomalyDetector

# Generate sample data
generator = TransactionDataGenerator(random_state=42)
df = generator.generate_dataset(n_normal=10000, n_anomalous=1000)

# Prepare features
feature_cols = ['amount', 'hour', 'num_transactions_24h', 
                'distance_from_home', 'is_international', 
                'amount_vs_avg_ratio', 'is_weekend', 'is_night']
X = df[feature_cols]

# Train detector
detector = AnomalyDetector(contamination=0.1)
detector.fit(X)

# Make predictions
predictions = detector.predict_ensemble(X)
anomal_scores = detector.get_anomaly_scores(X)
```

## Project Structure

```
transaction-anomaly-detection/
├── data/                      # Placeholder for datasets
├── src/
│   ├── anomaly_detector.py    # Main detection algorithms
│   └── data_generator.py      # Synthetic data generation
├── requirements.txt
├── LICENSE
└── README.md
```

## How It Works

### Data Generation

The `TransactionDataGenerator` creates realistic transaction data with four types of fraud patterns:

1. **High-amount fraud**: Unusually large transaction amounts
2. **Unusual timing**: Transactions at odd hours (late night)
3. **Rapid succession**: Multiple transactions in short timespan
4. **Location anomalies**: Transactions far from typical locations

### Detection Approach

I initially tried using just Isolation Forest, but found it had too many false positives. Adding PCA reconstruction and statistical methods, then combining their predictions through voting, significantly improved the results.

The ensemble approach works because each method catches different types of anomalies:
- Isolation Forest is good for global outliers
- PCA catches anomalies in feature relationships
- Statistical methods identify extreme values

## Performance

On synthetic data with 10% contamination:
- **Precision**: ~89%
- **Recall**: ~85%
- **F1-Score**: ~87%

*Note: Real-world performance will vary based on your data characteristics and contamination rate*

## Requirements

- Python 3.8+
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

See `requirements.txt` for detailed versions.

## Future Improvements

Things I'm planning to add:
- [ ] Autoencoder-based detection for comparison
- [ ] SHAP values for better explainability
- [ ] Visualization dashboard (probably Streamlit)
- [ ] Real transaction dataset examples
- [ ] Model persistence and loading

## Why This Project?

I built this as part of my data science portfolio to demonstrate:
- Handling imbalanced datasets
- Ensemble learning techniques
- Production-ready code structure
- Practical ML for financial applications

## License

MIT License - feel free to use this code for your own projects.

## Author

**Jaimin Prajapati**  
Data Scientist in the making  
📧 jaimin119p@gmail.com  
🔗 [GitHub](https://github.com/Jaimin-prajapati-ds)

---

*Built with Python • scikit-learn • pandas*