"""Example usage of the Transaction Anomaly Detection system.

This script demonstrates how to:
1. Generate synthetic transaction data
2. Train the anomaly detector
3. Evaluate performance
4. Show example predictions

Author: Jaimin Prajapati
Date: November 2025
"""

import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path (adjust if needed)
try:
    from src.data_generator import TransactionDataGenerator
    from src.anomaly_detector import AnomalyDetector
except ImportError:
    print("Error: Cannot import modules. Make sure you're running from project root.")
    sys.exit(1)

from sklearn.metrics import classification_report, confusion_matrix


def main():
    print("\n" + "="*70)
    print("  TRANSACTION ANOMALY DETECTION - DEMONSTRATION")
    print("="*70 + "\n")
    
    # Step 1: Generate data
    print("[1/5] Generating synthetic transaction data...")
    generator = TransactionDataGenerator(random_state=42)
    df = generator.generate_dataset(n_normal=10000, n_anomalous=1000)
    print(f"     ✓ Generated {len(df):,} transactions\n")
    
    # Step 2: Prepare features
    print("[2/5] Preparing features...")
    feature_cols = [
        'amount', 'hour', 'day_of_week', 'num_transactions_24h',
        'avg_amount_30d', 'distance_from_home', 'is_international',
        'amount_vs_avg_ratio', 'is_weekend', 'is_night'
    ]
    X = df[feature_cols]
    y_true = df['is_anomaly'].map({0: 1, 1: -1})  # Convert to sklearn format
    print(f"     ✓ Using {len(feature_cols)} features\n")
    
    # Step 3: Train detector
    print("[3/5] Training anomaly detector...")
    detector = AnomalyDetector(contamination=0.1, random_state=42)
    detector.fit(X)
    print("     ✓ Training complete\n")
    
    # Step 4: Make predictions
    print("[4/5] Detecting anomalies...")
    predictions = detector.predict_ensemble(X)
    n_anomalies = np.sum(predictions == -1)
    print(f"     ✓ Found {n_anomalies:,} anomalies\n")
    
    # Step 5: Evaluate
    print("[5/5] Evaluating performance...")
    print("\n" + "-"*70)
    print("Classification Report:")
    print("-"*70)
    print(classification_report(y_true, predictions, 
                                target_names=['Normal', 'Anomaly'],
                                digits=3))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, predictions)
    print("\nConfusion Matrix:")
    print("-"*70)
    print(f"                Predicted Normal    Predicted Anomaly")
    print(f"Actual Normal        {cm[0][0]:6d}              {cm[0][1]:6d}")
    print(f"Actual Anomaly       {cm[1][0]:6d}              {cm[1][1]:6d}")
    
    # Show sample predictions
    print("\n" + "="*70)
    print("  SAMPLE DETECTED ANOMALIES")
    print("="*70)
    
    anomaly_mask = predictions == -1
    anomaly_df = df[anomaly_mask].head(5)
    
    for idx, row in anomaly_df.iterrows():
        print(f"\nTransaction: {row['transaction_id']}")
        print(f"  Amount: ${row['amount']:.2f} (avg: ${row['avg_amount_30d']:.2f})")
        print(f"  Time: {row['hour']:02d}:00, {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][row['day_of_week']]}")
        print(f"  Distance from home: {row['distance_from_home']:.1f} km")
        print(f"  International: {'Yes' if row['is_international'] else 'No'}")
        print(f"  True Label: {'🚨 FRAUD' if row['is_anomaly'] == 1 else '✓ Normal'}")
    
    print("\n" + "="*70)
    print("  ✓ DEMO COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()