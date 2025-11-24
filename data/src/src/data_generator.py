"""
Synthetic Transaction Data Generator
====================================
Author: Jaimin Prajapati
Date: November 2025

Generates realistic transaction data with anomalies for testing.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


class TransactionDataGenerator:
    """
    Generate synthetic transaction data with realistic patterns and anomalies.
    
    Creates both normal transactions following typical patterns and
    anomalous transactions representing various fraud scenarios.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the data generator.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        np.random.seed(random_state)
        random.seed(random_state)
        self.random_state = random_state
        
    def generate_normal_transactions(self, n_samples=10000):
        """
        Generate normal transaction data following typical patterns.
        
        Args:
            n_samples (int): Number of normal transactions to generate
            
        Returns:
            pd.DataFrame: Normal transaction data
        """
        print(f"[INFO] Generating {n_samples} normal transactions...")
        
        data = {
            'transaction_id': [f'TXN_{i:08d}' for i in range(n_samples)],
            'amount': np.random.lognormal(mean=4.5, sigma=1.2, size=n_samples),
            'hour': np.random.choice(range(24), size=n_samples, p=self._get_hour_distribution()),
            'day_of_week': np.random.randint(0, 7, size=n_samples),
            'merchant_category': np.random.choice(
                ['retail', 'food', 'gas', 'online', 'utilities'],
                size=n_samples,
                p=[0.3, 0.25, 0.15, 0.2, 0.1]
            ),
            'num_transactions_24h': np.random.poisson(lam=3, size=n_samples),
            'avg_amount_30d': np.random.normal(loc=150, scale=50, size=n_samples),
            'distance_from_home': np.random.exponential(scale=10, size=n_samples),
            'is_international': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        }
        
        df = pd.DataFrame(data)
        
        # Add some correlated features
        df['amount_vs_avg_ratio'] = df['amount'] / (df['avg_amount_30d'] + 1)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        
        # Add label
        df['is_anomaly'] = 0
        
        return df
    
    def generate_anomalous_transactions(self, n_samples=1000):
        """
        Generate anomalous transaction data representing fraud patterns.
        
        Args:
            n_samples (int): Number of anomalous transactions to generate
            
        Returns:
            pd.DataFrame: Anomalous transaction data
        """
        print(f"[INFO] Generating {n_samples} anomalous transactions...")
        
        # Different types of anomalies
        anomaly_types = [
            self._generate_high_amount_fraud,
            self._generate_unusual_time_fraud,
            self._generate_rapid_succession_fraud,
            self._generate_unusual_location_fraud,
        ]
        
        samples_per_type = n_samples // len(anomaly_types)
        dfs = []
        
        for i, anomaly_func in enumerate(anomaly_types):
            df_anomaly = anomaly_func(samples_per_type, start_id=i*samples_per_type)
            dfs.append(df_anomaly)
        
        df_all_anomalies = pd.concat(dfs, ignore_index=True)
        return df_all_anomalies
    
    def _generate_high_amount_fraud(self, n, start_id=0):
        """Generate transactions with unusually high amounts."""
        data = {
            'transaction_id': [f'TXN_A{start_id+i:07d}' for i in range(n)],
            'amount': np.random.uniform(1000, 5000, size=n),  # Very high amounts
            'hour': np.random.randint(0, 24, size=n),
            'day_of_week': np.random.randint(0, 7, size=n),
            'merchant_category': np.random.choice(['retail', 'online', 'gas'], size=n),
            'num_transactions_24h': np.random.randint(1, 5, size=n),
            'avg_amount_30d': np.random.normal(loc=150, scale=50, size=n),
            'distance_from_home': np.random.exponential(scale=10, size=n),
            'is_international': np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        }
        df = pd.DataFrame(data)
        df['amount_vs_avg_ratio'] = df['amount'] / (df['avg_amount_30d'] + 1)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['is_anomaly'] = 1
        return df
    
    def _generate_unusual_time_fraud(self, n, start_id=0):
        """Generate transactions at unusual times (late night)."""
        data = {
            'transaction_id': [f'TXN_A{start_id+i:07d}' for i in range(n)],
            'amount': np.random.lognormal(mean=4.5, sigma=1.2, size=n),
            'hour': np.random.choice([0, 1, 2, 3, 4], size=n),  # Late night
            'day_of_week': np.random.randint(0, 7, size=n),
            'merchant_category': np.random.choice(['retail', 'online'], size=n),
            'num_transactions_24h': np.random.poisson(lam=8, size=n),  # High frequency
            'avg_amount_30d': np.random.normal(loc=150, scale=50, size=n),
            'distance_from_home': np.random.uniform(50, 200, size=n),  # Far from home
            'is_international': np.random.choice([0, 1], size=n, p=[0.6, 0.4]),
        }
        df = pd.DataFrame(data)
        df['amount_vs_avg_ratio'] = df['amount'] / (df['avg_amount_30d'] + 1)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = 1  # All night transactions
        df['is_anomaly'] = 1
        return df
    
    def _generate_rapid_succession_fraud(self, n, start_id=0):
        """Generate rapid succession of transactions."""
        data = {
            'transaction_id': [f'TXN_A{start_id+i:07d}' for i in range(n)],
            'amount': np.random.lognormal(mean=4.5, sigma=1.2, size=n),
            'hour': np.random.randint(0, 24, size=n),
            'day_of_week': np.random.randint(0, 7, size=n),
            'merchant_category': np.random.choice(['retail', 'online'], size=n),
            'num_transactions_24h': np.random.randint(15, 30, size=n),  # Very high frequency
            'avg_amount_30d': np.random.normal(loc=150, scale=50, size=n),
            'distance_from_home': np.random.exponential(scale=10, size=n),
            'is_international': np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
        }
        df = pd.DataFrame(data)
        df['amount_vs_avg_ratio'] = df['amount'] / (df['avg_amount_30d'] + 1)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['is_anomaly'] = 1
        return df
    
    def _generate_unusual_location_fraud(self, n, start_id=0):
        """Generate transactions from unusual locations."""
        data = {
            'transaction_id': [f'TXN_A{start_id+i:07d}' for i in range(n)],
            'amount': np.random.lognormal(mean=4.5, sigma=1.2, size=n),
            'hour': np.random.randint(0, 24, size=n),
            'day_of_week': np.random.randint(0, 7, size=n),
            'merchant_category': np.random.choice(['retail', 'online', 'food'], size=n),
            'num_transactions_24h': np.random.poisson(lam=3, size=n),
            'avg_amount_30d': np.random.normal(loc=150, scale=50, size=n),
            'distance_from_home': np.random.uniform(200, 1000, size=n),  # Very far
            'is_international': 1,  # All international
        }
        df = pd.DataFrame(data)
        df['amount_vs_avg_ratio'] = df['amount'] / (df['avg_amount_30d'] + 1)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['is_anomaly'] = 1
        return df
    
    def _get_hour_distribution(self):
        """Get realistic hourly distribution for transactions."""
        # Lower probability at night, higher during day
        probs = [0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5 AM
                 0.03, 0.05, 0.07, 0.08, 0.08, 0.07,  # 6-11 AM
                 0.08, 0.07, 0.07, 0.06, 0.06, 0.07,  # 12-5 PM
                 0.08, 0.07, 0.05, 0.03, 0.02, 0.01]  # 6-11 PM
        return np.array(probs) / sum(probs)
    
    def generate_dataset(self, n_normal=10000, n_anomalous=1000, shuffle=True):
        """
        Generate complete dataset with normal and anomalous transactions.
        
        Args:
            n_normal (int): Number of normal transactions
            n_anomalous (int): Number of anomalous transactions
            shuffle (bool): Whether to shuffle the dataset
            
        Returns:
            pd.DataFrame: Complete dataset
        """
        print("\n" + "="*60)
        print("   SYNTHETIC TRANSACTION DATA GENERATION")
        print("="*60 + "\n")
        
        # Generate data
        df_normal = self.generate_normal_transactions(n_normal)
        df_anomalous = self.generate_anomalous_transactions(n_anomalous)
        
        # Combine
        df = pd.concat([df_normal, df_anomalous], ignore_index=True)
        
        if shuffle:
            df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        print(f"\n[SUCCESS] Generated {len(df)} total transactions")
        print(f"  - Normal: {len(df_normal)} ({len(df_normal)/len(df)*100:.1f}%)")
        print(f"  - Anomalous: {len(df_anomalous)} ({len(df_anomalous)/len(df)*100:.1f}%)")
        print(f"  - Features: {df.shape[1]}")
        
        return df