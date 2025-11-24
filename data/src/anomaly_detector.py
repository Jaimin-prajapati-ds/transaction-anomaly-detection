"""
Transaction Anomaly Detection System
=====================================
Author: Jaimin Prajapati
Date: November 2025

Description:
A comprehensive anomaly detection system for identifying fraudulent transactions
and suspicious network behavior using multiple ML techniques including Isolation
Forest, Autoencoders, and statistical methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """
    Main class for detecting anomalies in transaction data.
    
    This class implements multiple anomaly detection algorithms:
    - Isolation Forest: Tree-based anomaly detection
    - Statistical Methods: Z-score and IQR-based detection
    - PCA-based detection: Reconstruction error method
    
    Attributes:
        contamination (float): Expected proportion of anomalies
        scaler (StandardScaler): Feature scaler
        isolation_forest (IsolationForest): IF model
        pca (PCA): Principal Component Analysis model
    """
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination (float): Expected proportion of outliers (default: 0.1)
            random_state (int): Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        # Initialize models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=150,
            max_samples='auto',
            n_jobs=-1
        )
        
        self.pca = PCA(n_components=0.95, random_state=random_state)
        self.is_fitted = False
        
    def fit(self, X):
        """
        Fit the anomaly detection models on training data.
        
        Args:
            X (pd.DataFrame or np.array): Training features
            
        Returns:
            self: Fitted detector instance
        """
        print("[INFO] Fitting anomaly detection models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Isolation Forest
        print("  - Training Isolation Forest...")
        self.isolation_forest.fit(X_scaled)
        
        # Fit PCA for reconstruction-based detection
        print("  - Training PCA model...")
        self.pca.fit(X_scaled)
        
        # Calculate baseline statistics for statistical methods
        self.feature_means = np.mean(X_scaled, axis=0)
        self.feature_stds = np.std(X_scaled, axis=0)
        
        self.is_fitted = True
        print("[SUCCESS] Models trained successfully!\n")
        return self
    
    def predict_isolation_forest(self, X):
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            X (pd.DataFrame or np.array): Input features
            
        Returns:
            np.array: Predictions (-1 for anomaly, 1 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction!")
            
        X_scaled = self.scaler.transform(X)
        predictions = self.isolation_forest.predict(X_scaled)
        return predictions
    
    def predict_statistical(self, X, threshold=3.0):
        """
        Detect anomalies using statistical methods (Z-score).
        
        Args:
            X (pd.DataFrame or np.array): Input features
            threshold (float): Z-score threshold for anomaly detection
            
        Returns:
            np.array: Predictions (-1 for anomaly, 1 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction!")
            
        X_scaled = self.scaler.transform(X)
        z_scores = np.abs((X_scaled - self.feature_means) / (self.feature_stds + 1e-7))
        
        # Mark as anomaly if any feature exceeds threshold
        anomaly_mask = np.any(z_scores > threshold, axis=1)
        predictions = np.where(anomaly_mask, -1, 1)
        return predictions
    
    def predict_pca_reconstruction(self, X, threshold_percentile=95):
        """
        Detect anomalies using PCA reconstruction error.
        
        Args:
            X (pd.DataFrame or np.array): Input features
            threshold_percentile (float): Percentile for error threshold
            
        Returns:
            np.array: Predictions (-1 for anomaly, 1 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction!")
            
        X_scaled = self.scaler.transform(X)
        
        # Calculate reconstruction error
        X_pca = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_pca)
        reconstruction_error = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
        
        # Set threshold based on percentile
        threshold = np.percentile(reconstruction_error, threshold_percentile)
        predictions = np.where(reconstruction_error > threshold, -1, 1)
        return predictions, reconstruction_error
    
    def predict_ensemble(self, X):
        """
        Ensemble prediction combining multiple methods.
        
        Uses voting: if 2+ methods detect anomaly, flag as anomaly.
        
        Args:
            X (pd.DataFrame or np.array): Input features
            
        Returns:
            np.array: Final ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction!")
        
        # Get predictions from all methods
        pred_if = self.predict_isolation_forest(X)
        pred_stat = self.predict_statistical(X)
        pred_pca, _ = self.predict_pca_reconstruction(X)
        
        # Voting mechanism: majority vote
        predictions_matrix = np.column_stack([pred_if, pred_stat, pred_pca])
        anomaly_votes = np.sum(predictions_matrix == -1, axis=1)
        
        # Classify as anomaly if 2 or more methods agree
        final_predictions = np.where(anomaly_votes >= 2, -1, 1)
        return final_predictions
    
    def get_anomaly_scores(self, X):
        """
        Get anomaly scores (higher = more anomalous).
        
        Args:
            X (pd.DataFrame or np.array): Input features
            
        Returns:
            np.array: Anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring!")
            
        X_scaled = self.scaler.transform(X)
        # Negative of decision function (lower score = more anomalous)
        scores = -self.isolation_forest.decision_function(X_scaled)
        return scores