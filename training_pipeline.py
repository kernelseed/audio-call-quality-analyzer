"""
ClariAI Training Pipeline

A comprehensive machine learning pipeline for training custom audio quality classifiers
using ClariAI's advanced feature extraction and multiple ML algorithms.

GitHub Repository: https://github.com/kernelseed/audio-call-quality-analyzer
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

# Machine learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, roc_curve, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score, top_k_accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# XGBoost and advanced ML
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")

# ClariAI imports
from audio_call_quality_model import ClariAIAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClariAITrainer:
    """
    ClariAI Machine Learning Training Pipeline
    
    Provides comprehensive training capabilities for custom audio quality classifiers
    using ClariAI's advanced feature extraction and multiple ML algorithms.
    """
    
    def __init__(self, 
                 test_size: float = 0.2,
                 random_state: int = 42,
                 cv_folds: int = 5):
        """
        Initialize ClariAI trainer.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random state for reproducibility
            cv_folds: Number of cross-validation folds
        """
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        
        # Initialize ClariAI analyzer
        self.analyzer = ClariAIAnalyzer()
        
        # Training data storage
        self.training_data = []
        self.feature_names = [
            'spectral_centroid', 'zcr', 'rms_energy', 'mfcc_mean', 
            'snr', 'voice_activity_ratio', 'spectral_rolloff', 
            'spectral_bandwidth', 'mfcc_std', 'dynamic_range', 'crest_factor'
        ]
        
        # Model storage
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_importance = None
        self.training_history = {}
        
        # Available models
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=random_state
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1
            ),
            'ada_boost': AdaBoostClassifier(
                n_estimators=100,
                random_state=random_state
            ),
            'svm': SVC(
                random_state=random_state,
                probability=True
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=random_state,
                max_iter=1000
            ),
            'logistic_regression': LogisticRegression(
                random_state=random_state,
                max_iter=1000
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=random_state
            ),
            'naive_bayes': GaussianNB()
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBClassifier(
                random_state=random_state,
                eval_metric='mlogloss',
                verbosity=0
            )
    
    def add_training_sample(self, 
                           audio_path: str, 
                           quality_label: str,
                           metadata: Optional[Dict] = None):
        """
        Add a training sample to the dataset.
        
        Args:
            audio_path: Path to audio file
            quality_label: Quality label (e.g., 'excellent', 'good', 'poor')
            metadata: Optional metadata dictionary
        """
        try:
            # Extract features using ClariAI
            features = self.analyzer.extract_audio_features(audio_path)
            
            # Create training sample
            sample = {
                'audio_path': audio_path,
                'quality_label': quality_label,
                'features': features,
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat()
            }
            
            self.training_data.append(sample)
            logger.info(f"Added training sample: {audio_path} -> {quality_label}")
            
        except Exception as e:
            logger.error(f"Error adding training sample {audio_path}: {e}")
            raise
    
    def load_training_data(self, data_file: str):
        """
        Load training data from JSON file.
        
        Args:
            data_file: Path to JSON file containing training data
        """
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            self.training_data = data
            logger.info(f"Loaded {len(self.training_data)} training samples from {data_file}")
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def save_training_data(self, data_file: str):
        """
        Save training data to JSON file.
        
        Args:
            data_file: Path to save training data
        """
        try:
            with open(data_file, 'w') as f:
                json.dump(self.training_data, f, indent=2)
            
            logger.info(f"Saved {len(self.training_data)} training samples to {data_file}")
            
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
            raise
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for model training.
        
        Returns:
            Tuple of (features, labels)
        """
        if not self.training_data:
            raise ValueError("No training data available. Add samples first.")
        
        # Extract features and labels
        features_list = []
        labels_list = []
        
        for sample in self.training_data:
            features = sample['features']
            label = sample['quality_label']
            
            # Create feature vector
            feature_vector = [features.get(name, 0) for name in self.feature_names]
            features_list.append(feature_vector)
            labels_list.append(label)
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def train_model(self, 
                   model_name: str = 'random_forest',
                   optimize_hyperparameters: bool = True) -> Dict[str, float]:
        """
        Train a machine learning model.
        
        Args:
            model_name: Name of model to train
            optimize_hyperparameters: Whether to optimize hyperparameters
            
        Returns:
            Dictionary of training metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.models.keys())}")
        
        try:
            # Prepare training data
            X, y = self.prepare_training_data()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)
            
            # Get model
            model = self.models[model_name]
            
            # Hyperparameter optimization
            if optimize_hyperparameters:
                model = self._optimize_hyperparameters(model, model_name, X_train_scaled, y_train_encoded)
            
            # Train model
            logger.info(f"Training {model_name} model...")
            model.fit(X_train_scaled, y_train_encoded)
            self.model = model
            
            # Evaluate model
            train_score = model.score(X_train_scaled, y_train_encoded)
            test_score = model.score(X_test_scaled, y_test_encoded)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train_encoded, cv=self.cv_folds)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance = dict(zip(self.feature_names, model.feature_importances_))
            
            # Training metrics
            metrics = {
                'model_name': model_name,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'n_samples': len(X_train),
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y))
            }
            
            self.training_history[model_name] = metrics
            
            logger.info(f"Training completed. Test accuracy: {test_score:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def _optimize_hyperparameters(self, 
                                 model, 
                                 model_name: str, 
                                 X_train: np.ndarray, 
                                 y_train: np.ndarray):
        """Optimize hyperparameters using GridSearchCV."""
        try:
            # Define parameter grids for different models
            param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'extra_trees': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'ada_boost': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'svm': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto', 0.001, 0.01]
                },
                'neural_network': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                    'learning_rate': ['constant', 'adaptive'],
                    'alpha': [0.0001, 0.001, 0.01]
                },
                'logistic_regression': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'decision_tree': {
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }
            
            # Add XGBoost parameters if available
            if XGBOOST_AVAILABLE and model_name == 'xgboost':
                param_grids['xgboost'] = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            
            if model_name not in param_grids:
                logger.warning(f"No hyperparameter grid for {model_name}, using default parameters")
                return model
            
            # Perform grid search
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name], 
                cv=3,  # Use fewer folds for speed
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            return grid_search.best_estimator_
            
        except Exception as e:
            logger.warning(f"Hyperparameter optimization failed: {e}")
            return model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Union[float, str, Dict]]:
        """
        Evaluate trained model with comprehensive metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of comprehensive evaluation metrics
        """
        if self.model is None:
            raise ValueError("No trained model available")
        
        try:
            # Scale test features
            X_test_scaled = self.scaler.transform(X_test)
            y_test_encoded = self.label_encoder.transform(y_test)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled) if hasattr(self.model, 'predict_proba') else None
            
            # Basic metrics
            accuracy = accuracy_score(y_test_encoded, y_pred)
            balanced_acc = balanced_accuracy_score(y_test_encoded, y_pred)
            
            # Precision, Recall, F1 Score (macro, micro, weighted averages)
            precision_macro = precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)
            precision_micro = precision_score(y_test_encoded, y_pred, average='micro', zero_division=0)
            precision_weighted = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            
            recall_macro = recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)
            recall_micro = recall_score(y_test_encoded, y_pred, average='micro', zero_division=0)
            recall_weighted = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            
            f1_macro = f1_score(y_test_encoded, y_pred, average='macro', zero_division=0)
            f1_micro = f1_score(y_test_encoded, y_pred, average='micro', zero_division=0)
            f1_weighted = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            
            # Additional metrics
            mcc = matthews_corrcoef(y_test_encoded, y_pred)
            kappa = cohen_kappa_score(y_test_encoded, y_pred)
            
            # ROC AUC (if probabilities available)
            roc_auc = None
            if y_pred_proba is not None and len(np.unique(y_test_encoded)) > 1:
                try:
                    if len(np.unique(y_test_encoded)) == 2:
                        # Binary classification
                        roc_auc = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
                    else:
                        # Multi-class classification
                        roc_auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr', average='macro')
                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC: {e}")
            
            # Classification report
            class_names = self.label_encoder.classes_
            report = classification_report(
                y_test_encoded, y_pred, 
                target_names=class_names, 
                output_dict=True
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_test_encoded, y_pred)
            
            # Comprehensive evaluation
            evaluation = {
                'basic_metrics': {
                    'accuracy': accuracy,
                    'balanced_accuracy': balanced_acc,
                    'matthews_correlation_coefficient': mcc,
                    'cohen_kappa': kappa
                },
                'precision_metrics': {
                    'macro': precision_macro,
                    'micro': precision_micro,
                    'weighted': precision_weighted
                },
                'recall_metrics': {
                    'macro': recall_macro,
                    'micro': recall_micro,
                    'weighted': recall_weighted
                },
                'f1_metrics': {
                    'macro': f1_macro,
                    'micro': f1_micro,
                    'weighted': f1_weighted
                },
                'roc_auc': roc_auc,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'class_names': class_names.tolist(),
                'n_samples': len(y_test),
                'n_classes': len(np.unique(y_test_encoded))
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def predict_quality(self, audio_path: str) -> str:
        """
        Predict quality for a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Predicted quality label
        """
        if self.model is None or self.scaler is None:
            return "No trained classifier available"
        
        try:
            # Extract features
            features = self.analyzer.extract_audio_features(audio_path)
            
            # Create feature vector
            feature_vector = np.array([features.get(name, 0) for name in self.feature_names]).reshape(1, -1)
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Make prediction
            prediction_encoded = self.model.predict(feature_vector_scaled)[0]
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            return prediction
            
        except Exception as e:
            return f"Error predicting quality: {e}"
    
    def save_model(self, model_path: str):
        """
        Save trained model and preprocessing objects.
        
        Args:
            model_path: Path to save model files
        """
        try:
            model_dir = Path(model_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            if self.model is not None:
                with open(model_dir / 'model.pkl', 'wb') as f:
                    pickle.dump(self.model, f)
            
            # Save scaler
            if self.scaler is not None:
                with open(model_dir / 'scaler.pkl', 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            # Save label encoder
            if self.label_encoder is not None:
                with open(model_dir / 'label_encoder.pkl', 'wb') as f:
                    pickle.dump(self.label_encoder, f)
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'training_history': self.training_history,
                'feature_importance': self.feature_importance,
                'model_type': type(self.model).__name__ if self.model else None,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(model_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_path: str):
        """
        Load trained model and preprocessing objects.
        
        Args:
            model_path: Path to model directory
        """
        try:
            model_dir = Path(model_path)
            
            # Load model
            with open(model_dir / 'model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            with open(model_dir / 'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load label encoder
            with open(model_dir / 'label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Load metadata
            with open(model_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata['feature_names']
                self.training_history = metadata['training_history']
                self.feature_importance = metadata['feature_importance']
            
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if self.feature_importance is None:
            return {}
        return self.feature_importance
    
    def compare_models(self, 
                      X: np.ndarray, 
                      y: np.ndarray, 
                      models_to_compare: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Compare multiple models using cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            models_to_compare: List of model names to compare (None for all)
            
        Returns:
            Dictionary with comparison results
        """
        if models_to_compare is None:
            models_to_compare = list(self.models.keys())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        results = {}
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name in models_to_compare:
            if model_name not in self.models:
                logger.warning(f"Model {model_name} not available, skipping...")
                continue
                
            try:
                model = self.models[model_name]
                
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
                cv_f1_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='f1_macro')
                cv_precision_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='precision_macro')
                cv_recall_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='recall_macro')
                
                results[model_name] = {
                    'accuracy': {
                        'mean': cv_scores.mean(),
                        'std': cv_scores.std(),
                        'scores': cv_scores.tolist()
                    },
                    'f1_macro': {
                        'mean': cv_f1_scores.mean(),
                        'std': cv_f1_scores.std(),
                        'scores': cv_f1_scores.tolist()
                    },
                    'precision_macro': {
                        'mean': cv_precision_scores.mean(),
                        'std': cv_precision_scores.std(),
                        'scores': cv_precision_scores.tolist()
                    },
                    'recall_macro': {
                        'mean': cv_recall_scores.mean(),
                        'std': cv_recall_scores.std(),
                        'scores': cv_recall_scores.tolist()
                    }
                }
                
                logger.info(f"Completed evaluation for {model_name}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def plot_model_comparison(self, comparison_results: Dict[str, Dict], 
                            metric: str = 'f1_macro', 
                            save_path: Optional[str] = None):
        """
        Plot model comparison results.
        
        Args:
            comparison_results: Results from compare_models
            metric: Metric to plot ('accuracy', 'f1_macro', 'precision_macro', 'recall_macro')
            save_path: Path to save plot (optional)
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available for plotting")
            return
        
        try:
            # Extract data for plotting
            model_names = []
            means = []
            stds = []
            
            for model_name, results in comparison_results.items():
                if 'error' not in results and metric in results:
                    model_names.append(model_name)
                    means.append(results[metric]['mean'])
                    stds.append(results[metric]['std'])
            
            if not model_names:
                logger.warning(f"No valid data for metric {metric}")
                return
            
            # Create plot
            plt.figure(figsize=(12, 6))
            x_pos = np.arange(len(model_names))
            
            bars = plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
            plt.xlabel('Models')
            plt.ylabel(f'{metric.replace("_", " ").title()}')
            plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
            plt.xticks(x_pos, model_names, rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (mean, std) in enumerate(zip(means, stds)):
                plt.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                        ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating plot: {e}")
    
    def plot_confusion_matrix(self, 
                            X_test: np.ndarray, 
                            y_test: np.ndarray, 
                            save_path: Optional[str] = None):
        """
        Plot confusion matrix for the trained model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save plot (optional)
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available for plotting")
            return
        
        if self.model is None:
            logger.warning("No trained model available")
            return
        
        try:
            # Get predictions
            X_test_scaled = self.scaler.transform(X_test)
            y_test_encoded = self.label_encoder.transform(y_test)
            y_pred = self.model.predict(X_test_scaled)
            
            # Create confusion matrix
            cm = confusion_matrix(y_test_encoded, y_pred)
            class_names = self.label_encoder.classes_
            
            # Plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating confusion matrix plot: {e}")
    
    def get_feature_importance_plot(self, save_path: Optional[str] = None):
        """
        Plot feature importance for the trained model.
        
        Args:
            save_path: Path to save plot (optional)
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available for plotting")
            return
        
        if self.feature_importance is None:
            logger.warning("No feature importance available")
            return
        
        try:
            # Sort features by importance
            sorted_features = sorted(self.feature_importance.items(), 
                                  key=lambda x: x[1], reverse=True)
            
            features, importances = zip(*sorted_features)
            
            # Create plot
            plt.figure(figsize=(10, 6))
            bars = plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance Analysis')
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (bar, importance) in enumerate(zip(bars, importances)):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{importance:.3f}', va='center', fontsize=8)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {e}")
    
    def get_training_summary(self) -> Dict[str, Union[str, int, float]]:
        """Get training summary."""
        return {
            'n_samples': len(self.training_data),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'available_models': list(self.models.keys()),
            'trained_model': type(self.model).__name__ if self.model else None,
            'training_history': self.training_history,
            'xgboost_available': XGBOOST_AVAILABLE,
            'plotting_available': PLOTTING_AVAILABLE
        }

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ClariAI Training Pipeline")
    parser.add_argument("--data", help="Path to training data JSON file")
    parser.add_argument("--model", default="random_forest", help="Model to train")
    parser.add_argument("--output", help="Output directory for trained model")
    parser.add_argument("--no-optimize", action="store_true", help="Skip hyperparameter optimization")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ClariAITrainer()
    
    # Load training data if provided
    if args.data:
        trainer.load_training_data(args.data)
    
    # Train model
    print(f"Training {args.model} model...")
    metrics = trainer.train_model(
        model_name=args.model,
        optimize_hyperparameters=not args.no_optimize
    )
    
    # Print results
    print("\n🎯 ClariAI Training Results")
    print("=" * 50)
    print(f"Model: {metrics['model_name']}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.3f}")
    print(f"CV Mean: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
    print(f"Samples: {metrics['n_samples']}")
    print(f"Features: {metrics['n_features']}")
    print(f"Classes: {metrics['n_classes']}")
    
    # Show available models
    print(f"\nAvailable Models: {', '.join(trainer.models.keys())}")
    print(f"XGBoost Available: {XGBOOST_AVAILABLE}")
    print(f"Plotting Available: {PLOTTING_AVAILABLE}")
    
    # Save model if output directory provided
    if args.output:
        trainer.save_model(args.output)
        print(f"\nModel saved to: {args.output}")

if __name__ == "__main__":
    main()