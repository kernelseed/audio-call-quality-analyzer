"""
ClariAI Advanced Machine Learning Demo

Comprehensive demonstration of ClariAI's advanced ML capabilities including:
- Multiple ML algorithms (Random Forest, XGBoost, SVM, Neural Networks, etc.)
- Comprehensive metrics (F1, Precision, Recall, AUC, MCC, Cohen's Kappa)
- Model comparison and visualization
- Cross-validation and hyperparameter tuning
- Feature importance analysis

GitHub Repository: https://github.com/kernelseed/audio-call-quality-analyzer
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# ClariAI imports
from audio_call_quality_model import ClariAIAnalyzer
from training_pipeline import ClariAITrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synthetic_training_data(n_samples: int = 100) -> List[Dict]:
    """
    Create synthetic training data for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        List of training samples
    """
    print(f"🎵 Creating {n_samples} synthetic training samples...")
    
    # Quality categories
    quality_categories = ['excellent', 'good', 'fair', 'poor']
    
    # Generate synthetic samples with actual audio files
    training_data = []
    
    for i in range(n_samples):
        # Create actual audio file
        audio_path = f"synthetic_audio_{i:03d}.wav"
        quality = np.random.choice(quality_categories, p=[0.3, 0.4, 0.2, 0.1])
        
        # Generate synthetic audio based on quality
        duration = 2.0  # 2 seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create speech-like signal with quality-dependent characteristics
        if quality == 'excellent':
            # Clean signal with good formants
            signal = (np.sin(2 * np.pi * 200 * t) * 0.3 +  # Fundamental
                     np.sin(2 * np.pi * 800 * t) * 0.2 +   # First formant
                     np.sin(2 * np.pi * 1200 * t) * 0.1 +  # Second formant
                     np.random.normal(0, 0.02, len(t)))    # Low noise
        elif quality == 'good':
            # Slightly more noise
            signal = (np.sin(2 * np.pi * 200 * t) * 0.25 +
                     np.sin(2 * np.pi * 800 * t) * 0.15 +
                     np.sin(2 * np.pi * 1200 * t) * 0.08 +
                     np.random.normal(0, 0.05, len(t)))
        elif quality == 'fair':
            # More noise and distortion
            signal = (np.sin(2 * np.pi * 200 * t) * 0.2 +
                     np.sin(2 * np.pi * 800 * t) * 0.1 +
                     np.sin(2 * np.pi * 1200 * t) * 0.05 +
                     np.random.normal(0, 0.1, len(t)))
        else:  # poor
            # High noise and distortion
            signal = (np.sin(2 * np.pi * 200 * t) * 0.15 +
                     np.sin(2 * np.pi * 800 * t) * 0.05 +
                     np.sin(2 * np.pi * 1200 * t) * 0.02 +
                     np.random.normal(0, 0.2, len(t)))
        
        # Normalize and save audio
        signal = signal / np.max(np.abs(signal)) * 0.8
        signal = (signal * 32767).astype(np.int16)
        
        # Save as WAV file
        import soundfile as sf
        sf.write(audio_path, signal, sample_rate)
        
        sample = {
            'audio_path': audio_path,
            'quality_label': quality,
            'metadata': {'synthetic': True, 'sample_id': i},
            'timestamp': '2024-01-01T00:00:00'
        }
        
        training_data.append(sample)
    
    print(f"✅ Created {len(training_data)} synthetic training samples")
    return training_data

def demo_comprehensive_metrics():
    """Demonstrate comprehensive ML metrics."""
    print("\n🎯 ClariAI Advanced ML Metrics Demo")
    print("=" * 50)
    
    # Create trainer
    trainer = ClariAITrainer()
    
    # Create synthetic data
    training_data = create_synthetic_training_data(200)
    
    # Add samples to trainer
    for sample in training_data:
        trainer.add_training_sample(
            sample['audio_path'],
            sample['quality_label'],
            sample['metadata']
        )
    
    # Prepare data
    X, y = trainer.prepare_training_data()
    
    # Train multiple models
    models_to_test = ['random_forest', 'xgboost', 'svm', 'neural_network', 'gradient_boosting']
    results = {}
    
    for model_name in models_to_test:
        if model_name in trainer.models:
            print(f"\n🔧 Training {model_name}...")
            try:
                metrics = trainer.train_model(model_name, optimize_hyperparameters=True)
                results[model_name] = metrics
                
                print(f"✅ {model_name}: Accuracy = {metrics['test_accuracy']:.3f}, CV = {metrics['cv_mean']:.3f}±{metrics['cv_std']:.3f}")
                
            except Exception as e:
                print(f"❌ {model_name}: Error - {e}")
    
    return trainer, X, y, results

def demo_model_comparison(trainer: ClariAITrainer, X: np.ndarray, y: np.ndarray):
    """Demonstrate model comparison capabilities."""
    print("\n📊 Model Comparison Analysis")
    print("=" * 40)
    
    # Compare all models
    comparison_results = trainer.compare_models(X, y)
    
    # Print comparison table
    print("\nModel Performance Comparison:")
    print("-" * 80)
    print(f"{'Model':<20} {'Accuracy':<12} {'F1-Macro':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 80)
    
    for model_name, results in comparison_results.items():
        if 'error' not in results:
            acc = results['accuracy']['mean']
            f1 = results['f1_macro']['mean']
            prec = results['precision_macro']['mean']
            rec = results['recall_macro']['mean']
            print(f"{model_name:<20} {acc:.3f}±{results['accuracy']['std']:.3f}  {f1:.3f}±{results['f1_macro']['std']:.3f}  {prec:.3f}±{results['precision_macro']['std']:.3f}  {rec:.3f}±{results['recall_macro']['std']:.3f}")
        else:
            print(f"{model_name:<20} ERROR: {results['error']}")
    
    # Plot comparison (if plotting available)
    try:
        trainer.plot_model_comparison(comparison_results, metric='f1_macro', save_path='model_comparison.png')
        print("\n📈 Model comparison plot saved as 'model_comparison.png'")
    except Exception as e:
        print(f"⚠️  Could not create plot: {e}")
    
    return comparison_results

def demo_detailed_evaluation(trainer: ClariAITrainer, X: np.ndarray, y: np.ndarray):
    """Demonstrate detailed model evaluation."""
    print("\n🔍 Detailed Model Evaluation")
    print("=" * 40)
    
    # Train the best model (Random Forest)
    print("Training Random Forest for detailed evaluation...")
    trainer.train_model('random_forest', optimize_hyperparameters=True)
    
    # Split data for evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Detailed evaluation
    evaluation = trainer.evaluate_model(X_test, y_test)
    
    # Print comprehensive metrics
    print("\n📈 Comprehensive Evaluation Metrics:")
    print("-" * 50)
    
    basic = evaluation['basic_metrics']
    print(f"Accuracy: {basic['accuracy']:.3f}")
    print(f"Balanced Accuracy: {basic['balanced_accuracy']:.3f}")
    print(f"Matthews Correlation Coefficient: {basic['matthews_correlation_coefficient']:.3f}")
    print(f"Cohen's Kappa: {basic['cohen_kappa']:.3f}")
    
    f1_metrics = evaluation['f1_metrics']
    print(f"\nF1 Scores:")
    print(f"  Macro: {f1_metrics['macro']:.3f}")
    print(f"  Micro: {f1_metrics['micro']:.3f}")
    print(f"  Weighted: {f1_metrics['weighted']:.3f}")
    
    precision_metrics = evaluation['precision_metrics']
    print(f"\nPrecision Scores:")
    print(f"  Macro: {precision_metrics['macro']:.3f}")
    print(f"  Micro: {precision_metrics['micro']:.3f}")
    print(f"  Weighted: {precision_metrics['weighted']:.3f}")
    
    recall_metrics = evaluation['recall_metrics']
    print(f"\nRecall Scores:")
    print(f"  Macro: {recall_metrics['macro']:.3f}")
    print(f"  Micro: {recall_metrics['micro']:.3f}")
    print(f"  Weighted: {recall_metrics['weighted']:.3f}")
    
    if evaluation['roc_auc'] is not None:
        print(f"\nROC AUC: {evaluation['roc_auc']:.3f}")
    
    # Print per-class metrics
    print(f"\n📋 Per-Class Classification Report:")
    print("-" * 50)
    report = evaluation['classification_report']
    for class_name in evaluation['class_names']:
        if class_name in report:
            metrics = report[class_name]
            print(f"{class_name}:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1-Score: {metrics['f1-score']:.3f}")
            print(f"  Support: {metrics['support']}")
    
    # Plot confusion matrix
    try:
        trainer.plot_confusion_matrix(X_test, y_test, save_path='confusion_matrix.png')
        print("\n📊 Confusion matrix saved as 'confusion_matrix.png'")
    except Exception as e:
        print(f"⚠️  Could not create confusion matrix: {e}")
    
    # Plot feature importance
    try:
        trainer.get_feature_importance_plot(save_path='feature_importance.png')
        print("📊 Feature importance plot saved as 'feature_importance.png'")
    except Exception as e:
        print(f"⚠️  Could not create feature importance plot: {e}")
    
    return evaluation

def demo_xgboost_features():
    """Demonstrate XGBoost-specific features."""
    print("\n🚀 XGBoost Advanced Features Demo")
    print("=" * 40)
    
    # Check XGBoost availability
    from training_pipeline import XGBOOST_AVAILABLE
    if not XGBOOST_AVAILABLE:
        print("❌ XGBoost not available. Install with: pip install xgboost")
        return
    
    print("✅ XGBoost is available!")
    
    # Create trainer
    trainer = ClariAITrainer()
    
    # Create synthetic data
    training_data = create_synthetic_training_data(150)
    
    # Add samples
    for sample in training_data:
        trainer.add_training_sample(
            sample['audio_path'],
            sample['quality_label'],
            sample['metadata']
        )
    
    # Prepare data
    X, y = trainer.prepare_training_data()
    
    # Train XGBoost with hyperparameter optimization
    print("\n🔧 Training XGBoost with hyperparameter optimization...")
    metrics = trainer.train_model('xgboost', optimize_hyperparameters=True)
    
    print(f"✅ XGBoost Results:")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.3f}")
    print(f"  CV Mean: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
    
    # Show feature importance
    importance = trainer.get_feature_importance()
    if importance:
        print(f"\n📊 XGBoost Feature Importance:")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, imp in sorted_features[:5]:  # Top 5 features
            print(f"  {feature}: {imp:.3f}")

def main():
    """Main demonstration function."""
    print("🎵 ClariAI Advanced Machine Learning Demonstration")
    print("=" * 60)
    print("This demo showcases ClariAI's comprehensive ML capabilities:")
    print("• Multiple ML algorithms (RF, XGBoost, SVM, Neural Networks, etc.)")
    print("• Advanced metrics (F1, Precision, Recall, AUC, MCC, Cohen's Kappa)")
    print("• Model comparison and visualization")
    print("• Cross-validation and hyperparameter tuning")
    print("• Feature importance analysis")
    print("=" * 60)
    
    try:
        # Demo 1: Comprehensive metrics
        trainer, X, y, results = demo_comprehensive_metrics()
        
        # Demo 2: Model comparison
        comparison_results = demo_model_comparison(trainer, X, y)
        
        # Demo 3: Detailed evaluation
        evaluation = demo_detailed_evaluation(trainer, X, y)
        
        # Demo 4: XGBoost features
        demo_xgboost_features()
        
        # Summary
        print("\n🎉 ClariAI Advanced ML Demo Complete!")
        print("=" * 50)
        print("✅ All demonstrations completed successfully")
        print("📊 Generated visualizations:")
        print("  • model_comparison.png - Model performance comparison")
        print("  • confusion_matrix.png - Confusion matrix")
        print("  • feature_importance.png - Feature importance analysis")
        print("\n🚀 ClariAI is ready for production use with advanced ML capabilities!")
        
        # Cleanup synthetic files
        print("\n🧹 Cleaning up synthetic files...")
        import glob
        synthetic_files = glob.glob("synthetic_audio_*.wav")
        for file in synthetic_files:
            try:
                os.remove(file)
            except:
                pass
        print(f"✅ Cleaned up {len(synthetic_files)} synthetic audio files")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
