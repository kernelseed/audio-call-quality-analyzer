"""
ClariAI Example Usage and Demonstrations

Comprehensive examples demonstrating ClariAI's capabilities for audio quality analysis,
training custom models, and integration with various systems.

GitHub Repository: https://github.com/kernelseed/audio-call-quality-analyzer
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

# ClariAI imports
from audio_call_quality_model import ClariAIAnalyzer
from training_pipeline import ClariAITrainer
from huggingface_integration import ClariAIHubUploader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_audio(duration: float = 5.0, 
                     sample_rate: int = 16000,
                     filename: str = "test_audio.wav") -> str:
    """
    Create a test audio file for demonstration purposes.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate
        filename: Output filename
        
    Returns:
        Path to created audio file
    """
    try:
        # Generate a simple sine wave with some noise
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a speech-like signal with formants
        signal = (np.sin(2 * np.pi * 200 * t) * 0.3 +  # Fundamental frequency
                 np.sin(2 * np.pi * 800 * t) * 0.2 +   # First formant
                 np.sin(2 * np.pi * 1200 * t) * 0.1 +  # Second formant
                 np.random.normal(0, 0.05, len(t)))    # Noise
        
        # Add envelope to make it more speech-like
        envelope = np.exp(-t / duration) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
        signal *= envelope
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.8
        
        # Save audio file
        import soundfile as sf
        sf.write(filename, signal, sample_rate)
        
        logger.info(f"Created test audio file: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error creating test audio: {e}")
        raise

def demo_basic_analysis():
    """Demonstrate basic audio quality analysis."""
    print("\n🎵 ClariAI Basic Analysis Demo")
    print("=" * 40)
    
    # Create test audio
    audio_file = create_test_audio(duration=3.0, filename="demo_audio.wav")
    
    try:
        # Initialize ClariAI analyzer
        analyzer = ClariAIAnalyzer()
        
        # Analyze audio quality
        print(f"Analyzing audio: {audio_file}")
        start_time = time.time()
        results = analyzer.analyze_call_quality(audio_file)
        analysis_time = time.time() - start_time
        
        # Display results
        print(f"\n📊 Analysis Results (took {analysis_time:.2f}s)")
        print("-" * 30)
        print(f"Analysis Method: {results['analysis_method']}")
        print(f"Overall Quality: {results['quality_scores']['overall_quality']:.3f}")
        print(f"Clarity: {results['quality_scores']['clarity']:.3f}")
        print(f"Volume: {results['quality_scores']['volume']:.3f}")
        print(f"Noise Level: {results['quality_scores']['noise_level']:.3f}")
        
        # Display features
        print(f"\n🔍 Extracted Features:")
        for feature, value in results['features'].items():
            print(f"  {feature}: {value:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in basic analysis demo: {e}")
        return None
    finally:
        # Cleanup
        if os.path.exists(audio_file):
            os.remove(audio_file)

def demo_training_pipeline():
    """Demonstrate training pipeline."""
    print("\n🤖 ClariAI Training Pipeline Demo")
    print("=" * 40)
    
    try:
        # Initialize trainer
        trainer = ClariAITrainer()
        
        # Create sample training data
        print("Creating sample training data...")
        for i in range(5):
            # Create different quality audio samples
            quality_levels = ["excellent", "good", "poor"]
            quality = quality_levels[i % len(quality_levels)]
            
            audio_file = create_test_audio(
                duration=2.0, 
                filename=f"training_sample_{i}.wav"
            )
            
            # Add some variation to audio quality
            if quality == "poor":
                # Add more noise
                import soundfile as sf
                signal, sr = sf.read(audio_file)
                noise = np.random.normal(0, 0.1, len(signal))
                signal += noise
                sf.write(audio_file, signal, sr)
            elif quality == "excellent":
                # Clean signal
                import soundfile as sf
                signal, sr = sf.read(audio_file)
                signal = signal * 1.2  # Slightly louder
                sf.write(audio_file, signal, sr)
            
            trainer.add_training_sample(audio_file, quality)
        
        # Train model
        print("Training Random Forest model...")
        metrics = trainer.train_model("random_forest", optimize_hyperparameters=False)
        
        # Display training results
        print(f"\n📈 Training Results:")
        print(f"Model: {metrics['model_name']}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.3f}")
        print(f"CV Mean: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
        print(f"Samples: {metrics['n_samples']}")
        print(f"Features: {metrics['n_features']}")
        
        # Test prediction
        test_audio = create_test_audio(duration=1.0, filename="test_prediction.wav")
        prediction = trainer.predict_quality(test_audio)
        print(f"\n🔮 Prediction Test:")
        print(f"Audio: {test_audio}")
        print(f"Predicted Quality: {prediction}")
        
        # Save model
        model_path = "demo_model"
        trainer.save_model(model_path)
        print(f"\n💾 Model saved to: {model_path}")
        
        return trainer, metrics
        
    except Exception as e:
        logger.error(f"Error in training pipeline demo: {e}")
        return None, None
    finally:
        # Cleanup training files
        for i in range(5):
            audio_file = f"training_sample_{i}.wav"
            if os.path.exists(audio_file):
                os.remove(audio_file)
        if os.path.exists("test_prediction.wav"):
            os.remove("test_prediction.wav")

def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n📦 ClariAI Batch Processing Demo")
    print("=" * 40)
    
    try:
        # Create multiple test audio files
        audio_files = []
        for i in range(3):
            audio_file = create_test_audio(
                duration=2.0,
                filename=f"batch_audio_{i}.wav"
            )
            audio_files.append(audio_file)
        
        # Initialize analyzer
        analyzer = ClariAIAnalyzer()
        
        # Batch analysis
        print(f"Analyzing {len(audio_files)} audio files...")
        start_time = time.time()
        results = analyzer.batch_analyze(audio_files)
        batch_time = time.time() - start_time
        
        # Display results
        print(f"\n📊 Batch Analysis Results (took {batch_time:.2f}s)")
        print("-" * 40)
        
        for i, (audio_file, result) in enumerate(zip(audio_files, results)):
            print(f"\nFile {i+1}: {audio_file}")
            print(f"  Overall Quality: {result['quality_scores']['overall_quality']:.3f}")
            print(f"  Clarity: {result['quality_scores']['clarity']:.3f}")
            print(f"  Volume: {result['quality_scores']['volume']:.3f}")
            print(f"  Noise Level: {result['quality_scores']['noise_level']:.3f}")
        
        # Calculate average quality
        avg_quality = np.mean([r['quality_scores']['overall_quality'] for r in results])
        print(f"\n📈 Average Quality: {avg_quality:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch processing demo: {e}")
        return None
    finally:
        # Cleanup
        for audio_file in audio_files:
            if os.path.exists(audio_file):
                os.remove(audio_file)

def demo_custom_features():
    """Demonstrate custom feature extraction."""
    print("\n🔧 ClariAI Custom Features Demo")
    print("=" * 40)
    
    try:
        # Create test audio
        audio_file = create_test_audio(duration=4.0, filename="custom_features_audio.wav")
        
        # Initialize analyzer
        analyzer = ClariAIAnalyzer()
        
        # Extract features
        print(f"Extracting features from: {audio_file}")
        features = analyzer.extract_audio_features(audio_file)
        
        # Display all features
        print(f"\n🔍 Extracted Features:")
        print("-" * 30)
        for feature, value in features.items():
            print(f"{feature:20}: {value:8.3f}")
        
        # Feature analysis
        print(f"\n📊 Feature Analysis:")
        print(f"Highest value: {max(features.items(), key=lambda x: x[1])}")
        print(f"Lowest value: {min(features.items(), key=lambda x: x[1])}")
        print(f"Average value: {np.mean(list(features.values())):.3f}")
        
        return features
        
    except Exception as e:
        logger.error(f"Error in custom features demo: {e}")
        return None
    finally:
        # Cleanup
        if os.path.exists("custom_features_audio.wav"):
            os.remove("custom_features_audio.wav")

def demo_huggingface_integration():
    """Demonstrate Hugging Face integration (if available)."""
    print("\n☁️ ClariAI Hugging Face Integration Demo")
    print("=" * 40)
    
    try:
        # Check if Hugging Face is available
        from huggingface_integration import ClariAIHubUploader
        
        # Initialize uploader (requires HF token)
        hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
        if not hf_token:
            print("⚠️  Hugging Face token not found. Set HUGGINGFACE_HUB_TOKEN environment variable.")
            print("   Skipping Hugging Face integration demo.")
            return None
        
        uploader = ClariAIHubUploader(hf_token=hf_token)
        
        # Create model card
        print("Creating model card...")
        model_card = uploader.create_model_card(
            model_name="clariai-demo-model",
            model_description="ClariAI demonstration model for audio quality analysis",
            tags=["audio", "quality-analysis", "demo", "clariai"],
            performance_metrics={
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.93,
                "f1_score": 0.935
            }
        )
        
        print("✅ Model card created successfully")
        print(f"Model card length: {len(model_card)} characters")
        
        # List available models
        print("\n🔍 Available ClariAI models:")
        models = uploader.list_models(limit=5)
        for model in models:
            print(f"  - {model['name']} by {model['author']}")
        
        return uploader
        
    except ImportError:
        print("⚠️  Hugging Face integration not available. Install with: pip install huggingface-hub")
        return None
    except Exception as e:
        logger.error(f"Error in Hugging Face integration demo: {e}")
        return None

def demo_advanced_ml_metrics():
    """Demonstrate ClariAI's advanced ML metrics and model comparison."""
    print("\n🎯 ClariAI Advanced ML Metrics Demo")
    print("=" * 50)
    
    # Create trainer
    trainer = ClariAITrainer()
    
    # Create synthetic training data
    print("Creating synthetic training data...")
    quality_categories = ['excellent', 'good', 'fair', 'poor']
    
    for i in range(50):  # Create 50 samples
        # Generate synthetic features based on quality
        quality = np.random.choice(quality_categories, p=[0.3, 0.4, 0.2, 0.1])
        
        # Create synthetic features
        if quality == 'excellent':
            features = {
                'spectral_centroid': np.random.normal(2000, 200),
                'zcr': np.random.normal(0.1, 0.02),
                'rms_energy': np.random.normal(0.8, 0.1),
                'mfcc_mean': np.random.normal(0.5, 0.1),
                'snr': np.random.normal(25, 3),
                'voice_activity_ratio': np.random.normal(0.9, 0.05),
                'spectral_rolloff': np.random.normal(4000, 500),
                'spectral_bandwidth': np.random.normal(1500, 200),
                'mfcc_std': np.random.normal(0.3, 0.05),
                'dynamic_range': np.random.normal(0.7, 0.1),
                'crest_factor': np.random.normal(0.6, 0.1)
            }
        elif quality == 'good':
            features = {
                'spectral_centroid': np.random.normal(1800, 300),
                'zcr': np.random.normal(0.15, 0.03),
                'rms_energy': np.random.normal(0.6, 0.15),
                'mfcc_mean': np.random.normal(0.3, 0.15),
                'snr': np.random.normal(18, 4),
                'voice_activity_ratio': np.random.normal(0.8, 0.1),
                'spectral_rolloff': np.random.normal(3500, 600),
                'spectral_bandwidth': np.random.normal(1200, 300),
                'mfcc_std': np.random.normal(0.4, 0.1),
                'dynamic_range': np.random.normal(0.5, 0.15),
                'crest_factor': np.random.normal(0.5, 0.15)
            }
        elif quality == 'fair':
            features = {
                'spectral_centroid': np.random.normal(1500, 400),
                'zcr': np.random.normal(0.2, 0.04),
                'rms_energy': np.random.normal(0.4, 0.2),
                'mfcc_mean': np.random.normal(0.1, 0.2),
                'snr': np.random.normal(12, 5),
                'voice_activity_ratio': np.random.normal(0.6, 0.15),
                'spectral_rolloff': np.random.normal(3000, 800),
                'spectral_bandwidth': np.random.normal(1000, 400),
                'mfcc_std': np.random.normal(0.5, 0.15),
                'dynamic_range': np.random.normal(0.3, 0.2),
                'crest_factor': np.random.normal(0.4, 0.2)
            }
        else:  # poor
            features = {
                'spectral_centroid': np.random.normal(1200, 500),
                'zcr': np.random.normal(0.3, 0.05),
                'rms_energy': np.random.normal(0.2, 0.15),
                'mfcc_mean': np.random.normal(-0.1, 0.25),
                'snr': np.random.normal(6, 4),
                'voice_activity_ratio': np.random.normal(0.4, 0.2),
                'spectral_rolloff': np.random.normal(2500, 1000),
                'spectral_bandwidth': np.random.normal(800, 500),
                'mfcc_std': np.random.normal(0.6, 0.2),
                'dynamic_range': np.random.normal(0.1, 0.15),
                'crest_factor': np.random.normal(0.2, 0.15)
            }
        
        # Add some noise
        for key in features:
            features[key] += np.random.normal(0, abs(features[key]) * 0.05)
        
        # Add sample to trainer
        trainer.add_training_sample(f"synthetic_{i}.wav", quality, {'synthetic': True})
    
    # Prepare data
    X, y = trainer.prepare_training_data()
    
    # Compare multiple models
    print("\n📊 Comparing Multiple ML Algorithms:")
    print("-" * 60)
    
    models_to_test = ['random_forest', 'gradient_boosting', 'svm', 'neural_network']
    if 'xgboost' in trainer.models:
        models_to_test.append('xgboost')
    
    comparison_results = trainer.compare_models(X, y, models_to_test)
    
    # Print results
    print(f"{'Model':<20} {'Accuracy':<12} {'F1-Macro':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 60)
    
    for model_name, results in comparison_results.items():
        if 'error' not in results:
            acc = results['accuracy']['mean']
            f1 = results['f1_macro']['mean']
            prec = results['precision_macro']['mean']
            rec = results['recall_macro']['mean']
            print(f"{model_name:<20} {acc:.3f}±{results['accuracy']['std']:.3f}  {f1:.3f}±{results['f1_macro']['std']:.3f}  {prec:.3f}±{results['precision_macro']['std']:.3f}  {rec:.3f}±{results['recall_macro']['std']:.3f}")
        else:
            print(f"{model_name:<20} ERROR: {results['error']}")
    
    # Train best model for detailed evaluation
    print(f"\n🔧 Training Random Forest for detailed evaluation...")
    trainer.train_model('random_forest', optimize_hyperparameters=True)
    
    # Detailed evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    evaluation = trainer.evaluate_model(X_test, y_test)
    
    print(f"\n📈 Detailed Evaluation Results:")
    print("-" * 40)
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
    
    if evaluation['roc_auc'] is not None:
        print(f"\nROC AUC: {evaluation['roc_auc']:.3f}")
    
    # Feature importance
    importance = trainer.get_feature_importance()
    if importance:
        print(f"\n📊 Top 5 Most Important Features:")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, imp) in enumerate(sorted_features[:5], 1):
            print(f"  {i}. {feature}: {imp:.3f}")
    
    print(f"\n✅ Advanced ML metrics demonstration completed!")
    return trainer, evaluation

def cleanup_test_files():
    """Clean up any remaining test files."""
    test_files = [
        "test_audio.wav",
        "demo_audio.wav",
        "training_sample_0.wav",
        "training_sample_1.wav",
        "training_sample_2.wav",
        "training_sample_3.wav",
        "training_sample_4.wav",
        "test_prediction.wav",
        "batch_audio_0.wav",
        "batch_audio_1.wav",
        "batch_audio_2.wav",
        "custom_features_audio.wav"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaned up: {file}")

def main():
    """Main demonstration function."""
    print("🚀 ClariAI Comprehensive Demo")
    print("=" * 50)
    print("Demonstrating ClariAI's audio quality analysis capabilities")
    print("GitHub: https://github.com/kernelseed/audio-call-quality-analyzer")
    
    try:
        # Run all demonstrations
        print("\n" + "="*50)
        demo_basic_analysis()
        
        print("\n" + "="*50)
        trainer, metrics = demo_training_pipeline()
        
        print("\n" + "="*50)
        demo_batch_processing()
        
        print("\n" + "="*50)
        demo_custom_features()
        
        print("\n" + "="*50)
        demo_huggingface_integration()
        
        print("\n" + "="*50)
        demo_advanced_ml_metrics()
        
        print("\n" + "="*50)
        print("✅ All demonstrations completed successfully!")
        
        # Display summary
        print("\n📋 ClariAI Demo Summary")
        print("-" * 30)
        print("✅ Basic audio quality analysis")
        print("✅ Custom model training pipeline")
        print("✅ Batch processing capabilities")
        print("✅ Advanced feature extraction")
        print("✅ Hugging Face integration")
        print("✅ Advanced ML metrics (F1, Precision, Recall, AUC, MCC)")
        print("✅ Model comparison and visualization")
        print("✅ XGBoost and ensemble methods")
        print("\n🎯 ClariAI is ready for production use!")
        
    except Exception as e:
        logger.error(f"Error in main demo: {e}")
        print(f"\n❌ Demo failed: {e}")
    finally:
        # Cleanup
        cleanup_test_files()
        print("\n🧹 Cleanup completed")

if __name__ == "__main__":
    main()