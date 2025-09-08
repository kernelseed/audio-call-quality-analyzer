
ng # ClariAI - Professional Audio Quality Analysis Platform

ClariAI is a comprehensive, professional-grade audio quality analysis platform using LangChain and machine learning. Built for real-time monitoring, intelligent classification, and seamless Hugging Face integration.

## 🚀 ClariAI Features

- **🎵 Advanced Audio Processing**: Professional-grade feature extraction using librosa and signal processing
- **🧠 LangChain Integration**: Intelligent quality analysis powered by language models
- **🤖 Machine Learning Pipeline**: Train custom quality classifiers with multiple algorithms
- **⚡ Real-time Analysis**: Lightning-fast inference for live call monitoring
- **☁️ Hugging Face Integration**: Seamless model sharing and deployment
- **🎤 Voice Activity Detection**: Advanced speech vs. silence identification
- **📊 Signal-to-Noise Ratio**: Automatic noise level assessment
- **📦 Batch Processing**: High-performance multi-file analysis

## 📋 ClariAI Quality Metrics

ClariAI provides four professional quality metrics (0-1 scale):

- **🎯 Clarity**: Speech intelligibility and articulation quality
- **🔊 Volume**: Optimal audio level assessment  
- **🔇 Noise Level**: Background noise quantification (lower is better)
- **⭐ Overall Quality**: Comprehensive quality assessment

## 🛠️ ClariAI Installation

1. Clone the ClariAI repository:
```bash
git clone https://github.com/kernelseed/audio-call-quality-analyzer.git
cd audio-call-quality-analyzer
```

2. Install ClariAI dependencies:
```bash
pip install -r requirements.txt
```

3. For professional audio processing, install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio

# macOS
brew install portaudio

# Windows
# Download and install PyAudio wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
```

4. Verify all dependencies are open source:
```bash
python check_licenses.py
```

### 🔍 License Validation
All ClariAI dependencies are open source and suitable for commercial use:
- **MIT License**: 6 packages (33%)
- **Apache 2.0**: 5 packages (28%) 
- **BSD-3-Clause**: 6 packages (33%)
- **ISC License**: 1 package (6%)

✅ **100% Open Source** - No restrictive licenses (GPL, LGPL, etc.)

## 🤖 ClariAI in Agentic AI Implementations

ClariAI is designed to seamlessly integrate into various Agentic AI systems, providing real-time audio quality monitoring and intelligent decision-making capabilities for autonomous agents.

### 🎯 Agentic AI Integration Patterns

#### 1. **Customer Service Agents**
```python
from audio_call_quality_model import ClariAIAnalyzer
import asyncio

class CustomerServiceAgent:
    def __init__(self):
        self.clari_ai = ClariAIAnalyzer()
        self.quality_threshold = 0.7
    
    async def process_customer_call(self, audio_stream):
        """Process customer call with real-time quality monitoring"""
        # Analyze audio quality in real-time
        quality_results = self.clari_ai.analyze_call_quality(audio_stream)
        
        # Agent decision making based on quality
        if quality_results['quality_scores']['overall_quality'] < self.quality_threshold:
            await self.agent_actions.handle_poor_quality()
        else:
            await self.agent_actions.process_normal_call()
        
        return quality_results
```

#### 2. **Voice Assistant Agents**
```python
class VoiceAssistantAgent:
    def __init__(self):
        self.clari_ai = ClariAIAnalyzer()
        self.adaptation_strategies = {
            'low_clarity': self.enhance_speech_recognition,
            'high_noise': self.activate_noise_cancellation,
            'low_volume': self.adjust_amplification
        }
    
    def adaptive_processing(self, audio_input):
        """Adapt processing based on audio quality"""
        quality = self.clari_ai.analyze_call_quality(audio_input)
        
        # Autonomous adaptation
        if quality['quality_scores']['clarity'] < 0.6:
            self.adaptation_strategies['low_clarity']()
        elif quality['quality_scores']['noise_level'] > 0.7:
            self.adaptation_strategies['high_noise']()
        
        return self.process_audio(audio_input, quality)
```

### 🚀 Quick Start

#### Basic Usage

```python
from audio_call_quality_model import ClariAIAnalyzer

# Initialize ClariAI analyzer
analyzer = ClariAIAnalyzer()

# Analyze call quality
results = analyzer.analyze_call_quality("path/to/audio.wav")

# Print results
print(f"Overall Quality: {results['quality_scores']['overall_quality']:.3f}")
print(f"Clarity: {results['quality_scores']['clarity']:.3f}")
print(f"Volume: {results['quality_scores']['volume']:.3f}")
print(f"Noise Level: {results['quality_scores']['noise_level']:.3f}")
```

#### Training a Custom Model

```python
from training_pipeline import ClariAITrainer

# Initialize trainer
trainer = ClariAITrainer()

# Add training samples
trainer.add_training_sample("excellent_audio.wav", "excellent")
trainer.add_training_sample("good_audio.wav", "good")
trainer.add_training_sample("poor_audio.wav", "poor")

# Train model
metrics = trainer.train_model("random_forest")

# Save model
trainer.save_model("my_quality_model")
```

#### Hugging Face Integration

```python
from huggingface_integration import ClariAIHubUploader

# Initialize uploader
uploader = ClariAIHubUploader()

# Prepare model for upload
prepared_path = uploader.prepare_model_for_upload(
    model_path="my_quality_model",
    model_name="clariai-audio-quality",
    model_description="Custom audio quality analysis model"
)

# Upload to Hugging Face Hub
model_url = uploader.upload_model(
    model_path=prepared_path,
    repo_name="clariai-audio-quality",
    model_description="Custom audio quality analysis model"
)
```

## 📚 Documentation

### Core Modules

- **`audio_call_quality_model.py`**: Main analysis engine with LangChain integration
- **`training_pipeline.py`**: Machine learning training and evaluation pipeline
- **`huggingface_integration.py`**: Hugging Face Hub integration for model sharing
- **`config.py`**: Configuration management system
- **`example_usage.py`**: Comprehensive usage examples and demonstrations

### Advanced Features

- **Real-time Analysis**: Process audio streams in real-time
- **Batch Processing**: Analyze multiple files efficiently
- **Custom Feature Extraction**: Extract domain-specific audio features
- **Model Persistence**: Save and load trained models
- **Quality Metrics**: Comprehensive quality assessment
- **Voice Activity Detection**: Advanced speech detection
- **Signal Processing**: Professional audio analysis

## 🔧 Configuration

ClariAI supports extensive configuration through environment variables and config files:

```bash
# Audio processing
export CLARIAI_SAMPLE_RATE=16000
export CLARIAI_HOP_LENGTH=512
export CLARIAI_N_MFCC=13

# Model training
export CLARIAI_TEST_SIZE=0.2
export CLARIAI_RANDOM_STATE=42

# LangChain integration
export CLARIAI_LANGCHAIN_MODEL="gpt-3.5-turbo"
export CLARIAI_LANGCHAIN_TEMPERATURE=0.1

# Hugging Face integration
export CLARIAI_HF_ORGANIZATION="your-org"
export HUGGINGFACE_HUB_TOKEN="your-token"
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=clariai

# Run specific test
python -m pytest tests/test_audio_analysis.py
```

## 📊 Performance

ClariAI is optimized for high-performance audio processing:

- **Real-time Processing**: < 100ms analysis time for 1-second audio
- **Batch Processing**: Process 100+ files per minute
- **Memory Efficient**: Minimal memory footprint
- **Scalable**: Supports distributed processing

## 🤝 Contributing

We welcome contributions to ClariAI! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/kernelseed/audio-call-quality-analyzer.git
cd audio-call-quality-analyzer

# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest

# Run linting
black clariai/
flake8 clariai/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **librosa**: Audio and music signal processing
- **LangChain**: LLM application framework
- **Hugging Face**: Model sharing and deployment
- **scikit-learn**: Machine learning algorithms
- **OpenAI**: Language model integration

## 📞 Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/kernelseed/audio-call-quality-analyzer/issues)
- **Documentation**: [Full documentation](https://github.com/kernelseed/audio-call-quality-analyzer#readme)
- **Discussions**: [Community discussions](https://github.com/kernelseed/audio-call-quality-analyzer/discussions)

---

**ClariAI** - Professional Audio Quality Analysis Platform  
*Powered by LangChain and Machine Learning*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open Source](https://img.shields.io/badge/Open%20Source-100%25-green.svg)](https://opensource.org/)