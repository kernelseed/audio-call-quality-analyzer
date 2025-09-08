"""
ClariAI Hugging Face Integration

Seamless integration with Hugging Face Hub for model sharing, deployment,
and collaborative development of audio quality analysis models.

GitHub Repository: https://github.com/kernelseed/audio-call-quality-analyzer
"""

import os
import json
import shutil
import tempfile
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

# Hugging Face imports
try:
    from huggingface_hub import HfApi, Repository, create_repo
    from huggingface_hub.utils import RepositoryNotFoundError
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("Hugging Face Hub not available. Install with: pip install huggingface-hub")

# ClariAI imports
from audio_call_quality_model import ClariAIAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClariAIHubUploader:
    """
    ClariAI Hugging Face Hub Integration
    
    Provides seamless integration with Hugging Face Hub for model sharing,
    deployment, and collaborative development.
    """
    
    def __init__(self, 
                 hf_token: Optional[str] = None,
                 organization: Optional[str] = None):
        """
        Initialize ClariAI Hub uploader.
        
        Args:
            hf_token: Hugging Face authentication token
            organization: Hugging Face organization name (optional)
        """
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("Hugging Face Hub not available. Install with: pip install huggingface-hub")
        
        self.hf_token = hf_token or os.getenv('HUGGINGFACE_HUB_TOKEN')
        self.organization = organization
        
        if not self.hf_token:
            raise ValueError("Hugging Face token required. Set HUGGINGFACE_HUB_TOKEN environment variable or pass hf_token parameter.")
        
        # Initialize HF API
        self.api = HfApi(token=self.hf_token)
        
        # Initialize ClariAI analyzer
        self.analyzer = ClariAIAnalyzer()
    
    def create_model_card(self, 
                         model_name: str,
                         model_description: str,
                         model_type: str = "audio-quality-analysis",
                         tags: Optional[List[str]] = None,
                         dataset_info: Optional[Dict] = None,
                         performance_metrics: Optional[Dict] = None) -> str:
        """
        Create a comprehensive model card for Hugging Face Hub.
        
        Args:
            model_name: Name of the model
            model_description: Description of the model
            model_type: Type of model
            tags: List of tags for the model
            dataset_info: Information about training dataset
            performance_metrics: Model performance metrics
            
        Returns:
            Model card content as string
        """
        tags = tags or ["audio", "quality-analysis", "machine-learning", "clariai"]
        
        model_card = f"""---
license: mit
tags:
{chr(10).join([f"  - {tag}" for tag in tags])}
library_name: clariai
pipeline_tag: audio-classification
---

# {model_name}

## Model Description

{model_description}

## Model Type
**{model_type}**

## ClariAI Integration
This model is part of the ClariAI ecosystem for professional audio quality analysis.

## Features
- **Advanced Audio Processing**: Professional-grade feature extraction
- **LangChain Integration**: Intelligent quality analysis
- **Real-time Analysis**: Lightning-fast inference
- **Multiple ML Algorithms**: Support for various classification methods
- **Hugging Face Integration**: Seamless deployment and sharing

## Usage

### Basic Usage
```python
from audio_call_quality_model import ClariAIAnalyzer

# Initialize ClariAI analyzer
analyzer = ClariAIAnalyzer()

# Analyze audio quality
results = analyzer.analyze_call_quality("path/to/audio.wav")

# Print results
print(f"Overall Quality: {{results['quality_scores']['overall_quality']:.3f}}")
print(f"Clarity: {{results['quality_scores']['clarity']:.3f}}")
print(f"Volume: {{results['quality_scores']['volume']:.3f}}")
print(f"Noise Level: {{results['quality_scores']['noise_level']:.3f}}")
```

### Training Custom Model
```python
from training_pipeline import ClariAITrainer

# Initialize trainer
trainer = ClariAITrainer()

# Add training samples
trainer.add_training_sample("audio1.wav", "excellent")
trainer.add_training_sample("audio2.wav", "good")
trainer.add_training_sample("audio3.wav", "poor")

# Train model
metrics = trainer.train_model("random_forest")

# Save model
trainer.save_model("my_quality_model")
```

## Model Performance
"""
        
        if performance_metrics:
            model_card += f"""
### Performance Metrics
- **Accuracy**: {performance_metrics.get('accuracy', 'N/A')}
- **Precision**: {performance_metrics.get('precision', 'N/A')}
- **Recall**: {performance_metrics.get('recall', 'N/A')}
- **F1-Score**: {performance_metrics.get('f1_score', 'N/A')}
"""
        
        if dataset_info:
            model_card += f"""
## Training Dataset
- **Size**: {dataset_info.get('size', 'N/A')} samples
- **Features**: {dataset_info.get('features', 'N/A')}
- **Quality Classes**: {dataset_info.get('classes', 'N/A')}
- **Audio Format**: {dataset_info.get('format', 'WAV')}
"""
        
        model_card += f"""
## Installation

```bash
pip install clariai
```

## Requirements
- Python 3.8+
- librosa
- soundfile
- scikit-learn
- langchain-community
- transformers

## License
MIT License - See LICENSE file for details.

## Citation

```bibtex
@software{{clariai_audio_quality,
  title={{ClariAI: Professional Audio Quality Analysis Platform}},
  author={{ClariAI Team}},
  year={{2024}},
  url={{https://github.com/kernelseed/audio-call-quality-analyzer}}
}}
```

## Contact
- **GitHub**: [https://github.com/kernelseed/audio-call-quality-analyzer](https://github.com/kernelseed/audio-call-quality-analyzer)
- **Issues**: [Report issues](https://github.com/kernelseed/audio-call-quality-analyzer/issues)

---
*Generated by ClariAI Hub Integration*
"""
        
        return model_card
    
    def prepare_model_for_upload(self, 
                                model_path: str,
                                model_name: str,
                                model_description: str,
                                include_examples: bool = True) -> str:
        """
        Prepare model files for Hugging Face Hub upload.
        
        Args:
            model_path: Path to trained model directory
            model_path: Name for the model
            model_description: Description of the model
            include_examples: Whether to include example files
            
        Returns:
            Path to prepared model directory
        """
        try:
            # Create temporary directory for upload
            temp_dir = tempfile.mkdtemp(prefix="clariai_upload_")
            temp_path = Path(temp_dir)
            
            # Copy model files
            model_source = Path(model_path)
            if not model_source.exists():
                raise ValueError(f"Model path does not exist: {model_path}")
            
            # Copy all model files
            for file_path in model_source.iterdir():
                if file_path.is_file():
                    shutil.copy2(file_path, temp_path / file_path.name)
            
            # Create model card
            model_card = self.create_model_card(
                model_name=model_name,
                model_description=model_description
            )
            
            with open(temp_path / "README.md", "w") as f:
                f.write(model_card)
            
            # Create requirements.txt
            requirements = """# ClariAI Model Requirements
clariai>=1.0.0
librosa>=0.10.0
soundfile>=0.12.0
scikit-learn>=1.3.0
langchain-community>=0.0.10
transformers>=4.30.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
"""
            
            with open(temp_path / "requirements.txt", "w") as f:
                f.write(requirements)
            
            # Create example usage script
            if include_examples:
                example_script = f'''"""
Example usage of {model_name}
"""

from audio_call_quality_model import ClariAIAnalyzer
import json

def main():
    # Initialize ClariAI analyzer
    analyzer = ClariAIAnalyzer()
    
    # Example audio file (replace with your audio file)
    audio_file = "example_audio.wav"
    
    # Analyze audio quality
    print(f"Analyzing audio: {{audio_file}}")
    results = analyzer.analyze_call_quality(audio_file)
    
    # Print results
    print("\\n🎵 ClariAI Audio Quality Analysis")
    print("=" * 40)
    print(f"Overall Quality: {{results['quality_scores']['overall_quality']:.3f}}")
    print(f"Clarity: {{results['quality_scores']['clarity']:.3f}}")
    print(f"Volume: {{results['quality_scores']['volume']:.3f}}")
    print(f"Noise Level: {{results['quality_scores']['noise_level']:.3f}}")
    
    # Save detailed results
    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\\nDetailed results saved to analysis_results.json")

if __name__ == "__main__":
    main()
'''
                
                with open(temp_path / "example_usage.py", "w") as f:
                    f.write(example_script)
            
            # Create metadata
            metadata = {
                "model_name": model_name,
                "description": model_description,
                "created_at": datetime.now().isoformat(),
                "clariai_version": "1.0.0",
                "model_type": "audio-quality-analysis",
                "tags": ["audio", "quality-analysis", "machine-learning", "clariai"]
            }
            
            with open(temp_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model prepared for upload in: {temp_dir}")
            return temp_dir
            
        except Exception as e:
            logger.error(f"Error preparing model for upload: {e}")
            raise
    
    def upload_model(self, 
                    model_path: str,
                    repo_name: str,
                    model_description: str,
                    private: bool = False,
                    organization: Optional[str] = None) -> str:
        """
        Upload model to Hugging Face Hub.
        
        Args:
            model_path: Path to prepared model directory
            repo_name: Name for the repository
            model_description: Description of the model
            private: Whether to create private repository
            organization: Organization name (optional)
            
        Returns:
            URL of the uploaded model
        """
        try:
            # Determine repository name
            if organization:
                full_repo_name = f"{organization}/{repo_name}"
            else:
                full_repo_name = repo_name
            
            # Create repository
            try:
                create_repo(
                    repo_id=full_repo_name,
                    token=self.hf_token,
                    private=private,
                    exist_ok=True
                )
                logger.info(f"Repository created: {full_repo_name}")
            except Exception as e:
                logger.warning(f"Repository creation warning: {e}")
            
            # Upload files
            self.api.upload_folder(
                folder_path=model_path,
                repo_id=full_repo_name,
                token=self.hf_token,
                commit_message=f"Upload {repo_name} model"
            )
            
            model_url = f"https://huggingface.co/{full_repo_name}"
            logger.info(f"Model uploaded successfully: {model_url}")
            
            return model_url
            
        except Exception as e:
            logger.error(f"Error uploading model: {e}")
            raise
    
    def download_model(self, 
                      repo_name: str,
                      local_path: str,
                      organization: Optional[str] = None) -> str:
        """
        Download model from Hugging Face Hub.
        
        Args:
            repo_name: Name of the repository
            local_path: Local path to save the model
            organization: Organization name (optional)
            
        Returns:
            Path to downloaded model
        """
        try:
            # Determine repository name
            if organization:
                full_repo_name = f"{organization}/{repo_name}"
            else:
                full_repo_name = repo_name
            
            # Create local directory
            local_path = Path(local_path)
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Download repository
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=full_repo_name,
                local_dir=str(local_path),
                token=self.hf_token
            )
            
            logger.info(f"Model downloaded to: {local_path}")
            return str(local_path)
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise
    
    def list_models(self, 
                   organization: Optional[str] = None,
                   limit: int = 10) -> List[Dict[str, str]]:
        """
        List available models.
        
        Args:
            organization: Organization name (optional)
            limit: Maximum number of models to return
            
        Returns:
            List of model information dictionaries
        """
        try:
            # Get models from organization or user
            if organization:
                models = self.api.list_models(
                    author=organization,
                    limit=limit
                )
            else:
                models = self.api.list_models(
                    limit=limit
                )
            
            model_list = []
            for model in models:
                model_list.append({
                    'name': model.id,
                    'author': model.author,
                    'created_at': model.created_at,
                    'downloads': model.downloads,
                    'tags': model.tags,
                    'url': f"https://huggingface.co/{model.id}"
                })
            
            return model_list
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def search_models(self, 
                     query: str,
                     limit: int = 10) -> List[Dict[str, str]]:
        """
        Search for models by query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching model information
        """
        try:
            models = self.api.list_models(
                search=query,
                limit=limit
            )
            
            model_list = []
            for model in models:
                model_list.append({
                    'name': model.id,
                    'author': model.author,
                    'created_at': model.created_at,
                    'downloads': model.downloads,
                    'tags': model.tags,
                    'url': f"https://huggingface.co/{model.id}"
                })
            
            return model_list
            
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            return []

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ClariAI Hugging Face Integration")
    parser.add_argument("--action", choices=["upload", "download", "list", "search"], required=True, help="Action to perform")
    parser.add_argument("--model-path", help="Path to model directory")
    parser.add_argument("--repo-name", help="Repository name")
    parser.add_argument("--description", help="Model description")
    parser.add_argument("--local-path", help="Local path for download")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--organization", help="Organization name")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    
    args = parser.parse_args()
    
    # Initialize uploader
    uploader = ClariAIHubUploader()
    
    if args.action == "upload":
        if not all([args.model_path, args.repo_name, args.description]):
            print("Error: --model-path, --repo-name, and --description are required for upload")
            return
        
        # Prepare model
        prepared_path = uploader.prepare_model_for_upload(
            model_path=args.model_path,
            model_name=args.repo_name,
            model_description=args.description
        )
        
        # Upload model
        model_url = uploader.upload_model(
            model_path=prepared_path,
            repo_name=args.repo_name,
            model_description=args.description,
            private=args.private,
            organization=args.organization
        )
        
        print(f"Model uploaded successfully: {model_url}")
    
    elif args.action == "download":
        if not all([args.repo_name, args.local_path]):
            print("Error: --repo-name and --local-path are required for download")
            return
        
        local_path = uploader.download_model(
            repo_name=args.repo_name,
            local_path=args.local_path,
            organization=args.organization
        )
        
        print(f"Model downloaded to: {local_path}")
    
    elif args.action == "list":
        models = uploader.list_models(organization=args.organization)
        print(f"Found {len(models)} models:")
        for model in models:
            print(f"  - {model['name']} by {model['author']}")
    
    elif args.action == "search":
        if not args.query:
            print("Error: --query is required for search")
            return
        
        models = uploader.search_models(query=args.query)
        print(f"Found {len(models)} models matching '{args.query}':")
        for model in models:
            print(f"  - {model['name']} by {model['author']}")

if __name__ == "__main__":
    main()