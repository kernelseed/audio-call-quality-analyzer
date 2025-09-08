"""
ClariAI Configuration Management

Centralized configuration management for ClariAI audio quality analysis platform.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    hop_length: int = 512
    n_mfcc: int = 13
    vad_aggressiveness: int = 3
    max_audio_duration: float = 300.0
    min_audio_duration: float = 0.1

@dataclass
class ModelConfig:
    """Machine learning model configuration."""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    max_iterations: int = 1000
    early_stopping: bool = True

@dataclass
class LangChainConfig:
    """LangChain integration configuration."""
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 200
    timeout: int = 30

@dataclass
class ClariAIConfig:
    """Main ClariAI configuration."""
    audio: AudioConfig = AudioConfig()
    model: ModelConfig = ModelConfig()
    langchain: LangChainConfig = LangChainConfig()
    
    # File paths
    data_dir: str = "data"
    models_dir: str = "models"
    temp_dir: str = "temp"
    
    # Feature extraction
    feature_names: list = None
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.feature_names is None:
            self.feature_names = [
                'spectral_centroid', 'zcr', 'rms_energy', 'mfcc_mean', 
                'snr', 'voice_activity_ratio', 'spectral_rolloff', 
                'spectral_bandwidth', 'mfcc_std', 'dynamic_range', 'crest_factor'
            ]

class ConfigManager:
    """ClariAI Configuration Manager"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "clariai_config.json"
        self.config = ClariAIConfig()
        self.load_config()
        self.apply_env_overrides()
    
    def load_config(self, config_file: Optional[str] = None):
        """Load configuration from file."""
        config_file = config_file or self.config_file
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                self._update_config_from_dict(config_data)
                logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                logger.warning(f"Error loading configuration: {e}")
        else:
            logger.info(f"Configuration file {config_file} not found, using defaults")
    
    def save_config(self, config_file: Optional[str] = None):
        """Save configuration to file."""
        config_file = config_file or self.config_file
        
        try:
            config_dict = asdict(self.config)
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        try:
            if 'audio' in config_dict:
                for key, value in config_dict['audio'].items():
                    if hasattr(self.config.audio, key):
                        setattr(self.config.audio, key, value)
            
            if 'model' in config_dict:
                for key, value in config_dict['model'].items():
                    if hasattr(self.config.model, key):
                        setattr(self.config.model, key, value)
            
            if 'langchain' in config_dict:
                for key, value in config_dict['langchain'].items():
                    if hasattr(self.config.langchain, key):
                        setattr(self.config.langchain, key, value)
                        
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            raise
    
    def apply_env_overrides(self):
        """Apply environment variable overrides."""
        try:
            if os.getenv('CLARIAI_SAMPLE_RATE'):
                self.config.audio.sample_rate = int(os.getenv('CLARIAI_SAMPLE_RATE'))
            
            if os.getenv('CLARIAI_HOP_LENGTH'):
                self.config.audio.hop_length = int(os.getenv('CLARIAI_HOP_LENGTH'))
            
            if os.getenv('CLARIAI_TEST_SIZE'):
                self.config.model.test_size = float(os.getenv('CLARIAI_TEST_SIZE'))
            
            if os.getenv('CLARIAI_LANGCHAIN_MODEL'):
                self.config.langchain.model_name = os.getenv('CLARIAI_LANGCHAIN_MODEL')
            
            logger.info("Environment variable overrides applied")
        except Exception as e:
            logger.warning(f"Error applying environment overrides: {e}")
    
    def get_config(self) -> ClariAIConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        try:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    logger.warning(f"Unknown configuration parameter: {key}")
            logger.info("Configuration updated")
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            raise

# Global configuration manager instance
config_manager = ConfigManager()

def get_config() -> ClariAIConfig:
    """Get global configuration instance."""
    return config_manager.get_config()

def update_config(**kwargs):
    """Update global configuration."""
    config_manager.update_config(**kwargs)

def main():
    """Main function for command-line configuration management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ClariAI Configuration Manager")
    parser.add_argument("--action", choices=["show", "save"], default="show")
    parser.add_argument("--config-file", help="Configuration file path")
    
    args = parser.parse_args()
    
    config_mgr = ConfigManager(args.config_file)
    
    if args.action == "show":
        config = config_mgr.get_config()
        print("ClariAI Configuration:")
        print("=" * 30)
        print(f"Sample Rate: {config.audio.sample_rate}")
        print(f"Hop Length: {config.audio.hop_length}")
        print(f"MFCC Coefficients: {config.audio.n_mfcc}")
        print(f"Test Size: {config.model.test_size}")
        print(f"LangChain Model: {config.langchain.model_name}")
        print(f"Data Directory: {config.data_dir}")
        print(f"Models Directory: {config.models_dir}")
    
    elif args.action == "save":
        config_mgr.save_config()
        print("Configuration saved successfully")

if __name__ == "__main__":
    main()