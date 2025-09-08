"""
ClariAI - Professional Audio Quality Analysis Platform

A comprehensive audio quality analysis system using LangChain and machine learning
for real-time call quality monitoring and intelligent classification.

GitHub Repository: https://github.com/kernelseed/audio-call-quality-analyzer
"""

import os
import warnings
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

# LangChain imports
try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.prompts import PromptTemplate
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    warnings.warn("LangChain not available. Install with: pip install langchain-community")

# Optional audio processing imports
try:
    import webrtcvad
    import speech_recognition as sr
    from pydub import AudioSegment
    ADVANCED_AUDIO_AVAILABLE = True
except ImportError:
    ADVANCED_AUDIO_AVAILABLE = False
    warnings.warn("Advanced audio features not available. Install with: pip install webrtcvad speechrecognition pydub")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityMetrics(BaseModel):
    """Pydantic model for quality metrics output."""
    clarity: float = Field(description="Speech intelligibility score (0-1)")
    volume: float = Field(description="Audio volume level score (0-1)")
    noise_level: float = Field(description="Background noise level (0-1, lower is better)")
    overall_quality: float = Field(description="Overall audio quality score (0-1)")

class ClariAIAnalyzer:
    """
    ClariAI Professional Audio Quality Analyzer
    
    A comprehensive audio quality analysis system that provides intelligent
    assessment of call quality using advanced signal processing and LangChain.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 hop_length: int = 512,
                 n_mfcc: int = 13,
                 vad_aggressiveness: int = 3):
        """
        Initialize ClariAI analyzer.
        
        Args:
            sample_rate: Target sample rate for audio processing
            hop_length: Hop length for STFT computation
            n_mfcc: Number of MFCC coefficients to extract
            vad_aggressiveness: VAD aggressiveness level (0-3)
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.vad_aggressiveness = vad_aggressiveness
        
        # Initialize VAD if available
        self.vad = None
        if ADVANCED_AUDIO_AVAILABLE:
            try:
                self.vad = webrtcvad.Vad(vad_aggressiveness)
            except Exception as e:
                logger.warning(f"VAD initialization failed: {e}")
        
        # Initialize LangChain components
        self.llm = None
        self.parser = None
        if LANGCHAIN_AVAILABLE:
            self._setup_langchain()
    
    def _setup_langchain(self):
        """Setup LangChain components for intelligent analysis."""
        try:
            # Use a simple model for analysis
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.1,
                max_tokens=200
            )
            
            # Setup output parser
            self.parser = PydanticOutputParser(pydantic_object=QualityMetrics)
            
            # Create analysis prompt
            self.analysis_prompt = PromptTemplate(
                template="""
                Analyze the following audio quality metrics and provide a professional assessment:
                
                Spectral Centroid: {spectral_centroid:.3f}
                Zero Crossing Rate: {zcr:.3f}
                RMS Energy: {rms_energy:.3f}
                MFCC Mean: {mfcc_mean:.3f}
                Signal-to-Noise Ratio: {snr:.3f}
                Voice Activity Ratio: {voice_activity_ratio:.3f}
                Spectral Rolloff: {spectral_rolloff:.3f}
                Spectral Bandwidth: {spectral_bandwidth:.3f}
                Dynamic Range: {dynamic_range:.3f}
                Crest Factor: {crest_factor:.3f}
                
                Provide quality scores (0-1 scale) for:
                1. Clarity: Speech intelligibility and articulation
                2. Volume: Appropriate audio level
                3. Noise Level: Background noise (lower is better)
                4. Overall Quality: Comprehensive assessment
                
                {format_instructions}
                """,
                input_variables=[
                    "spectral_centroid", "zcr", "rms_energy", "mfcc_mean", "snr",
                    "voice_activity_ratio", "spectral_rolloff", "spectral_bandwidth",
                    "dynamic_range", "crest_factor"
                ],
                partial_variables={"format_instructions": self.parser.get_format_instructions()}
            )
            
        except Exception as e:
            logger.warning(f"LangChain setup failed: {e}")
            self.llm = None
            self.parser = None
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file with proper preprocessing.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio file
            audio_data, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = librosa.to_mono(audio_data)
            
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            return audio_data, sr
            
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
            raise
    
    def extract_audio_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract comprehensive audio features for quality analysis.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Load audio
            audio_data, sr = self.load_audio(audio_path)
            
            # Basic spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr))
            
            # Temporal features
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))
            rms_energy = np.mean(librosa.feature.rms(y=audio_data))
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfccs)
            mfcc_std = np.std(mfccs)
            
            # Voice Activity Detection
            voice_activity_ratio = self._detect_voice_activity(audio_data, sr)
            
            # Signal-to-Noise Ratio estimation
            snr = self._estimate_snr(audio_data)
            
            # Dynamic range and crest factor
            dynamic_range = np.max(audio_data) - np.min(audio_data)
            crest_factor = np.max(np.abs(audio_data)) / (rms_energy + 1e-8)
            
            features = {
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': float(spectral_rolloff),
                'spectral_bandwidth': float(spectral_bandwidth),
                'zcr': float(zcr),
                'rms_energy': float(rms_energy),
                'mfcc_mean': float(mfcc_mean),
                'mfcc_std': float(mfcc_std),
                'voice_activity_ratio': float(voice_activity_ratio),
                'snr': float(snr),
                'dynamic_range': float(dynamic_range),
                'crest_factor': float(crest_factor)
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            raise
    
    def _detect_voice_activity(self, audio_data: np.ndarray, sr: int) -> float:
        """Detect voice activity ratio using WebRTC VAD."""
        if not ADVANCED_AUDIO_AVAILABLE or self.vad is None:
            # Fallback: simple energy-based VAD
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            
            frames = []
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i + frame_length]
                frames.append(frame)
            
            # Simple energy threshold
            energy_threshold = 0.01
            voice_frames = sum(1 for frame in frames if np.mean(frame**2) > energy_threshold)
            return voice_frames / len(frames) if frames else 0.0
        
        try:
            # Convert to 16-bit PCM for VAD
            audio_16bit = (audio_data * 32767).astype(np.int16)
            
            # Process in 10ms frames
            frame_length = int(0.010 * sr)
            voice_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_16bit) - frame_length, frame_length):
                frame = audio_16bit[i:i + frame_length]
                if len(frame) == frame_length:
                    is_speech = self.vad.is_speech(frame.tobytes(), sr)
                    if is_speech:
                        voice_frames += 1
                    total_frames += 1
            
            return voice_frames / total_frames if total_frames > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"VAD processing failed: {e}")
            return 0.5  # Default neutral value
    
    def _estimate_snr(self, audio_data: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio."""
        try:
            # Simple SNR estimation using spectral subtraction approach
            # This is a simplified version - in practice, more sophisticated methods would be used
            
            # Estimate noise from quiet segments
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            frames = [audio_data[i:i + frame_length] for i in range(0, len(audio_data) - frame_length, frame_length)]
            
            if not frames:
                return 0.0
            
            # Find quietest 20% of frames as noise estimate
            frame_energies = [np.mean(frame**2) for frame in frames]
            noise_frames = int(0.2 * len(frames))
            quietest_frames = sorted(range(len(frame_energies)), key=lambda i: frame_energies[i])[:noise_frames]
            
            noise_energy = np.mean([frame_energies[i] for i in quietest_frames])
            signal_energy = np.mean(frame_energies)
            
            if noise_energy > 0:
                snr_db = 10 * np.log10(signal_energy / noise_energy)
                # Convert to 0-1 scale (assuming reasonable SNR range of -20 to 40 dB)
                snr_normalized = np.clip((snr_db + 20) / 60, 0, 1)
                return snr_normalized
            else:
                return 1.0  # Perfect SNR if no noise detected
                
        except Exception as e:
            logger.warning(f"SNR estimation failed: {e}")
            return 0.5  # Default neutral value
    
    def analyze_call_quality(self, audio_path: str) -> Dict[str, Union[Dict[str, float], str, float]]:
        """
        Analyze call quality using ClariAI's intelligent assessment.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing quality analysis results
        """
        try:
            # Extract audio features
            features = self.extract_audio_features(audio_path)
            
            # Get intelligent analysis if LangChain is available
            if self.llm and self.parser:
                try:
                    # Create prompt with features
                    prompt = self.analysis_prompt.format(**features)
                    
                    # Get LLM analysis
                    messages = [HumanMessage(content=prompt)]
                    response = self.llm(messages)
                    
                    # Parse response
                    quality_metrics = self.parser.parse(response.content)
                    
                    return {
                        'quality_scores': {
                            'clarity': quality_metrics.clarity,
                            'volume': quality_metrics.volume,
                            'noise_level': quality_metrics.noise_level,
                            'overall_quality': quality_metrics.overall_quality
                        },
                        'features': features,
                        'analysis_method': 'langchain_intelligent',
                        'processing_time': 0.0  # Would be measured in practice
                    }
                    
                except Exception as e:
                    logger.warning(f"LangChain analysis failed: {e}")
                    # Fall back to rule-based analysis
            
            # Fallback: Rule-based quality assessment
            quality_scores = self._rule_based_analysis(features)
            
            return {
                'quality_scores': quality_scores,
                'features': features,
                'analysis_method': 'rule_based',
                'processing_time': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audio quality: {e}")
            return {
                'quality_scores': {
                    'clarity': 0.0,
                    'volume': 0.0,
                    'noise_level': 1.0,
                    'overall_quality': 0.0
                },
                'features': {},
                'analysis_method': 'error',
                'error': str(e)
            }
    
    def _rule_based_analysis(self, features: Dict[str, float]) -> Dict[str, float]:
        """Fallback rule-based quality analysis."""
        try:
            # Clarity assessment based on spectral features
            clarity = min(1.0, max(0.0, 
                (features.get('spectral_centroid', 0) / 4000) * 0.3 +  # Higher centroid = clearer
                (1 - features.get('zcr', 0)) * 0.3 +  # Lower ZCR = clearer
                features.get('voice_activity_ratio', 0) * 0.4  # More voice activity = clearer
            ))
            
            # Volume assessment based on RMS energy
            rms_energy = features.get('rms_energy', 0)
            if rms_energy < 0.01:
                volume = 0.0  # Too quiet
            elif rms_energy > 0.3:
                volume = 0.5  # Too loud
            else:
                volume = min(1.0, rms_energy * 2)  # Optimal range
            
            # Noise level assessment (inverted SNR)
            snr = features.get('snr', 0.5)
            noise_level = 1 - snr  # Lower SNR = higher noise
            
            # Overall quality (weighted combination)
            overall_quality = (
                clarity * 0.4 +
                volume * 0.2 +
                (1 - noise_level) * 0.4
            )
            
            return {
                'clarity': float(clarity),
                'volume': float(volume),
                'noise_level': float(noise_level),
                'overall_quality': float(overall_quality)
            }
            
        except Exception as e:
            logger.error(f"Rule-based analysis failed: {e}")
            return {
                'clarity': 0.5,
                'volume': 0.5,
                'noise_level': 0.5,
                'overall_quality': 0.5
            }
    
    def batch_analyze(self, audio_paths: List[str]) -> List[Dict[str, Union[Dict[str, float], str, float]]]:
        """
        Analyze multiple audio files in batch.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            List of analysis results
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.analyze_call_quality(audio_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")
                results.append({
                    'quality_scores': {
                        'clarity': 0.0,
                        'volume': 0.0,
                        'noise_level': 1.0,
                        'overall_quality': 0.0
                    },
                    'features': {},
                    'analysis_method': 'error',
                    'error': str(e)
                })
        
        return results

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ClariAI Audio Quality Analysis")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize analyzer
    analyzer = ClariAIAnalyzer()
    
    # Analyze audio
    print(f"Analyzing audio file: {args.audio_file}")
    results = analyzer.analyze_call_quality(args.audio_file)
    
    # Print results
    print("\n🎵 ClariAI Audio Quality Analysis Results")
    print("=" * 50)
    print(f"Analysis Method: {results['analysis_method']}")
    print(f"Overall Quality: {results['quality_scores']['overall_quality']:.3f}")
    print(f"Clarity: {results['quality_scores']['clarity']:.3f}")
    print(f"Volume: {results['quality_scores']['volume']:.3f}")
    print(f"Noise Level: {results['quality_scores']['noise_level']:.3f}")
    
    # Save results if output file specified
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()