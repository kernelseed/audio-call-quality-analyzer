"""
ClariAI Test Audio Generator
This script creates a 1-minute WAV file with AI-generated speech content for ClariAI testing.
"""

import numpy as np
import soundfile as sf
import librosa
from scipy import signal
import os
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")


class AudioGenerator:
    """Generate realistic speech-like audio for testing."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio generator.
        
        Args:
            sample_rate: Sample rate for generated audio
        """
        self.sample_rate = sample_rate
        self.duration = 60  # 1 minute
        
    def generate_speech_like_audio(self, text: str) -> np.ndarray:
        """
        Generate speech-like audio from text using formant synthesis.
        
        Args:
            text: Text to convert to speech-like audio
            
        Returns:
            Generated audio array
        """
        # Create time array
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), False)
        
        # Initialize audio array
        audio = np.zeros_like(t)
        
        # Split text into words for timing
        words = text.split()
        word_duration = self.duration / len(words)
        
        # Generate speech-like signal for each word
        for i, word in enumerate(words):
            start_time = i * word_duration
            end_time = (i + 1) * word_duration
            
            # Convert time to samples
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Generate formant frequencies based on word characteristics
            word_audio = self._generate_word_audio(word, word_duration)
            
            # Add to main audio
            if end_sample <= len(audio):
                # Ensure word_audio fits in the target slice
                target_length = end_sample - start_sample
                if len(word_audio) > target_length:
                    word_audio = word_audio[:target_length]
                elif len(word_audio) < target_length:
                    # Pad with zeros if needed
                    padding = target_length - len(word_audio)
                    word_audio = np.pad(word_audio, (0, padding), mode='constant')
                
                audio[start_sample:start_sample + len(word_audio)] = word_audio
        
        # Add natural pauses and breathing
        audio = self._add_natural_pauses(audio)
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio
    
    def _generate_word_audio(self, word: str, duration: float) -> np.ndarray:
        """Generate audio for a single word using formant synthesis."""
        # Create time array for this word
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Base fundamental frequency (pitch)
        f0 = 120 + np.random.normal(0, 20)  # Hz
        
        # Generate fundamental frequency with some variation
        f0_variation = 1 + 0.1 * np.sin(2 * np.pi * 3 * t)  # Slow pitch variation
        f0_actual = f0 * f0_variation
        
        # Generate harmonics
        audio = np.zeros_like(t)
        
        # Add fundamental and first few harmonics
        for harmonic in range(1, 6):
            freq = f0_actual * harmonic
            amplitude = 1.0 / harmonic  # Decreasing amplitude for harmonics
            audio += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Add formant filtering (simulate vocal tract)
        audio = self._apply_formant_filtering(audio, word)
        
        # Add envelope (attack and decay)
        envelope = self._generate_envelope(len(t))
        audio *= envelope
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.05, len(audio))
        audio += noise
        
        return audio
    
    def _apply_formant_filtering(self, audio: np.ndarray, word: str) -> np.ndarray:
        """Apply formant filtering to simulate vocal tract."""
        # Different formant frequencies for different vowel sounds
        vowel_formants = {
            'a': (800, 1200, 2500),
            'e': (500, 2000, 2800),
            'i': (300, 2200, 3000),
            'o': (500, 1000, 2500),
            'u': (300, 800, 2200),
        }
        
        # Find dominant vowel in word
        vowels = [v for v in word.lower() if v in vowel_formants]
        if vowels:
            formants = vowel_formants[vowels[0]]
        else:
            formants = (800, 1200, 2500)  # Default formants
        
        # Apply formant filtering
        filtered_audio = audio.copy()
        for formant_freq in formants:
            # Simple formant filter - use lowpass instead of bandpass
            nyquist = self.sample_rate / 2
            normalized_freq = formant_freq / nyquist
            if normalized_freq < 1.0:  # Ensure frequency is within valid range
                b, a = signal.butter(2, normalized_freq, btype='low')
                filtered_audio += 0.3 * signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def _generate_envelope(self, length: int) -> np.ndarray:
        """Generate attack-decay envelope for natural speech."""
        envelope = np.ones(length)
        
        # Attack (first 10% of duration)
        attack_length = int(0.1 * length)
        envelope[:attack_length] = np.linspace(0, 1, attack_length)
        
        # Decay (last 20% of duration)
        decay_length = int(0.2 * length)
        envelope[-decay_length:] = np.linspace(1, 0, decay_length)
        
        return envelope
    
    def _add_natural_pauses(self, audio: np.ndarray) -> np.ndarray:
        """Add natural pauses and breathing sounds."""
        # Add pauses between sentences
        pause_length = int(0.3 * self.sample_rate)  # 0.3 second pause
        
        # Find sentence boundaries (approximate)
        sentence_boundaries = [int(len(audio) * 0.25), int(len(audio) * 0.5), int(len(audio) * 0.75)]
        
        for boundary in sentence_boundaries:
            if boundary + pause_length < len(audio):
                # Add silence
                audio[boundary:boundary + pause_length] *= 0.1
        
        # Add breathing sounds (low frequency noise)
        breathing_points = [int(len(audio) * 0.2), int(len(audio) * 0.6)]
        for point in breathing_points:
            if point + int(0.5 * self.sample_rate) < len(audio):
                breath_length = int(0.5 * self.sample_rate)
                breath = np.random.normal(0, 0.1, breath_length)
                # Low-pass filter for breathing sound
                b, a = signal.butter(4, 200 / (self.sample_rate / 2), btype='low')
                breath = signal.filtfilt(b, a, breath)
                audio[point:point + breath_length] += breath * 0.3
        
        return audio


def create_ai_future_speech() -> str:
    """Create the speech content about AI and the future."""
    return """
    Artificial Intelligence represents one of the most transformative technologies of our time. 
    As we look toward the future, AI promises to revolutionize every aspect of human life. 
    From healthcare and education to transportation and communication, intelligent systems 
    are becoming increasingly sophisticated and capable. Machine learning algorithms can now 
    process vast amounts of data, recognize patterns, and make predictions with remarkable accuracy. 
    Deep learning networks mimic the human brain's neural pathways, enabling computers to see, 
    hear, and understand the world around us. The future of AI holds incredible potential for 
    solving complex global challenges, from climate change to disease prevention. However, 
    we must approach this future with careful consideration of ethics, privacy, and the 
    responsible development of these powerful technologies. The key is to ensure that AI 
    serves humanity's best interests while maintaining human agency and control over these 
    increasingly intelligent systems.
    """


def generate_test_audio(filename: str = "ai_future_speech.wav") -> str:
    """
    Generate a 1-minute test audio file about AI and the future.
    
    Args:
        filename: Output filename for the WAV file
        
    Returns:
        Path to the generated audio file
    """
    print("🎵 Generating 1-minute AI and Future speech audio...")
    
    # Create audio generator
    generator = AudioGenerator(sample_rate=16000)
    
    # Get speech content
    speech_text = create_ai_future_speech()
    print(f"📝 Speech content: {len(speech_text.split())} words")
    
    # Generate audio
    print("🎤 Synthesizing speech-like audio...")
    audio = generator.generate_speech_like_audio(speech_text)
    
    # Save to file
    sf.write(filename, audio, generator.sample_rate)
    print(f"✅ Audio saved to: {filename}")
    print(f"   Duration: {len(audio) / generator.sample_rate:.1f} seconds")
    print(f"   Sample Rate: {generator.sample_rate} Hz")
    print(f"   File Size: {os.path.getsize(filename) / 1024:.1f} KB")
    
    return filename


def test_audio_quality(audio_file: str):
    """Test the generated audio with our quality analysis model."""
    print(f"\n🔍 Testing audio quality for: {audio_file}")
    
    try:
        from audio_call_quality_model import ClariAIAnalyzer
        
        # Initialize ClariAI analyzer
        analyzer = ClariAIAnalyzer()
        
        # Analyze the audio
        results = analyzer.analyze_call_quality(audio_file)
        
        # Display results
        print("\n📊 Audio Quality Analysis Results:")
        print("=" * 50)
        print(f"File: {results['audio_path']}")
        print(f"Duration: {results['features']['duration']:.2f} seconds")
        print(f"Sample Rate: {results['features']['sample_rate']} Hz")
        
        print(f"\n🎯 Quality Scores:")
        for metric, score in results['quality_scores'].items():
            print(f"  {metric.capitalize()}: {score:.3f}")
        
        # Overall assessment
        overall = results['quality_scores']['overall_quality']
        if overall >= 0.8:
            assessment = "Excellent"
            emoji = "🌟"
        elif overall >= 0.6:
            assessment = "Good"
            emoji = "👍"
        elif overall >= 0.4:
            assessment = "Fair"
            emoji = "⚠️"
        else:
            assessment = "Poor"
            emoji = "❌"
        
        print(f"\n{emoji} Overall Assessment: {assessment} ({overall:.3f})")
        
        # Feature details
        print(f"\n🔧 Audio Features:")
        features = results['features']
        print(f"  Spectral Centroid: {features['spectral_centroid']:.1f} Hz")
        print(f"  Zero Crossing Rate: {features['zcr']:.3f}")
        print(f"  RMS Energy: {features['rms_energy']:.3f}")
        print(f"  SNR: {features['snr']:.3f}")
        print(f"  Voice Activity Ratio: {features['voice_activity_ratio']:.3f}")
        print(f"  Dynamic Range: {features['dynamic_range']:.3f}")
        
        # Show transcript if available
        if results['transcript'] and results['transcript'] != "Transcription failed":
            print(f"\n📝 Transcript Preview: {results['transcript'][:100]}...")
        
        return results
        
    except ImportError:
        print("⚠️  Audio quality analyzer not available. Install required dependencies.")
        return None
    except Exception as e:
        print(f"❌ Error analyzing audio: {e}")
        return None


def main():
    """Main function to generate and test audio for ClariAI."""
    print("🤖 ClariAI Test Audio Generator")
    print("=" * 50)
    
    # Generate test audio
    audio_file = generate_test_audio()
    
    # Test with quality analyzer
    results = test_audio_quality(audio_file)
    
    if results:
        print(f"\n🎉 Audio generation and testing completed successfully!")
        print(f"📁 Generated file: {audio_file}")
        print(f"📊 Quality score: {results['quality_scores']['overall_quality']:.3f}")
    else:
        print(f"\n✅ Audio generation completed!")
        print(f"📁 Generated file: {audio_file}")
        print("ℹ️  Run 'python example_usage.py' to test with the full quality analysis system.")


if __name__ == "__main__":
    main()
