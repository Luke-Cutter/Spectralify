"""
Spectralify.py - Audio Analysis Tool
-----------------------------------
A modular tool for extracting audio features from music files.
"""

# Standard library imports
import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import gc
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings
from typing import Optional
import time

# Audio processing imports
import librosa
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from mutagen.aac import AAC
from mutagen.aiff import AIFF

# Data handling imports
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('spectralify')

# ============================================================================
# Configuration and Constants
# ============================================================================

class AudioFormats:
    """Audio format configurations and constants"""
    
    FORMATS = {
        '.flac': {'mime_type': 'audio/flac', 'parser': FLAC},
        '.mp3': {'mime_type': 'audio/mp3', 'parser': MP3},
        '.wav': {'mime_type': 'audio/wav', 'parser': WAVE},
        '.aac': {'mime_type': 'audio/aac', 'parser': AAC},
        '.aiff': {'mime_type': 'audio/aiff', 'parser': AIFF},
        '.wma': {'mime_type': 'audio/wma', 'parser': None}
    }
    
    @classmethod
    def is_supported(cls, file_extension):
        """Check if file extension is supported"""
        return file_extension.lower() in cls.FORMATS
    
    @classmethod
    def get_parser(cls, file_extension):
        """Get parser for file extension"""
        return cls.FORMATS.get(file_extension.lower(), {}).get('parser')
    
    @classmethod
    def get_mime_type(cls, file_extension):
        """Get MIME type for file extension"""
        return cls.FORMATS.get(file_extension.lower(), {}).get('mime_type', 'audio/unknown')


class MetadataTags:
    """Constants for metadata tag mapping"""
    
    # Common tag mapping for audio formats
    COMMON_TAGS = {
        'title': 'Title',
        'artist': 'Artist',
        'album': 'Album',
        # Common variations
        'TITLE': 'Title',
        'ARTIST': 'Artist',
        'ALBUM': 'Album'
    }
    
    # ID3-specific tag mapping
    ID3_TAGS = {
        'TIT2': 'Title',     # Title/songname/content description
        'TPE1': 'Artist',    # Lead performer(s)/Soloist(s)
        'TALB': 'Album',     # Album/Movie/Show title
        # Alternate tag names for backwards compatibility
        'TT2': 'Title',      # ID3v2.2 equivalent of TIT2
        'TP1': 'Artist',     # ID3v2.2 equivalent of TPE1
        'TAL': 'Album',      # ID3v2.2 equivalent of TALB
    }


# ============================================================================
# Utility Classes
# ============================================================================

class ProgressTracker:
    """Track and display progress for long-running operations"""
    
    def __init__(self, total: int, description: str = "Processing", bar_length: int = 50):
        self.total = total
        self.current = 0
        self.start_time = time.time()
        self.description = description
        self.bar_length = bar_length
        
    def update(self, amount: int = 1) -> None:
        """Update progress and display the progress bar"""
        self.current += amount
        self._display_progress()
        
    def _format_time(self, seconds: float) -> str:
        """Convert seconds to a readable format"""
        return str(timedelta(seconds=int(seconds)))
        
    def _calculate_eta(self) -> Optional[float]:
        """Calculate estimated time remaining"""
        if self.current == 0:
            return None
        
        elapsed_time = time.time() - self.start_time
        items_per_second = self.current / elapsed_time
        remaining_items = self.total - self.current
        
        return remaining_items / items_per_second if items_per_second > 0 else None
        
    def _display_progress(self) -> None:
        """Display progress bar with time estimates"""
        percentage = min(100, (self.current / self.total) * 100)
        filled_length = int(self.bar_length * self.current // self.total)
        
        # Create the progress bar
        bar = '█' * filled_length + '░' * (self.bar_length - filled_length)
        
        # Calculate time metrics
        elapsed_time = time.time() - self.start_time
        eta = self._calculate_eta()
        
        # Format the progress message
        progress_msg = (
            f'\r{self.description}: |{bar}| '
            f'{percentage:>6.2f}% ({self.current}/{self.total}) '
            f'[{self._format_time(elapsed_time)} elapsed'
        )
        
        if eta is not None:
            progress_msg += f' | ETA: {self._format_time(eta)}]'
        else:
            progress_msg += ']'
            
        # Print the progress
        print(progress_msg, end='', flush=True)
        
        # Print newline if complete
        if self.current >= self.total:
            print(f"\nCompleted in {self._format_time(elapsed_time)}")


class ResourceManager:
    """Context manager for resource-intensive operations"""
    
    def __init__(self, name):
        self.name = name
        
    def __enter__(self):
        logger.debug(f"Starting resource-intensive operation: {self.name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.debug(f"Completed operation: {self.name}, cleaning up resources")
        gc.collect()
        return False


# ============================================================================
# Metadata Extraction
# ============================================================================

class MetadataExtractor:
    """Base class for metadata extraction"""
    
    @classmethod
    def create_extractor(cls, file_path):
        """Factory method to create appropriate metadata extractor"""
        ext = os.path.splitext(file_path)[1].lower()
        parser_class = AudioFormats.get_parser(ext)
        
        if parser_class == MP3:
            return MP3MetadataExtractor()
        else:
            return DefaultMetadataExtractor()
    
    def extract(self, file_path):
        """Extract metadata from file"""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            parser_class = AudioFormats.get_parser(ext)
            
            if parser_class:
                audio_meta = parser_class(file_path)
                metadata = self._extract_from_meta(audio_meta)
                
                # Fill in missing values with basic metadata
                for key in ['Title', 'Artist', 'Album']:
                    if not metadata.get(key):
                        basic_metadata = self._extract_from_path(file_path)
                        metadata[key] = basic_metadata.get(key, f'Unknown {key}')
                
                return metadata
            else:
                return self._extract_from_path(file_path)
                
        except Exception as e:
            logger.error(f"Metadata extraction error for {file_path}: {str(e)}")
            return self._extract_from_path(file_path)
    
    def _extract_from_meta(self, audio_meta):
        """Extract metadata from audio metadata object"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def _extract_from_path(self, file_path):
        """Extract basic metadata from file path"""
        metadata = {
            'Title': 'Unknown Title',
            'Album': 'Unknown Album',
            'Artist': 'Unknown Artist'
        }
        
        if file_path:
            try:
                path = Path(file_path)
                
                # Get title from filename
                metadata['Title'] = path.stem
                
                # Get artist and album from directory structure
                parts = list(path.parts)
                if len(parts) > 2:
                    # Look for year pattern in album folder name
                    album_dir = parts[-2]
                    if '[' in album_dir and ']' in album_dir:
                        # Extract album name without year
                        metadata['Album'] = album_dir.split(']')[-1].strip()
                    else:
                        metadata['Album'] = album_dir
                    
                    metadata['Artist'] = parts[-3]
                elif len(parts) > 1:
                    metadata['Album'] = parts[-2]
                
                # Clean up values
                for key in metadata:
                    if metadata[key]:
                        # Remove file extensions, underscores, excessive spaces
                        cleaned = metadata[key].replace('_', ' ').strip()
                        # Remove common file prefixes/numbers
                        if key == 'Title':
                            cleaned = ' '.join(cleaned.split()[1:]) if cleaned.split() and cleaned.split()[0].isdigit() else cleaned
                        metadata[key] = cleaned
                
            except Exception as e:
                logger.error(f"Error parsing file path metadata: {str(e)}")
        
        return metadata


class MP3MetadataExtractor(MetadataExtractor):
    """Extract metadata from MP3 files"""
    
    def _extract_from_meta(self, audio_meta):
        """Extract metadata specifically from MP3 files"""
        metadata = {
            'Title': '',
            'Artist': '',
            'Album': '',
        }
        
        try:
            if hasattr(audio_meta, 'tags') and audio_meta.tags:
                # Standard ID3 tags
                for id3_key, meta_key in MetadataTags.ID3_TAGS.items():
                    if id3_key in audio_meta.tags:
                        tag_value = audio_meta.tags[id3_key]
                        if hasattr(tag_value, 'text'):
                            metadata[meta_key] = str(tag_value.text[0])
                        else:
                            metadata[meta_key] = str(tag_value)
                
                # Try alternate tag names if standard ones aren't found
                if not metadata['Title'] and 'TIT1' in audio_meta.tags:
                    metadata['Title'] = str(audio_meta.tags['TIT1'].text[0])
                if not metadata['Artist'] and 'TPE2' in audio_meta.tags:
                    metadata['Artist'] = str(audio_meta.tags['TPE2'].text[0])
        except Exception as e:
            logger.error(f"MP3 metadata extraction error: {str(e)}")
            
        return metadata


class DefaultMetadataExtractor(MetadataExtractor):
    """Default metadata extractor for non-MP3 formats"""
    
    def _extract_from_meta(self, audio_meta):
        """Extract metadata from non-MP3 audio files"""
        metadata = {
            'Title': '',
            'Artist': '',
            'Album': '',
        }
        
        try:
            if hasattr(audio_meta, 'tags'):
                for tag_key, tag_value in audio_meta.tags.items():
                    # Convert tag key to lowercase for consistent matching
                    tag_lower = tag_key.lower()
                    
                    # Try to match with known tag mappings
                    for known_key, meta_key in MetadataTags.COMMON_TAGS.items():
                        if known_key.lower() in tag_lower:
                            # Handle different tag value formats
                            if isinstance(tag_value, list):
                                metadata[meta_key] = str(tag_value[0])
                            elif isinstance(tag_value, (str, int, float)):
                                metadata[meta_key] = str(tag_value)
                            else:
                                try:
                                    metadata[meta_key] = str(tag_value)
                                except:
                                    continue
                            break
        except Exception as e:
            logger.error(f"General metadata extraction error: {str(e)}")
        
        return metadata


# ============================================================================
# Audio Feature Extraction
# ============================================================================

class FeatureExtractor:
    """Base class for audio feature extraction"""
    
    def __init__(self, audio_data, sample_rate):
        self.audio_data = audio_data
        self.sr = sample_rate
        self.features = {}
    
    def extract_all(self):
        """Extract all features and return as dictionary"""
        # Extract each feature group
        self.extract_basic_features()
        self.extract_spectral_features()
        self.extract_rhythm_features()
        self.extract_harmonic_features()
        self.extract_instrument_features()
        self.extract_vocal_features()
        self.extract_emotional_features()
        self.extract_pitch_features()
        
        # Ensure all float values are Python float type (not numpy)
        for key in self.features:
            if isinstance(self.features[key], (np.float32, np.float64)):
                self.features[key] = float(self.features[key])
        
        return self.features
    
    def extract_basic_features(self):
        """Extract basic temporal features"""
        self.features['Duration_Seconds'] = len(self.audio_data) / self.sr
        return self
    
    def extract_spectral_features(self):
        """Extract spectral features"""
        with ResourceManager("Spectral Features"):
            # Compute STFT
            S = librosa.stft(self.audio_data)
            
            # Compute spectral features
            spectral_centroids = librosa.feature.spectral_centroid(S=np.abs(S), sr=self.sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(S=np.abs(S), sr=self.sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=np.abs(S), sr=self.sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(S=np.abs(S), sr=self.sr)
            
            # Add features to dictionary
            self._add_mean_std_features('Spectral_Centroid', spectral_centroids)
            self._add_mean_std_features('Spectral_Rolloff', spectral_rolloff)
            self._add_mean_std_features('Spectral_Bandwidth', spectral_bandwidth)
            
            self.features['Spectral_Contrast_Mean'] = float(np.mean(spectral_contrast))
            self.features['Spectral_Contrast_Std'] = float(np.std(spectral_contrast))
            
            # Compute spectral entropy
            S_norm = np.abs(S) / (np.sum(np.abs(S)) + 1e-10)
            self.features['Spectral_Entropy'] = float(-np.sum(S_norm * np.log2(S_norm + 1e-10)))
            
            # Normalize spectral entropy
            max_entropy = -np.log2(1.0/len(S))  # Maximum possible entropy
            self.features['Spectral_Entropy'] = float(min(1.0, self.features['Spectral_Entropy'] / max_entropy))
            
            # Compute spectral flatness
            self.features['Spectral_Flatness'] = float(np.mean(librosa.feature.spectral_flatness(y=self.audio_data)))
            self.features['Spectral_Flatness'] = float(min(1.0, self.features['Spectral_Flatness']))
            
            # Compute zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(self.audio_data)[0]
            self.features['Zero_Crossing_Rate_Mean'] = float(np.mean(zcr))
            self.features['Zero_Crossing_Rate_Std'] = float(np.std(zcr))
            
            # Reassigned spectrogram features
            freqs, times, mags = librosa.reassigned_spectrogram(self.audio_data)
            self.features['Reassigned_Frequency_Mean'] = float(np.mean(freqs[np.abs(mags) > np.median(np.abs(mags))]))
            self.features['Reassigned_Magnitude_Mean'] = float(np.mean(mags))
            
            # Polynomial spectral coefficients
            poly_order = 4
            freqs = librosa.fft_frequencies(sr=self.sr)
            poly_coeffs = np.polyfit(np.arange(len(freqs)), np.mean(np.abs(S), axis=1), poly_order)
            for i, coeff in enumerate(poly_coeffs):
                self.features[f'Poly_Coefficient_{i+1}'] = float(coeff)
            
            # Bass prominence
            bass_band = librosa.fft_frequencies(sr=self.sr) <= 250
            self.features['Bass_Prominence'] = float(np.mean(np.abs(S)[bass_band]) / np.mean(np.abs(S)))
            
            # MFCC features with deltas
            mfccs_all = librosa.feature.mfcc(y=self.audio_data, sr=self.sr, n_mfcc=13)
            mfcc_deltas = librosa.feature.delta(mfccs_all)
            mfcc_delta2 = librosa.feature.delta(mfccs_all, order=2)
            
            # MFCC coefficients with deltas and delta2s
            for i, (mfcc, delta, delta2) in enumerate(zip(mfccs_all, mfcc_deltas, mfcc_delta2)):
                self.features.update({
                    f'MFCC_{i+1}_Mean': float(np.mean(mfcc)),
                    f'MFCC_{i+1}_Std': float(np.std(mfcc)),
                    f'MFCC_{i+1}_Delta_Mean': float(np.mean(delta)),
                    f'MFCC_{i+1}_Delta_Std': float(np.std(delta)),
                    f'MFCC_{i+1}_Delta2_Mean': float(np.mean(delta2)),
                    f'MFCC_{i+1}_Delta2_Std': float(np.std(delta2))
                })

        return self
    
    def extract_harmonic_features(self):
        """Extract harmonic features"""
        with ResourceManager("Harmonic Features"):
            # Harmonic-percussive source separation
            y_harmonic, y_percussive = librosa.effects.hpss(self.audio_data)
            
            # Harmonic features
            self.features['Harmonic_Energy'] = float(np.mean(np.abs(y_harmonic)))
            self.features['Percussive_Energy'] = float(np.mean(np.abs(y_percussive)))
            
            harmonic_energy = np.mean(y_harmonic**2)
            percussive_energy = np.mean(y_percussive**2)
            self.features['Harmonic_Ratio'] = float(harmonic_energy/(percussive_energy + 1e-10))
            self.features['Tonal_Energy_Ratio'] = float(np.sum(y_harmonic**2) / (np.sum(self.audio_data**2) + 1e-10))
            
            # Chroma features
            chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=self.sr)
            self.features['Chroma_Mean'] = float(np.mean(chroma))
            self.features['Chroma_Std'] = float(np.std(chroma))
            
            # Add Harmonic_Salience calculation
            self.features['Harmonic_Salience'] = float(np.mean(np.abs(y_harmonic)))
        
            # Key detection
            key, mode, confidence = self._detect_key(chroma, y_harmonic)
            self.features['Estimated_Key'] = f"{key} {mode}"
            self.features['Key_Confidence'] = float(confidence)
            
            # Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=self.sr)
            for i in range(6):
                self.features[f'Tonnetz_{i+1}'] = float(np.mean(tonnetz[i]))
            
            # Variable-Q transform features
            VQT = librosa.vqt(self.audio_data, sr=self.sr)
            self.features['VQT_Mean'] = float(np.mean(np.abs(VQT)))
            self.features['VQT_Std'] = float(np.std(np.abs(VQT)))

            # HPSS metrics
            self.features['HPSS_Harmonic_Mean'] = float(np.mean(np.abs(y_harmonic)))
            self.features['HPSS_Percussive_Mean'] = float(np.mean(np.abs(y_percussive)))
            self.features['HPSS_Ratio'] = float(np.sum(np.abs(y_harmonic)) / (np.sum(np.abs(y_percussive)) + 1e-8))
            
        return self
    
    def extract_rhythm_features(self):
        """Extract rhythm and beat-related features"""
        with ResourceManager("Rhythm Features"):
            # Onset detection
            onset_env = librosa.onset.onset_strength(y=self.audio_data, sr=self.sr)
            
            # Beat tracking
            tempo, beats = librosa.beat.beat_track(y=self.audio_data, sr=self.sr)
            self.features['Tempo_BPM'] = float(tempo.item())
            
            # Calculate beat statistics if beats were found
            if len(beats) > 1:
                beat_times = librosa.frames_to_time(beats, sr=self.sr)
                beat_intervals = np.diff(beat_times)
                self.features['Beat_Regularity'] = float(1.0 / (np.std(beat_intervals) + 1e-6))
                self.features['Beat_Density'] = float(len(beats) / self.features['Duration_Seconds'])
                self.features['Beat_Strength'] = float(np.mean(onset_env))
                
                # Calculate groove metrics
                groove = librosa.feature.tempogram(onset_envelope=onset_env, sr=self.sr)
                self.features['Groove_Consistency'] = float(1.0 / (np.std(groove, axis=1).mean() + 1e-6))
            else:
                self.features['Beat_Regularity'] = 0.0
                self.features['Beat_Density'] = 0.0
                self.features['Beat_Strength'] = 0.0
                self.features['Groove_Consistency'] = 0.0
            
            # Onset features
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr)
            self.features['Onset_Rate'] = float(len(onset_frames) / self.features['Duration_Seconds'])
            self.features['Onset_Strength_Mean'] = float(np.mean(onset_env))
            self.features['Onset_Strength_Std'] = float(np.std(onset_env))
            
            # Tempogram features
            ftempo = librosa.feature.fourier_tempogram(y=self.audio_data, sr=self.sr)
            self.features['Tempogram_Mean'] = float(np.mean(np.abs(ftempo)))
            self.features['Tempogram_Std'] = float(np.std(np.abs(ftempo)))
            
            # Additional tempogram ratio
            tgram = librosa.feature.tempogram(onset_envelope=onset_env, sr=self.sr)
            self.features['Tempogram_Ratio'] = float(np.max(np.mean(tgram, axis=1)) / np.mean(tgram))
            
            # Pulse clarity
            pulse = librosa.beat.plp(onset_envelope=onset_env, sr=self.sr)
            self.features['Pulse_Clarity'] = float(min(1.0, np.mean(pulse)))
            
            # Segment features
            boundaries = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr)
            boundary_times = librosa.frames_to_time(boundaries, sr=self.sr)

            # Calculate segment statistics
            self.features['Segment_Count'] = float(len(boundary_times))
            if len(boundary_times) > 0:
                self.features['Average_Segment_Duration'] = float(np.mean(np.diff(boundary_times))) if len(boundary_times) > 1 else 0.0
                self.features['Segment_Duration_Std'] = float(np.std(np.diff(boundary_times))) if len(boundary_times) > 1 else 0.0
                self.features['First_Segment_Time'] = float(boundary_times[0])
                self.features['Last_Segment_Time'] = float(boundary_times[-1])
            else:
                self.features['Average_Segment_Duration'] = 0.0
                self.features['Segment_Duration_Std'] = 0.0
                self.features['First_Segment_Time'] = 0.0
                self.features['Last_Segment_Time'] = 0.0

        return self
    
    def extract_instrument_features(self):
        """Extract instrument-specific features"""
        with ResourceManager("Instrument Features"):
            # Sub-bands for different instruments
            bands = {
                'bass': (20, 250),
                'kick_drum': (40, 100),
                'snare': (120, 600),
                'cymbals': (2000, 16000),
                'electric_guitar': (400, 4000),
                'vocals': (200, 4000),
                'synthesizer': (100, 8000)
            }
            
            # Calculate normalized band energies
            S = librosa.stft(self.audio_data)
            freqs = librosa.fft_frequencies(sr=self.sr)
            
            for instrument, (low, high) in bands.items():
                band_mask = np.logical_and(freqs >= low, freqs <= high)
                band_energy = np.mean(np.abs(S)[band_mask])
                total_energy = np.mean(np.abs(S))
                self.features[f'{instrument}_presence'] = float(band_energy / (total_energy + 1e-8))
            
            # Harmonic-percussive source separation for instrument detection
            y_harmonic, y_percussive = librosa.effects.hpss(self.audio_data)
            
            # Guitar detection using harmonic content
            self.features['guitar_distortion'] = float(np.mean(librosa.feature.spectral_flatness(y=y_harmonic)))
            
            # Drum detection using percussive content
            self.features['drum_prominence'] = float(np.mean(np.abs(y_percussive)) / (np.mean(np.abs(self.audio_data)) + 1e-8))
            
            # Voice detection using harmonic-percussive separation
            self.features['vocal_harmonicity'] = float(np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y_percussive)) + 1e-8))
            
            # Extended instrument analysis using onset patterns
            onset_frames = librosa.onset.onset_detect(y=self.audio_data, sr=self.sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=self.sr)
            onset_env = librosa.onset.onset_strength(y=self.audio_data, sr=self.sr)
            
            if len(onset_times) > 1:
                # Analyze onset patterns for rhythm section detection
                onset_intervals = np.diff(onset_times)
                self.features.update({
                    'rhythm_regularity': float(1.0 / (np.std(onset_intervals) + 1e-8)),
                    'rhythm_density': float(len(onset_times) / (self.audio_data.shape[0] / self.sr)),
                    'drum_pattern_strength': float(np.mean(onset_env[onset_frames]))
                })
            else:
                self.features.update({
                    'rhythm_regularity': 0.0,
                    'rhythm_density': 0.0,
                    'drum_pattern_strength': 0.0
                })

            # Timbre classification using MFCCs
            mfccs = librosa.feature.mfcc(y=self.audio_data, sr=self.sr, n_mfcc=13)
            self.features.update({
                'timbre_brightness': float(np.mean(mfccs[1:])),
                'timbre_complexity': float(np.std(mfccs)),
                'instrument_richness': float(np.mean(np.abs(librosa.feature.spectral_contrast(S=np.abs(S)))))
            })
            
        return self
    
    def extract_vocal_features(self):
        """Extract vocal-specific features"""
        with ResourceManager("Vocal Features"):
            # Only analyze vocals if they might be present
            if self.features.get('vocals_presence', 0) > 0.3:
                # Pitch variation for vocal analysis
                pitches, magnitudes = librosa.piptrack(y=self.audio_data, sr=self.sr)
                strong_pitches = pitches[magnitudes > np.mean(magnitudes) * 0.5]
                
                if len(strong_pitches) > 0:
                    self.features.update({
                        'vocal_pitch_range': float(np.ptp(strong_pitches)),
                        'vocal_pitch_stability': float(1.0 / (np.std(strong_pitches) + 1e-8)),
                        'vocal_vibrato': float(np.std(np.diff(strong_pitches)))
                    })
                else:
                    self.features.update({
                        'vocal_pitch_range': 0.0,
                        'vocal_pitch_stability': 0.0,
                        'vocal_vibrato': 0.0
                    })
                
                # Vocal formant analysis
                spectral_rolloff = librosa.feature.spectral_rolloff(y=self.audio_data, sr=self.sr)[0]
                self.features.update({
                    'vocal_formant_variation': float(np.std(spectral_rolloff)),
                    'vocal_clarity': float(np.mean(librosa.feature.spectral_contrast(y=self.audio_data, sr=self.sr)[2:5]))
                })
            else:
                self.features.update({
                    'vocal_pitch_range': 0.0,
                    'vocal_pitch_stability': 0.0,
                    'vocal_vibrato': 0.0,
                    'vocal_formant_variation': 0.0,
                    'vocal_clarity': 0.0
                })
                
            # Vocal range analysis
            mel_spec = librosa.feature.melspectrogram(y=self.audio_data, sr=self.sr)
            vocal_range = (200, 4000)  # Hz
            vocal_band = np.logical_and(
                librosa.mel_frequencies(n_mels=mel_spec.shape[0]) >= vocal_range[0],
                librosa.mel_frequencies(n_mels=mel_spec.shape[0]) <= vocal_range[1]
            )
            self.features['Vocal_Presence'] = float(np.mean(mel_spec[vocal_band]) / np.mean(mel_spec))
            
        return self
    
    def extract_emotional_features(self):
        """Extract emotional content features"""
        with ResourceManager("Emotional Features"):
            # Calculate spectral centroid for valence
            S = librosa.stft(self.audio_data)
            spectral_centroids = librosa.feature.spectral_centroid(S=np.abs(S), sr=self.sr)[0]
            
            # RMS energy for arousal
            rms = librosa.feature.rms(y=self.audio_data)[0]
            self.features['RMS_Energy_Mean'] = float(np.mean(rms))
            self.features['RMS_Energy_Std'] = float(np.std(rms))
            self.features['Dynamic_Range'] = float(np.max(rms) - np.min(rms))
            self.features['Crest_Factor'] = float(np.max(np.abs(self.audio_data)) / np.sqrt(np.mean(self.audio_data**2)))
            
            # PCEN energy calculations
            mel_spec = librosa.feature.melspectrogram(y=self.audio_data, sr=self.sr)
            pcen = librosa.pcen(mel_spec)
            self.features['PCEN_Energy_Mean'] = float(np.mean(pcen))
            self.features['PCEN_Energy_Std'] = float(np.std(pcen))

            # Calculate emotional features
            self.features['Emotional_Valence'] = float(
                0.5 * (np.mean(spectral_centroids) / (self.sr/2) + 
                       min(self.features['Tempo_BPM']/180, 1))
            )
            self.features['Emotional_Arousal'] = float(
                0.5 * (np.mean(librosa.onset.onset_strength(y=self.audio_data, sr=self.sr)) + 
                       self.features['RMS_Energy_Mean'])
            )
            
        return self

    def extract_pitch_features(self):
        """Extract pitch-related features"""
        with ResourceManager("Pitch Features"):
            # Basic pitch extraction
            pitches, magnitudes = librosa.piptrack(y=self.audio_data, sr=self.sr)
            valid_pitches = pitches[magnitudes > np.mean(magnitudes) * 0.1]  # Filter weak pitches
            
            if len(valid_pitches) > 0:
                self.features.update({
                    'Average_Pitch': float(np.mean(valid_pitches)),
                    'Pitch_Std': float(np.std(valid_pitches)),
                    'Pitch_Range': float(np.ptp(valid_pitches))
                })
            else:
                self.features.update({
                    'Average_Pitch': 0.0,
                    'Pitch_Std': 0.0,
                    'Pitch_Range': 0.0
                })
            
            # pYIN pitch features
            try:
                # Downsample audio for pYIN if it's long
                if len(self.audio_data) > self.sr * 30:  # If longer than 30 seconds
                    hop_length = 512  # Increased hop length for longer files
                else:
                    hop_length = 256  # Default hop length for shorter files
                    
                # Calculate pYIN with correct parameters
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    self.audio_data,
                    sr=self.sr,
                    fmin=librosa.note_to_hz('C2'),  # Lower bound for pitch detection
                    fmax=librosa.note_to_hz('C7'),  # Upper bound for pitch detection
                    frame_length=2048,  # Reduced from default
                    hop_length=hop_length,
                    fill_na=None,  # Don't fill NaN values
                    center=False  # Disable centering to save memory
                )
                
                # Process only valid pitch values
                valid_f0 = f0[voiced_flag]
                if len(valid_f0) > 0:
                    self.features.update({
                        'pYIN_Mean_Pitch': float(np.mean(valid_f0)),
                        'pYIN_Pitch_Std': float(np.std(valid_f0)),
                        'pYIN_Pitch_Range': float(np.ptp(valid_f0)),
                        'pYIN_Voiced_Rate': float(np.mean(voiced_flag)),
                        'pYIN_Mean_Confidence': float(np.mean(voiced_probs))
                    })
                    
                    # Additional pitch statistics only if we have enough data
                    if len(valid_f0) > 10:
                        # Calculate pitch stability
                        pitch_changes = np.diff(valid_f0)
                        self.features.update({
                            'pYIN_Pitch_Stability': float(1.0 / (np.std(pitch_changes) + 1e-6)),
                            'pYIN_Pitch_Clarity': float(np.max(voiced_probs) / (np.mean(voiced_probs) + 1e-6))
                        })
                    else:
                        self.features.update({
                            'pYIN_Pitch_Stability': 0.0,
                            'pYIN_Pitch_Clarity': 0.0
                        })
                else:
                    # Set default values if no valid pitch found
                    self.features.update({
                        'pYIN_Mean_Pitch': 0.0,
                        'pYIN_Pitch_Std': 0.0,
                        'pYIN_Pitch_Range': 0.0,
                        'pYIN_Voiced_Rate': 0.0,
                        'pYIN_Mean_Confidence': 0.0,
                        'pYIN_Pitch_Stability': 0.0,
                        'pYIN_Pitch_Clarity': 0.0
                    })
                    
            except Exception as e:
                logger.warning(f"pYIN calculation failed: {str(e)}")
                self.features.update({
                    'pYIN_Mean_Pitch': 0.0,
                    'pYIN_Pitch_Std': 0.0,
                    'pYIN_Pitch_Range': 0.0,
                    'pYIN_Voiced_Rate': 0.0,
                    'pYIN_Mean_Confidence': 0.0,
                    'pYIN_Pitch_Stability': 0.0,
                    'pYIN_Pitch_Clarity': 0.0
                })
                
        return self
    
    def _add_mean_std_features(self, feature_name, feature_data):
        """Helper to add mean and std features for a data series"""
        self.features[f'Average_{feature_name}'] = float(np.mean(feature_data))
        self.features[f'{feature_name}_Std'] = float(np.std(feature_data))
        
    def _detect_key(self, chroma, y_harmonic):
        """
        Key detection using Krumhansl-Schmuckler key-finding algorithm
        """
        # Krumhansl-Schmuckler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        # Normalize profiles
        major_profile = major_profile / major_profile.sum()
        minor_profile = minor_profile / minor_profile.sum()
        
        # Average and normalize chroma
        mean_chroma = np.mean(chroma, axis=1)
        mean_chroma = mean_chroma / (mean_chroma.sum() + 1e-8)
        
        # Initialize correlation scores
        major_cors = []
        minor_cors = []
        
        # Test all possible keys
        for i in range(12):
            # Rotate profiles to test each key
            rolled_major = np.roll(major_profile, i)
            rolled_minor = np.roll(minor_profile, i)
            
            # Calculate correlations
            major_cor = np.corrcoef(mean_chroma, rolled_major)[0,1]
            minor_cor = np.corrcoef(mean_chroma, rolled_minor)[0,1]
            
            major_cors.append(major_cor)
            minor_cors.append(minor_cor)
        
        # Convert to numpy arrays
        major_cors = np.array(major_cors)
        minor_cors = np.array(minor_cors)
        
        # Find best key and mode
        max_major_cor = np.max(major_cors)
        max_minor_cor = np.max(minor_cors)
        
        if max_major_cor > max_minor_cor:
            key_idx = np.argmax(major_cors)
            mode = 'major'
            confidence = max_major_cor
        else:
            key_idx = np.argmax(minor_cors)
            mode = 'minor'
            confidence = max_minor_cor
        
        # Map key index to key name
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = key_names[key_idx]
        
        # Calculate confidence score (0-1)
        # Compare best correlation to mean of other correlations
        if mode == 'major':
            others_mean = np.mean(np.delete(major_cors, key_idx))
            confidence = (confidence - others_mean) / (1 - others_mean + 1e-8)
        else:
            others_mean = np.mean(np.delete(minor_cors, key_idx))
            confidence = (confidence - others_mean) / (1 - others_mean + 1e-8)
        
        confidence = max(0, min(1, confidence))  # Clip to [0,1]
        
        return key, mode, confidence


# ============================================================================
# File Processing
# ============================================================================

class AudioFileProcessor:
    """Process audio files to extract features"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_number = 0
    
    def set_file_number(self, number):
        """Set file number for tracking"""
        self.file_number = number
        return self
    
    def process(self):
        """Process audio file and extract features"""
        try:
            # Load audio data
            audio_data, sample_rate = librosa.load(
                self.file_path, 
                sr=None,  # Preserve original sample rate
                mono=True,  # Convert to mono
            )
            
            # Extract metadata
            metadata_extractor = MetadataExtractor.create_extractor(self.file_path)
            metadata = metadata_extractor.extract(self.file_path)
            
            # Extract features
            feature_extractor = FeatureExtractor(audio_data, sample_rate)
            features = feature_extractor.extract_all()
            
            # Combine metadata and features
            analysis = {**metadata, **features}
            
            return analysis
            
        except Exception as e:
            logger.error(f"\nError analyzing {os.path.basename(self.file_path)}: {str(e)}")
            return None


# ============================================================================
# Directory Processing
# ============================================================================

class DirectoryScanner:
    """Scan directories for audio files"""
    
    def __init__(self, root_path):
        self.root_path = root_path
    
    def scan(self):
        """Recursively scan directory for supported audio files"""
        audio_files = []
        total_size = 0
        start_time = time.time()
        
        print("\nScanning music directory...")
        print("This may take a while for large collections.\n")
        
        # Get total number of files for progress tracking
        total_files = sum(len(files) for _, _, files in os.walk(self.root_path))
        processed_files = 0
        
        for dirpath, dirnames, filenames in os.walk(self.root_path):
            for filename in filenames:
                processed_files += 1
                if processed_files % 100 == 0:  # Update progress every 100 files
                    elapsed = time.time() - start_time
                    rate = elapsed / processed_files
                    remaining = rate * (total_files - processed_files)
                    
                    progress = (processed_files / total_files) * 100
                    print(f"\rScanning: {processed_files}/{total_files} files ({progress:.1f}%) | "
                          f"ETA: {remaining/60:.1f}", end='', flush=True)
                    
                ext = os.path.splitext(filename)[1].lower()
                if AudioFormats.is_supported(ext):
                    file_path = os.path.join(dirpath, filename)
                    try:
                        size = os.path.getsize(file_path)
                        total_size += size
                        audio_files.append({
                            'path': file_path,
                            'size': size,
                            'parent_dir': os.path.basename(dirpath)
                        })
                    except OSError as e:
                        logger.error(f"\nError accessing file {file_path}: {str(e)}")
        
        total_time = time.time() - start_time
        
        print(f"\n\nScan complete!")
        print(f"Found {len(audio_files)} audio files")
        
        return audio_files
    
    @staticmethod
    def get_audio_files(directory, track_selection='all'):
        """Get list of audio files based on selection"""
        all_files = [f for f in os.listdir(directory) 
                     if os.path.splitext(f)[1].lower() in AudioFormats.FORMATS]
        all_files.sort()
        
        if track_selection == 'all':
            return all_files
        
        try:
            if '-' in track_selection:
                start, end = map(int, track_selection.split('-'))
                return all_files[start-1:end]
            else:
                tracks = list(map(int, track_selection.split(',')))
                return [f for i, f in enumerate(all_files, 1) if i in tracks]
        except:
            logger.warning("Invalid track selection. Using all tracks.")
            return all_files


class BatchProcessor:
    """Process batches of audio files with parallel processing"""
    
    def __init__(self, files, num_workers=None):
        self.files = files
        
        if num_workers is None:
            # Use 75% of available CPUs to avoid overwhelming system
            self.num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
        else:
            self.num_workers = num_workers
            
        self.batch_size = 100  # Process files in chunks of 100
        self.results = []
        self.lock = threading.Lock()
    
    def process(self):
        """Process all files in batches"""
        total_files = len(self.files)
        logger.info(f"Processing {total_files} files with {self.num_workers} workers")
        
        # Initialize progress tracker
        progress = ProgressTracker(total_files, "Analyzing audio files")
        
        for i in range(0, total_files, self.batch_size):
            batch = self.files[i:i + self.batch_size]
            self._process_batch(batch, progress)
            
        return pd.DataFrame(self.results) if self.results else None
    
    def _process_batch(self, batch, progress):
        """Process a batch of files in parallel"""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            
            for idx, file_info in enumerate(batch):
                processor = AudioFileProcessor(file_info['path']).set_file_number(idx + 1)
                futures[executor.submit(processor.process)] = file_info
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        with self.lock:
                            self.results.append(result)
                    progress.update()
                except Exception as e:
                    logger.error(f"Error processing {futures[future]['path']}: {str(e)}")
                    progress.update()


# ============================================================================
# Output and Results
# ============================================================================

class ResultsManager:
    """Manage analysis results and output"""
    
    def __init__(self, results_df, music_directory):
        self.results_df = results_df
        self.music_directory = music_directory
    
    def save_results(self):
        """Save results to CSV with organized directory structure"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            artist = self.results_df['Artist'].iloc[0] if 'Artist' in self.results_df.columns else 'Unknown'
            analysis_type = 'library'
            base_name = f"{artist}_{analysis_type}"
            
            # Clean name for filesystem
            base_name = "".join(x for x in base_name if x.isalnum() or x in (' ', '-', '_')).strip()
            
            # Create directory structure
            output_dir, data_dir = self._create_output_structure(
                self.music_directory,
                base_name,
                timestamp
            )
            
            # Save CSV in data directory
            csv_path = os.path.join(data_dir, f"{base_name}_{timestamp}.csv")
            self.results_df.to_csv(csv_path, index=False)
            
            logger.info(f"Results saved to: {csv_path}")
            return output_dir, data_dir, csv_path
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return None
    
    def _create_output_structure(self, base_path, analysis_name, timestamp):
        """Create organized output directory structure"""
        # Create main output directory
        output_dir = os.path.join(base_path, f"{analysis_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        data_dir = os.path.join(output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        return output_dir, data_dir


# ============================================================================
# Main Application
# ============================================================================

class SpectralifyApp:
    """Main application class"""
    
    def __init__(self, music_directory):
        self.music_directory = music_directory
    
    def run(self):
        """Run the full audio analysis workflow"""
        try:
            # Create Analysis directory at the root of music_directory
            analysis_dir = os.path.join(self.music_directory, "Analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Scan for audio files
            scanner = DirectoryScanner(self.music_directory)
            audio_files = scanner.scan()
            
            if not audio_files:
                logger.warning("No audio files found in the directory")
                return
            
            # Process files in batches
            batch_size = 250
            all_results = []
            
            for i in range(0, len(audio_files), batch_size):
                batch = audio_files[i:i + batch_size]
                
                # Process batch
                processor = BatchProcessor(batch)
                results = processor.process()
                
                if results is not None:
                    # Save directly to Analysis folder
                    results_manager = ResultsManager(results, self.music_directory)
                    results_manager.save_results()
                
                # Force garbage collection after each batch
                gc.collect()

        except KeyboardInterrupt:
            logger.info("Analysis interrupted by user")
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point"""
    # Get music directory from command line or use default
    if len(sys.argv) > 1:
        music_directory = sys.argv[1]
    else:
        # Default directory - change this to your music directory
        music_directory = "C://Users//Change//This//Path"
    
    # Run the application
    app = SpectralifyApp(music_directory)
    app.run()


if __name__ == "__main__":
    main()
