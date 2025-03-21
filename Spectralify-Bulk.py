"""
Getting Started: 
1. Get all of the required imports
pip install librosa numpy pandas scipy mutagen audioread

2. Change the path to the path of the music directory, artist, album, or song you want to analyze

3. Adjustable Variables For Smoother Extraction:
* Batch Size- If you're processing a large music collection and encounter memory issues, reduce the batch size in the run_analysis() function:
* Worker Count- By default, the code uses 75% of available CPU cores. Adjust if needed (num_workers)

4. 
"""

music_directory = "C:\\Users\\[User_Name]\\Music" # Change Me!


"""
Core imports and configuration setup
Organized by functionality
"""
# Standard library imports
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import gc
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings
from typing import Dict, List, Optional, Any, Tuple
#from IPython.display import HTML
import time

# Audio processing imports
import librosa
import audioread
#from audioread import rawread
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from mutagen.aac import AAC
from mutagen.aiff import AIFF
from mutagen.id3 import ID3

# Data handling imports
import numpy as np
import pandas as pd
from scipy import stats

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants and configurations
SUPPORTED_FORMATS = {
    '.flac': ['audio/flac', FLAC],
    '.mp3': ['audio/mp3', MP3],
    '.wav': ['audio/wav', WAVE],
    '.aac': ['audio/aac', AAC],
    '.aiff': ['audio/aiff', AIFF],
    '.wma': ['audio/wma', None]  # Requires additional handling
}

# Tag mapping for other audio formats
AUDIO_TAG_MAPPING = {
    'title': 'Title',
    'artist': 'Artist',
    'album': 'Album',
    # Common variations
    'TITLE': 'Title',
    'ARTIST': 'Artist',
    'ALBUM': 'Album'
}

ID3_TAG_MAPPING = {
    'TIT2': 'Title',     # Title/songname/content description
    'TPE1': 'Artist',    # Lead performer(s)/Soloist(s)
    'TALB': 'Album',     # Album/Movie/Show title

    # Alternate tag names for backwards compatibility
    'TT2': 'Title',      # ID3v2.2 equivalent of TIT2
    'TP1': 'Artist',     # ID3v2.2 equivalent of TPE1
    'TAL': 'Album',      # ID3v2.2 equivalent of TALB
}

def extract_metadata_mp3(audio_meta):
    """Extract metadata specifically from MP3 files"""
    metadata = {
        'Title': '',
        'Artist': '',
        'Album': '',
    }
    
    try:
        if hasattr(audio_meta, 'tags') and audio_meta.tags:
            # Standard ID3 tags
            for id3_key, meta_key in ID3_TAG_MAPPING.items():
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
        print(f"MP3 metadata extraction error: {str(e)}")
        
    return metadata

class ProgressTracker:
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
        """Convert seconds to a human-readable format"""
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


def extract_metadata_other(audio_meta):
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
                for known_key, meta_key in AUDIO_TAG_MAPPING.items():
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
        print(f"General metadata extraction error: {str(e)}")
    
    return metadata

def extract_metadata(audio_meta):
    """Extract metadata from audio file with enhanced format handling"""
    try:
        # Handle different audio formats
        if isinstance(audio_meta, MP3):
            metadata = extract_metadata_mp3(audio_meta)
        else:
            metadata = extract_metadata_other(audio_meta)
        

        return metadata
        
    except Exception as e:
        print(f"Metadata extraction failed: {str(e)}")
        return get_basic_metadata(None)

def get_basic_metadata(file_path):
    """Get basic metadata from file path with improved parsing"""
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
            print(f"Error parsing file path metadata: {str(e)}")
    
    return metadata

def analyze_audio_file(file_path, file_number):
    """Optimized audio file analysis with improved metadata handling"""
    try:
        # Load audio with optimized parameters
        audio_data, sample_rate = librosa.load(
            file_path, 
            sr=None,  # Preserve original sample rate
            mono=True,  # Convert to mono
        )
        
        # Get metadata based on file type
        ext = os.path.splitext(file_path)[1].lower()
        metadata_reader = SUPPORTED_FORMATS[ext][1]
        
        metadata = None
        if metadata_reader:
            try:
                audio_meta = metadata_reader(file_path)
                metadata = extract_metadata(audio_meta)
            except Exception as e:
                print(f"Error reading metadata: {str(e)}")
                metadata = None
        
        # Extract features
        features = extract_features(audio_data, sample_rate)
        
        # Combine metadata and features
        analysis = {
            **metadata,
            **features
        }
        
        return analysis
        
    except Exception as e:
        print(f"\nError analyzing {os.path.basename(file_path)}: {str(e)}")
        return None
    

def validate_audio_file(file_path):
    """Validate if file is supported audio format"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_FORMATS

def get_audio_files(directory, track_selection='all'):
    """Get list of audio files based on selection"""
    all_files = [f for f in os.listdir(directory) 
                 if validate_audio_file(f)]
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
        print("Invalid track selection. Using all tracks.")
        return all_files

def scan_music_directory(root_path):
    """Recursively scan directory for supported audio files with time tracking"""
    audio_files = []
    total_size = 0
    start_time = time.time()
    
    print("\nScanning music directory...")
    print("This may take a while for large collections.\n")
    
    # Get total number of files for progress tracking
    total_files = sum(len(files) for _, _, files in os.walk(root_path))
    processed_files = 0
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            processed_files += 1
            if processed_files % 100 == 0:  # Update progress every 100 files
                elapsed = time.time() - start_time
                rate = elapsed / processed_files
                remaining = rate * (total_files - processed_files)
                
                progress = (processed_files / total_files) * 100
                print(f"\rScanning: {processed_files}/{total_files} files ({progress:.1f}%) | "
                      f"ETA: {remaining/60:.1f}", end='', flush=True)
                
            if any(filename.lower().endswith(ext) for ext in SUPPORTED_FORMATS.keys()):
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
                    print(f"\nError accessing file {file_path}: {str(e)}")
    
    total_time = time.time() - start_time
    
    print(f"\n\nScan complete!")
    print(f"Found {len(audio_files)} audio files")
    
    return audio_files

def detect_key(chroma, y_harmonic, sr):
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


def extract_features(audio_data, sr):
    # """Extract comprehensive audio features with improved spectral analysis and flattened output"""
    features = {}

    # Cache commonly used computations
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    y_harmonic = librosa.effects.harmonic(audio_data)
    S = librosa.stft(audio_data)
    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
    
    # Use cached values instead of recomputing
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    
    # Batch compute MFCC features
    mfccs_all = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    mfcc_deltas = librosa.feature.delta(mfccs_all)
    mfcc_delta2 = librosa.feature.delta(mfccs_all, order=2)
    
    # Batch process spectral features using cached S
    spectral_features = {
        'centroids': librosa.feature.spectral_centroid(S=np.abs(S), sr=sr)[0],
        'rolloff': librosa.feature.spectral_rolloff(S=np.abs(S), sr=sr)[0],
        'bandwidth': librosa.feature.spectral_bandwidth(S=np.abs(S), sr=sr)[0],
        'contrast': librosa.feature.spectral_contrast(S=np.abs(S), sr=sr)
    }
    
    # Basic temporal features
    features['Duration_Seconds'] = len(audio_data)/sr

    # Calculate key correlation
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    key, mode, confidence = detect_key(chroma, y_harmonic, sr)
    features['Estimated_Key'] = f"{key} {mode}"
    features['Key_Confidence'] = float(confidence)


    # Pitch features with noise handling
    pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
    valid_pitches = pitches[magnitudes > np.mean(magnitudes) * 0.1]  # Filter weak pitches
    if len(valid_pitches) > 0:
        features.update({
            'Average_Pitch': float(np.mean(valid_pitches)),
            'Pitch_Std': float(np.std(valid_pitches)),
            'Pitch_Range': float(np.ptp(valid_pitches))
        })
    else:
        features.update({
            'Average_Pitch': 0.0,
            'Pitch_Std': 0.0,
            'Pitch_Range': 0.0
        })
    
    # pYIN pitch features
    try:
        # Downsample audio for pYIN if it's long
        if len(audio_data) > sr * 30:  # If longer than 30 seconds
            hop_length = 512  # Increased hop length for longer files
        else:
            hop_length = 256  # Default hop length for shorter files
            
        # Calculate pYIN with correct parameters
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data,
            sr=sr,
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
            features.update({
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
                features.update({
                    'pYIN_Pitch_Stability': float(1.0 / (np.std(pitch_changes) + 1e-6)),
                    'pYIN_Pitch_Clarity': float(np.max(voiced_probs) / (np.mean(voiced_probs) + 1e-6))
                })
            else:
                features.update({
                    'pYIN_Pitch_Stability': 0.0,
                    'pYIN_Pitch_Clarity': 0.0
                })
        else:
            # Set default values if no valid pitch found
            features.update({
                'pYIN_Mean_Pitch': 0.0,
                'pYIN_Pitch_Std': 0.0,
                'pYIN_Pitch_Range': 0.0,
                'pYIN_Voiced_Rate': 0.0,
                'pYIN_Mean_Confidence': 0.0,
                'pYIN_Pitch_Stability': 0.0,
                'pYIN_Pitch_Clarity': 0.0
            })
            
        # Clean up
        del f0
        del voiced_flag
        del voiced_probs
        gc.collect()
        
    except Exception as e:
        print(f"Warning: pYIN calculation failed - {str(e)}")
        features.update({
            'pYIN_Mean_Pitch': 0.0,
            'pYIN_Pitch_Std': 0.0,
            'pYIN_Pitch_Range': 0.0,
            'pYIN_Voiced_Rate': 0.0,
            'pYIN_Mean_Confidence': 0.0,
            'pYIN_Pitch_Stability': 0.0,
            'pYIN_Pitch_Clarity': 0.0
        })

    # Harmonic features
    features['Harmonic_Salience'] = float(np.mean(np.abs(y_harmonic)))

    # Rhythm features
    tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
    features['Tempo_BPM'] = float(tempo)
    
    if len(beats) > 1:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        beat_intervals = np.diff(beat_times)
        features['Beat_Regularity'] = float(1.0 / (np.std(beat_intervals) + 1e-6))
        features['Beat_Density'] = float(len(beats) / features['Duration_Seconds'])
        features['Beat_Strength'] = float(np.mean(onset_env))
        
        # Calculate groove first
        groove = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        features['Groove_Consistency'] = float(1.0 / (np.std(groove, axis=1).mean() + 1e-6))
    else:
        features['Beat_Regularity'] = 0.0
        features['Beat_Density'] = 0.0
        features['Beat_Strength'] = 0.0
        features['Groove_Consistency'] = 0.0
        
    # Use the pre-computed spectral features from earlier
    features.update({
        'Average_Spectral_Centroid': float(np.mean(spectral_features['centroids'])),
        'Spectral_Centroid_Std': float(np.std(spectral_features['centroids'])),
        'Average_Spectral_Rolloff': float(np.mean(spectral_features['rolloff'])),
        'Spectral_Rolloff_Std': float(np.std(spectral_features['rolloff'])),
        'Average_Spectral_Bandwidth': float(np.mean(spectral_features['bandwidth'])),
        'Spectral_Bandwidth_Std': float(np.std(spectral_features['bandwidth'])),
        'Spectral_Contrast_Mean': float(np.mean(spectral_features['contrast'])),
        'Spectral_Contrast_Std': float(np.std(spectral_features['contrast']))
    })

    S_norm = np.abs(S) / (np.sum(np.abs(S)) + 1e-10)
    features['Spectral_Entropy'] = float(-np.sum(S_norm * np.log2(S_norm + 1e-10)))
    features['Spectral_Flatness'] = float(np.mean(librosa.feature.spectral_flatness(y=audio_data)))

    # Tonnetz features expanded
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    for i in range(6):
        features[f'Tonnetz_{i+1}'] = float(np.mean(tonnetz[i]))
    
    # Polynomial spectral coefficients expanded
    poly_order = 4
    freqs = librosa.fft_frequencies(sr=sr)
    poly_coeffs = np.polyfit(np.arange(len(freqs)), np.mean(np.abs(S), axis=1), poly_order)
    for i, coeff in enumerate(poly_coeffs):
        features[f'Poly_Coefficient_{i+1}'] = float(coeff)

    # Energy features
    rms = librosa.feature.rms(y=audio_data)[0]
    features['RMS_Energy_Mean'] = float(np.mean(rms))
    features['RMS_Energy_Std'] = float(np.std(rms))
    features['Dynamic_Range'] = float(np.max(rms) - np.min(rms))
    features['Crest_Factor'] = float(np.max(np.abs(audio_data)) / np.sqrt(np.mean(audio_data**2)))

    # PCEN energy
    pcen = librosa.pcen(mel_spec)
    features['PCEN_Energy_Mean'] = float(np.mean(pcen))
    features['PCEN_Energy_Std'] = float(np.std(pcen))

    # HPSS features
    y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
    harmonic_energy = np.mean(y_harmonic**2)
    percussive_energy = np.mean(y_percussive**2)
    features['Harmonic_Energy'] = float(np.mean(np.abs(y_harmonic)))
    features['Percussive_Energy'] = float(np.mean(np.abs(y_percussive)))
    features['Harmonic_Ratio'] = float(harmonic_energy/(percussive_energy + 1e-10))
    features['Tonal_Energy_Ratio'] = float(np.sum(y_harmonic**2) / (np.sum(audio_data**2) + 1e-10))

    # Variable-Q transform features
    VQT = librosa.vqt(audio_data, sr=sr)
    features['VQT_Mean'] = float(np.mean(np.abs(VQT)))
    features['VQT_Std'] = float(np.std(np.abs(VQT)))
    # Clean up large variables for better performance
    del VQT
    gc.collect()

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
    freqs = librosa.fft_frequencies(sr=sr)
    for instrument, (low, high) in bands.items():
        band_mask = np.logical_and(freqs >= low, freqs <= high)
        band_energy = np.mean(np.abs(S)[band_mask])
        total_energy = np.mean(np.abs(S))
        features[f'{instrument}_presence'] = float(band_energy / (total_energy + 1e-8))
    
    # Additional instrument-specific features
    features.update({
        # Guitar detection using harmonic content
        'guitar_distortion': float(np.mean(librosa.feature.spectral_flatness(y=y_harmonic))),
        
        # Drum detection using percussive content
        'drum_prominence': float(np.mean(np.abs(y_percussive)) / (np.mean(np.abs(audio_data)) + 1e-8)),
        
        # Voice detection using harmonic-percussive separation
        'vocal_harmonicity': float(np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y_percussive)) + 1e-8)),
    })
    
    # Extended instrument analysis using onset patterns
    onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    if len(onset_times) > 1:
        # Analyze onset patterns for rhythm section detection
        onset_intervals = np.diff(onset_times)
        features.update({
            'rhythm_regularity': float(1.0 / (np.std(onset_intervals) + 1e-8)),
            'rhythm_density': float(len(onset_times) / (audio_data.shape[0] / sr)),
            'drum_pattern_strength': float(np.mean(onset_env[onset_frames]))
        })
    else:
        features.update({
            'rhythm_regularity': 0.0,
            'rhythm_density': 0.0,
            'drum_pattern_strength': 0.0
        })

    # Timbre classification using MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    features.update({
        'timbre_brightness': float(np.mean(mfccs[1:])),
        'timbre_complexity': float(np.std(mfccs)),
        'instrument_richness': float(np.mean(np.abs(librosa.feature.spectral_contrast(S=np.abs(S)))))
    })
    
    # Vocal characteristics analysis
    if features['vocals_presence'] > 0.3:  # Only if vocals are detected
        # Pitch variation for vocal analysis
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
        strong_pitches = pitches[magnitudes > np.mean(magnitudes) * 0.5]
        
        if len(strong_pitches) > 0:
            features.update({
                'vocal_pitch_range': float(np.ptp(strong_pitches)),
                'vocal_pitch_stability': float(1.0 / (np.std(strong_pitches) + 1e-8)),
                'vocal_vibrato': float(np.std(np.diff(strong_pitches)))
            })
        else:
            features.update({
                'vocal_pitch_range': 0.0,
                'vocal_pitch_stability': 0.0,
                'vocal_vibrato': 0.0
            })
            
        # Vocal formant analysis
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
        features.update({
            'vocal_formant_variation': float(np.std(spectral_rolloff)),
            'vocal_clarity': float(np.mean(librosa.feature.spectral_contrast(y=audio_data, sr=sr)[2:5]))
        })
    else:
        features.update({
            'vocal_pitch_range': 0.0,
            'vocal_pitch_stability': 0.0,
            'vocal_vibrato': 0.0,
            'vocal_formant_variation': 0.0,
            'vocal_clarity': 0.0
        })

    # Reassigned spectrogram features
    freqs, times, mags = librosa.reassigned_spectrogram(audio_data)
    features['Reassigned_Frequency_Mean'] = float(np.mean(freqs[np.abs(mags) > np.median(np.abs(mags))]))  # Only use significant magnitudes
    features['Reassigned_Magnitude_Mean'] = float(np.mean(mags))

    # Chroma features
    chroma_stft = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    features['Chroma_Mean'] = float(np.mean(chroma_stft))
    features['Chroma_Std'] = float(np.std(chroma_stft))

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
    features['Zero_Crossing_Rate_Mean'] = float(np.mean(zcr))
    features['Zero_Crossing_Rate_Std'] = float(np.std(zcr))

    # MFCCs with deltas
    for i, (mfcc, delta, delta2) in enumerate(zip(mfccs_all, mfcc_deltas, mfcc_delta2)):
        features.update({
            f'MFCC_{i+1}_Mean': float(np.mean(mfcc)),
            f'MFCC_{i+1}_Std': float(np.std(mfcc)),
            f'MFCC_{i+1}_Delta_Mean': float(np.mean(delta)),
            f'MFCC_{i+1}_Delta_Std': float(np.std(delta)),
            f'MFCC_{i+1}_Delta2_Mean': float(np.mean(delta2)),
            f'MFCC_{i+1}_Delta2_Std': float(np.std(delta2))
        })


    # Rhythm and onset features
    novelty = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    features['Onset_Rate'] = float(len(novelty) / features['Duration_Seconds'])
    features['Onset_Strength_Mean'] = float(np.mean(onset_env))
    features['Onset_Strength_Std'] = float(np.std(onset_env))

    # Tempogram features
    ftempo = librosa.feature.fourier_tempogram(y=audio_data, sr=sr)
    features['Tempogram_Mean'] = float(np.mean(np.abs(ftempo)))
    features['Tempogram_Std'] = float(np.std(np.abs(ftempo)))

    # Tempogram ratio features
    tgram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    features['Tempogram_Ratio'] = float(np.max(np.mean(tgram, axis=1)) / np.mean(tgram))

    # Groove and pulse features
    groove = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    
    features['Groove_Consistency'] = float(1.0 / (np.std(groove, axis=1).mean() + 1e-6))
    features['Pulse_Clarity'] = float(min(1.0, np.mean(pulse)))

    # HPSS separation measures
    features['HPSS_Harmonic_Mean'] = float(np.mean(np.abs(y_harmonic)))
    features['HPSS_Percussive_Mean'] = float(np.mean(np.abs(y_percussive)))
    features['HPSS_Ratio'] = float(np.sum(np.abs(y_harmonic)) / (np.sum(np.abs(y_percussive)) + 1e-8))
    # Clean up large variables for better performance
    del y_harmonic
    del y_percussive
    gc.collect()

    # Segmentation boundaries
    boundaries = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    boundary_times = librosa.frames_to_time(boundaries, sr=sr)

    # Convert boundary times to statistics instead of array
    features['Segment_Count'] = float(len(boundary_times))
    if len(boundary_times) > 0:
        features['Average_Segment_Duration'] = float(np.mean(np.diff(boundary_times)))
        features['Segment_Duration_Std'] = float(np.std(np.diff(boundary_times))) if len(boundary_times) > 1 else 0.0
        features['First_Segment_Time'] = float(boundary_times[0])
        features['Last_Segment_Time'] = float(boundary_times[-1])
    else:
        features['Average_Segment_Duration'] = 0.0
        features['Segment_Duration_Std'] = 0.0
        features['First_Segment_Time'] = 0.0
        features['Last_Segment_Time'] = 0.0

    # Normalize features in [0-1] range
    features['Key_Confidence'] = float(min(1.0, features['Key_Confidence']))
    features['Harmonic_Salience'] = float(min(1.0, features['Harmonic_Salience']))

    # Normalize spectral entropy
    max_entropy = -np.log2(1.0/len(S))  # Maximum possible entropy
    features['Spectral_Entropy'] = float(min(1.0, features['Spectral_Entropy'] / max_entropy))
    features['Spectral_Flatness'] = float(min(1.0, features['Spectral_Flatness']))

    # Bass Prominence
    bass_band = librosa.fft_frequencies(sr=sr) <= 250
    features['Bass_Prominence'] = float(np.mean(np.abs(S)[bass_band]) / np.mean(np.abs(S)))
    
    # Vocal Features
    vocal_range = (200, 4000)  # Hz
    vocal_band = np.logical_and(
        librosa.mel_frequencies(n_mels=mel_spec.shape[0]) >= vocal_range[0],
        librosa.mel_frequencies(n_mels=mel_spec.shape[0]) <= vocal_range[1]
    )
    features['Vocal_Presence'] = float(np.mean(mel_spec[vocal_band]) / np.mean(mel_spec))
    
    # Emotional Features
    features['Emotional_Valence'] = float(
        0.5 * (np.mean(spectral_features['centroids']) / (sr/2) + 
               min(features['Tempo_BPM']/180, 1))  # Use features['Tempo_BPM'] instead of tempo
    )
    features['Emotional_Arousal'] = float(
        0.5 * (np.mean(onset_env) + 
               features['RMS_Energy_Mean'])  # Use already calculated RMS energy
    )
    
    # Clean up large variables for better performance
    del mel_spec
    del S
    gc.collect()

    # Ensure all float values are Python float type
    for key in features:
        if isinstance(features[key], (np.float32, np.float64)):
            features[key] = float(features[key])
    
    return features


def process_directory(directory, track_selection='all', num_workers=None):
    """Process directory of audio files in parallel with time estimation"""
    if num_workers is None:
        # Use 75% of available CPUs to avoid overwhelming system
        num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
    
    # Use chunking for better memory management
    chunk_size = 100  # Process files in chunks of 100
    
    audio_files = get_audio_files(directory, track_selection)
    results = []
    
    for i in range(0, len(audio_files), chunk_size):
        chunk = audio_files[i:i + chunk_size]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(analyze_audio_file, 
                                    os.path.join(directory, file), 
                                    idx): file 
                      for idx, file in enumerate(chunk)}
            
            for future in as_completed(futures):
                if future.result():
                    results.append(future.result())
    
    return pd.DataFrame(results) if results else None

def create_output_structure(base_path, analysis_name, timestamp):
    """Create organized output directory structure"""
    # Create main output directory
    output_dir = os.path.join(base_path, f"{analysis_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    viz_dir = os.path.join(output_dir, "visualizations")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    return output_dir, viz_dir, data_dir

def handle_csv_output(df_results):
    """Enhanced CSV output handling with organized directory structure"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        artist = df_results['Artist'].iloc[0] if 'Artist' in df_results.columns else 'Unknown'
        analysis_type = {
            '1': 'single_track',
            '2': 'album',
            '3': 'artist',
            '4': 'library'
        }.get(4, 'analysis')
        base_name = f"{artist}_{analysis_type}"
        
        # Clean name for filesystem
        base_name = "".join(x for x in base_name if x.isalnum() or x in (' ', '-', '_')).strip()
        
        # Create directory structure
        output_dir, viz_dir, data_dir = create_output_structure(
            music_directory, #same as input dir
            base_name,
            timestamp
        )
        
        # Save CSV in data directory
        csv_path = os.path.join(data_dir, f"{base_name}_{timestamp}.csv")
        df_results.to_csv(csv_path, index=False)
        
        return output_dir, csv_path
        
    except Exception as e:
        print(f"\nError saving results: {str(e)}")
        return None
    

def run_analysis():
    try:
        # Create Analysis directory at the root of music_directory
        analysis_dir = os.path.join(music_directory, "Analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        audio_files = scan_music_directory(music_directory)
        if audio_files:
            # Process in batches to manage memory
            batch_size = 250
            all_results = []
            
            for i in range(0, len(audio_files), batch_size):
                batch = audio_files[i:i + batch_size]
                results = process_scan_results(batch)
                if results is not None:
                    # Save after every batch to avoid crashes
                    final_results = results
                    
                    # Save directly to Analysis folder
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    artist = final_results['Artist'].iloc[0] if 'Artist' in final_results.columns else 'Unknown'
                    base_name = f"{artist}_analysis"
                    base_name = "".join(x for x in base_name if x.isalnum() or x in (' ', '-', '_')).strip()
                    
                    # Save CSV in Analysis directory
                    csv_path = os.path.join(analysis_dir, f"{base_name}_{timestamp}.csv")
                    final_results.to_csv(csv_path, index=False)
                    
                    print(f"\nResults saved to: {csv_path}")
                
                # Force garbage collection after each batch
                gc.collect()

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")

def process_scan_results(audio_files):
    """Process results from a full music directory scan with parallel processing"""
    try:
        total_files = len(audio_files)
        print(f"\nInitializing analysis of {total_files} audio files...")
        
        # Determine optimal number of workers
        num_workers = min(multiprocessing.cpu_count(), total_files)
        print(f"Using {num_workers} parallel workers\n")
        
        all_results = []
        lock = threading.Lock()
        
        # Initialize progress tracker
        progress = ProgressTracker(total_files, "Analyzing audio files")
        
        def update_progress():
            with lock:
                progress.update()
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {
                executor.submit(
                    analyze_audio_file,
                    file_info['path'],
                    idx + 1
                ): file_info for idx, file_info in enumerate(audio_files)
            }
            
            try:
                for future in as_completed(future_to_file):
                    try:
                        result = future.result()
                        if result:
                            all_results.append(result)
                        update_progress()
                    except Exception as e:
                        print(f"\nError processing {future_to_file[future]['path']}: {str(e)}")
                        update_progress()   
            except KeyboardInterrupt:
                print("\nProcess interrupted by user.")
                executor.shutdown(wait=False)
                return None
        
        if all_results:
            return pd.DataFrame(all_results)
        else:
            print("No results generated from scan.")
            return None
            
    except Exception as e:
        print(f"Error processing scan results: {str(e)}")
        return None

if __name__ == "__main__":
    run_analysis()