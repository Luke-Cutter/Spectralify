music_directory = """C:\\Users\\cole.heigis\\Music"""

# run 
# python
# pip install pipreqs
# pipreqs C:\Users\cole.heigis\Desktop\capstone\Spectralify\Spectralify-Stripped.py
"""
Core imports and configuration setup
Organized by functionality
"""
# Standard library imports
import os
from datetime import datetime
from pathlib import Path
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings
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
    'genre': 'Genre',
    # Common variations
    'TITLE': 'Title',
    'ARTIST': 'Artist',
    'ALBUM': 'Album',
    'GENRE': 'Genre'
}


def get_default_genre(artist, album):
    """Get default genre based on artist/album"""
    
    for known_artist, genre in artist_genres.items():
        if known_artist.lower() in artist.lower():
            return genre
    
    return 'Unknown Genre'

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
        # Debug logging
        print(f"Extracting metadata from {type(audio_meta).__name__}")
        
        # Handle different audio formats
        if isinstance(audio_meta, MP3):
            metadata = extract_metadata_mp3(audio_meta)
        else:
            metadata = extract_metadata_other(audio_meta)
        
        # Fill in any missing values with defaults
        for key in metadata:
            if not metadata[key]:
                if key == 'Genre':
                    metadata[key] = get_default_genre(metadata['Artist'], metadata['Album'])
                else:
                    metadata[key] = f'Unknown {key}'
        
        print(f"Extracted metadata: {metadata}")
        return metadata
        
    except Exception as e:
        print(f"Metadata extraction failed: {str(e)}")
        return get_basic_metadata(None)

def get_basic_metadata(file_path):
    """Get basic metadata from file path with improved parsing"""
    metadata = {
        'Title': 'Unknown Title',
        'Album': 'Unknown Album',
        'Artist': 'Unknown Artist',
        'Genre': 'Unknown Genre'
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
            
            # Set genre based on artist/album
            metadata['Genre'] = get_default_genre(metadata['Artist'], metadata['Album'])
            
        except Exception as e:
            print(f"Error parsing file path metadata: {str(e)}")
    
    return metadata

def analyze_audio_file(file_path, file_number):
    """Optimized audio file analysis with improved metadata handling"""
    try:
        print(f"\nAnalyzing file: {os.path.basename(file_path)}")
        
        # Load audio with optimized parameters
        audio_data, sample_rate = librosa.load(
            file_path, 
            sr=None,  # Preserve original sample rate
            mono=True,  # Convert to mono
            duration=None
        )
        
        # Get metadata based on file type
        ext = os.path.splitext(file_path)[1].lower()
        metadata_reader = SUPPORTED_FORMATS[ext][1]
        
        metadata = None
        if metadata_reader:
            try:
                audio_meta = metadata_reader(file_path)
                metadata = extract_metadata(audio_meta)
                print(f"Metadata extracted successfully: {metadata}")
            except Exception as e:
                print(f"Error reading metadata: {str(e)}")
                metadata = None
        
        if not metadata or not any(v for v in metadata.values() if v and v != 'Unknown Genre'):
            print("Using basic metadata from file path")
            metadata = get_basic_metadata(file_path)
        
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


def extract_features(audio_data, sr):
    # """Extract comprehensive audio features with improved spectral analysis and flattened output"""
    features = {}
    
    print("1")
    # Basic temporal features
    features['Duration_Seconds'] = len(audio_data)/sr
    features['Sample_Rate'] = sr

    print("2")
    # Key and pitch detection
    y_harmonic = librosa.effects.harmonic(audio_data)
    print("21")
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    print("22")
    #key_raw = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    print("23")
    print("3")
    # Estimate musical key (same as before)
    key_profiles = {
        'C': [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        'C#': [2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29],
        'D': [2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66],
        'D#': [3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39],
        'E': [2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19],
        'F': [5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52],
        'F#': [2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09],
        'G': [4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38],
        'G#': [4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33],
        'A': [2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48],
        'A#': [3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23],
        'B': [2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35]
    }
    print("4")
    # Calculate key correlation
    key_scores = {}
    mean_chroma = np.mean(chroma, axis=1)
    for key, profile in key_profiles.items():
        correlation = np.correlate(mean_chroma, profile)[0]
        key_scores[key] = correlation * (1 + 0.1 * (profile[0] / 6.35))
    
    estimated_key = max(key_scores.items(), key=lambda x: x[1])[0]
    features['Estimated_Key'] = estimated_key
    features['Key_Confidence'] = min(1.0, float(key_scores[estimated_key]) / 10)
    print("5")


    # # Pitch features
    # pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
    # f0, voiced_flag, voiced_probs = librosa.pyin(audio_data, 
    #                                             fmin=librosa.note_to_hz('C2'),
    #                                             fmax=librosa.note_to_hz('C7'))
    
    # valid_pitches = pitches[magnitudes > np.mean(magnitudes) * 0.1]
    # print("51")
    # if len(valid_pitches) > 0:
    #     features['Average_Pitch'] = float(np.mean(valid_pitches))
    #     features['Pitch_Std'] = float(np.std(valid_pitches))
    #     features['Pitch_Range'] = float(np.ptp(valid_pitches))
    # else:
    #     features['Average_Pitch'] = 0.0
    #     features['Pitch_Std'] = 0.0
    #     features['Pitch_Range'] = 0.0

    # Enhanced pitch features with better noise handling
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
    
    
    print("6")
    # pYIN pitch features
    # valid_pyin = f0[voiced_flag]
    # features['pYIN_Pitch'] = float(np.mean(valid_pyin)) if len(valid_pyin) > 0 else 0.0
    # features['pYIN_Voiced_Rate'] = float(np.mean(voiced_flag))
    # features['pYIN_Confidence'] = float(np.mean(voiced_probs))
    print("7")
    # Harmonic features
    #harmonic_predictions = librosa.effects.harmonic(audio_data)
    features['Harmonic_Salience'] = float(np.mean(np.abs(y_harmonic)))
    print("8")
    # Rhythm features
    tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
    features['Tempo_BPM'] = float(tempo)
    
    if len(beats) > 1:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        beat_intervals = np.diff(beat_times)
        features['Beat_Regularity'] = float(min(1.0, 1.0 / (np.std(beat_intervals) + 1e-6)))
        features['Beat_Density'] = float(len(beats) / features['Duration_Seconds'])
        features['Beat_Strength'] = float(min(1.0, np.mean(librosa.onset.onset_strength(y=audio_data, sr=sr))))
    else:
        features['Beat_Regularity'] = 0.0
        features['Beat_Density'] = 0.0
        features['Beat_Strength'] = 0.0
    print("9")
    # Spectral features
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    S = librosa.stft(audio_data)
    print("10")
    # Basic spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
    
    features['Average_Spectral_Centroid'] = float(np.mean(spectral_centroids))
    features['Spectral_Centroid_Std'] = float(np.std(spectral_centroids))
    features['Average_Spectral_Rolloff'] = float(np.mean(spectral_rolloff))
    features['Spectral_Rolloff_Std'] = float(np.std(spectral_rolloff))
    features['Average_Spectral_Bandwidth'] = float(np.mean(spectral_bandwidth))
    features['Spectral_Bandwidth_Std'] = float(np.std(spectral_bandwidth))
    features['Spectral_Contrast_Mean'] = float(np.mean(spectral_contrast))
    features['Spectral_Contrast_Std'] = float(np.std(spectral_contrast))
    features['Spectral_Entropy'] = float(-np.sum(np.abs(S) * np.log2(np.abs(S) + 1e-10)))
    features['Spectral_Flatness'] = float(np.mean(librosa.feature.spectral_flatness(y=audio_data)))
    print("11")
    # Tonnetz features expanded
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    for i in range(6):
        features[f'Tonnetz_{i+1}'] = float(np.mean(tonnetz[i]))
    print("12")
    # Polynomial spectral coefficients expanded
    poly_order = 4
    freqs = librosa.fft_frequencies(sr=sr)
    poly_coeffs = np.polyfit(np.arange(len(freqs)), np.mean(np.abs(S), axis=1), poly_order)
    for i, coeff in enumerate(poly_coeffs):
        features[f'Poly_Coefficient_{i+1}'] = float(coeff)
    print("14")
    # Energy features
    rms = librosa.feature.rms(y=audio_data)[0]
    features['RMS_Energy_Mean'] = float(np.mean(rms))
    features['RMS_Energy_Std'] = float(np.std(rms))
    features['Dynamic_Range'] = float(np.max(rms) - np.min(rms))
    features['Crest_Factor'] = float(np.max(np.abs(audio_data)) / np.sqrt(np.mean(audio_data**2)))
    print("15 ")
    # PCEN energy
    pcen = librosa.pcen(mel_spec)
    features['PCEN_Energy_Mean'] = float(np.mean(pcen))
    features['PCEN_Energy_Std'] = float(np.std(pcen))
    print("16 ")
    # HPSS features
    y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
    harmonic_energy = np.mean(y_harmonic**2)
    percussive_energy = np.mean(y_percussive**2)
    
    features['Harmonic_Energy'] = float(np.mean(np.abs(y_harmonic)))
    features['Percussive_Energy'] = float(np.mean(np.abs(y_percussive)))
    features['Harmonic_Ratio'] = float(harmonic_energy/(percussive_energy + 1e-10))
    features['Tonal_Energy_Ratio'] = float(np.sum(y_harmonic**2) / (np.sum(audio_data**2) + 1e-10))
    print("17 ")
    # Variable-Q transform features
    VQT = librosa.vqt(audio_data, sr=sr)
    features['VQT_Mean'] = float(np.mean(np.abs(VQT)))
    features['VQT_Std'] = float(np.std(np.abs(VQT)))
    print("18 ")
    # Reassigned spectrogram features
    freqs, times, mags = librosa.reassigned_spectrogram(audio_data)
    features['Reassigned_Frequency_Mean'] = float(np.mean(freqs))
    features['Reassigned_Magnitude_Mean'] = float(np.mean(mags))
    print("19 ")
    # Chroma features
    chroma_stft = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    features['Chroma_Mean'] = float(np.mean(chroma_stft))
    features['Chroma_Std'] = float(np.std(chroma_stft))
    print("20 ")
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
    features['Zero_Crossing_Rate_Mean'] = float(np.mean(zcr))
    features['Zero_Crossing_Rate_Std'] = float(np.std(zcr))
    print("21 ")
    # MFCCs with deltas
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    
    for i, (mfcc, delta, delta2) in enumerate(zip(mfccs, mfcc_delta, mfcc_delta2)):
        features[f'MFCC_{i+1}_Mean'] = float(np.mean(mfcc))
        features[f'MFCC_{i+1}_Std'] = float(np.std(mfcc))
        features[f'MFCC_{i+1}_Delta_Mean'] = float(np.mean(delta))
        features[f'MFCC_{i+1}_Delta_Std'] = float(np.std(delta))
        features[f'MFCC_{i+1}_Delta2_Mean'] = float(np.mean(delta2))
        features[f'MFCC_{i+1}_Delta2_Std'] = float(np.std(delta2))
    print("22 ")
    # Rhythm and onset features
    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
    novelty = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    
    features['Onset_Rate'] = float(len(novelty) / features['Duration_Seconds'])
    features['Onset_Strength_Mean'] = float(np.mean(onset_env))
    features['Onset_Strength_Std'] = float(np.std(onset_env))
    print("23 ")
    # Tempogram features
    ftempo = librosa.feature.fourier_tempogram(y=audio_data, sr=sr)
    features['Tempogram_Mean'] = float(np.mean(np.abs(ftempo)))
    features['Tempogram_Std'] = float(np.std(np.abs(ftempo)))
    print("24")
    # Tempogram ratio features
    tgram = librosa.feature.tempogram(y=audio_data, sr=sr)
    features['Tempogram_Ratio'] = float(np.max(np.mean(tgram, axis=1)) / np.mean(tgram))
    print("25")
    # Groove and pulse features
    groove = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    
    features['Groove_Consistency'] = float(min(1.0, 1.0 / (np.std(groove, axis=1).mean() + 1e-6)))
    features['Pulse_Clarity'] = float(min(1.0, np.mean(pulse)))
    print("26")
    # Structural features
    # Recurrence Quantification Analysis (RQA)
    try:
        D = librosa.segment.recurrence_matrix(mel_spec, mode='distance')
        print("26.1")
        D = np.where(D > np.median(D), 1, 0)  # Convert to binary matrix
        print("26.2")
        features['RQA_Density'] = float(np.mean(D))
        print("26.3")
        # Calculate RQA histogram values individually
        hist_values, _ = np.histogram(D.astype(float), bins=10)
        print("26.4")
        for i, value in enumerate(hist_values):
            features[f'RQA_Hist_Bin_{i+1}'] = float(value)
        #print("26.5")
        # Path-enhanced structure
        # path_sim = librosa.segment.path_enhance(D.astype(float), n=7)
        # print("26.6")
        # features['Path_Structure_Mean'] = float(np.mean(path_sim))
        # print("26.7")
        # features['Path_Structure_Std'] = float(np.std(path_sim))
        # print("26.8")
    except Exception as e:
        print(f"Warning: RQA calculation failed - {str(e)}")
        features['RQA_Density'] = 0.0
        for i in range(10):
            features[f'RQA_Hist_Bin_{i+1}'] = 0.0
        features['Path_Structure_Mean'] = 0.0
        features['Path_Structure_Std'] = 0.0
    print("27 ")
    # HPSS separation measures
    features['HPSS_Harmonic_Mean'] = float(np.mean(np.abs(y_harmonic)))
    features['HPSS_Percussive_Mean'] = float(np.mean(np.abs(y_percussive)))
    features['HPSS_Ratio'] = float(np.sum(np.abs(y_harmonic)) / (np.sum(np.abs(y_percussive)) + 1e-8))
    print("28 ")
    # Segmentation boundaries
    boundaries = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    boundary_times = librosa.frames_to_time(boundaries, sr=sr)
    print("29 ")
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
    print("30 ")
    # Normalize features in [0-1] range
    features['Key_Confidence'] = float(min(1.0, features['Key_Confidence']))
    features['Harmonic_Salience'] = float(min(1.0, features['Harmonic_Salience']))
    features['Beat_Strength'] = float(min(1.0, features['Beat_Strength']))
    print("31 ")
    # Normalize spectral entropy
    max_entropy = -np.log2(1.0/len(S))  # Maximum possible entropy
    features['Spectral_Entropy'] = float(min(1.0, features['Spectral_Entropy'] / max_entropy))
    features['Spectral_Flatness'] = float(min(1.0, features['Spectral_Flatness']))
    print("32 ")
    # Ensure all float values are Python float type
    for key in features:
        if isinstance(features[key], (np.float32, np.float64)):
            features[key] = float(features[key])
    
    return features


def process_directory(directory, track_selection='all', num_workers=None):
    """Process directory of audio files in parallel with time estimation"""
    audio_files = get_audio_files(directory, track_selection)
    if not audio_files:
        print("No supported audio files found.")
        return None
    
    total_files = len(audio_files)
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), total_files)
    
    all_results = []
    completed = 0
    lock = threading.Lock()
    start_time = time.time()
    
    # def update_progress():
    #     nonlocal completed
    #     with lock:
    #         completed += 1
    #         progress_bar = create_progress_bar(completed, total_files, 
    #                                          start_time=start_time)
    #         print(progress_bar, end='', flush=True)
    
    print(f"\nProcessing {total_files} files using {num_workers} workers...")
    print("=" * 50)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(
                analyze_audio_file,
                os.path.join(directory, file),
                idx + 1
            ): file for idx, file in enumerate(audio_files)
        }
        
        try:
            for future in as_completed(future_to_file):
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                    #update_progress()
                except Exception as e:
                    print(f"\nError processing {future_to_file[future]}: {str(e)}")
                    #update_progress()
        except KeyboardInterrupt:
            print("\nProcess interrupted by user.")
            executor.shutdown(wait=False)
            return None
    
    # Show final statistics
    total_time = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average processing time: {total_time/total_files:.1f} seconds per file")
    print("=" * 50)
    
    return pd.DataFrame(all_results) if all_results else None

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
    """Main execution function with complete analysis pipeline"""
    try:
        print("\nInitializing analysis...")
        audio_files = scan_music_directory(music_directory)
        if audio_files:
            results = process_scan_results(audio_files,)
        else:
            print("No audio files found.")
            return
        
        if results is not None:
            # Save results with new directory structure
            output_dir, csv_path = handle_csv_output(results)
            
            if output_dir and csv_path:
                print(f"\nResults saved to: {output_dir}")
                print(f"CSV file: {os.path.basename(csv_path)}")
                print(f"\nAll outputs saved to: {output_dir}")
                        
        else:
            print("\nNo results generated.")
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        print("\nAnalysis process completed.")

def process_scan_results(audio_files):
    """Process results from a full music directory scan with parallel processing"""
    try:
        total_files = len(audio_files)
        print(f"\nProcessing {total_files} files...")
        
        # Determine optimal number of workers
        num_workers = min(multiprocessing.cpu_count(), total_files)
        print(f"Using {num_workers} parallel workers\n")
        
        all_results = []
        completed = 0
        lock = threading.Lock()
        start_time = time.time()
        
        def update_progress():
            nonlocal completed
            with lock:
                completed += 1
                #progress = create_progress_bar(completed, total_files, start_time=start_time)
                print(completed)
        
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
        
        print("\n")  # New line after progress bar
        
        if all_results:
            return pd.DataFrame(all_results)
        else:
            print("No results generated from scan.")
            return None
            
    except Exception as e:
        print(f"Error processing scan results: {str(e)}")
        return None
    
print("eeeeee")
print(analyze_audio_file("C:\\Users\\Cheig\\Music\\OnTheSpot\\Tracks\\Noah Kahan\\[2023] Stick Season\\1. The View Between Villages (Extended).flac",1))
run_analysis()