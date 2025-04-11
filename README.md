<p align="center">
  <img width="600" src="https://github.com/user-attachments/assets/7388b584-d8ed-4808-b5ce-7dfbe0f1c540" alt="Spectralify Logo">
</p>

# SPECTRALIFY

A comprehensive audio analysis toolkit for extracting spectral and musical features from your music library. Whether you're analyzing a single track or an entire music collection, Spectralify provides detailed insights through advanced feature extraction and analysis.

Spectralify extracts 142 musical characteristics from your audio files, outputting them to CSV format for further analysis. Get unprecedented depth of musical data for recommendations, DJ insights, music production, and more!

*Note: All music tracks analyzed in the Example Analysis folders are owned on CD by the repository owner, Luke Cutter*

## Try SpectralifyWeb!

SpectralifyWeb takes our data and creates music recommendations with Spotify playlist output. [Click Here to Try It!](https://luke-cutter.github.io/SpectralifyWeb/)

## Key Features

### Deep Audio Analysis (13 Categories of Information)
- **Basic Information** (1-4)
- **Pitch & Tonality** (5-16)
  - Key detection with confidence ratings
  - Pitch analysis with pYIN technology
  - Harmonic content analysis
- **Rhythm Features** (17-22)
  - Tempo (BPM) detection
  - Beat regularity and strength
  - Groove consistency
- **Spectral Features** (23-32)
  - Spectral centroid and bandwidth
  - Spectral flatness and entropy
  - Reassigned spectrogram analysis
- **Tonal Features** (33-43)
  - Chroma feature extraction
  - Tonnetz (tonal space) representation
  - VQT (Variable-Q Transform) analysis
- **Energy & Dynamics** (44-55)
  - RMS energy and dynamic range
  - PCEN (Per-Channel Energy Normalization)
  - Crest factor and zero-crossing rate
- **Instrument Features** (56-68)
  - Sub-band analysis for instrument detection
  - Bass, percussion, and vocal presence
  - Guitar distortion and rhythm section detection
- **Timbre Features** (69-72)
  - Timbre brightness and complexity
  - Instrumental richness
- **Vocal Features** (73-76)
  - Vocal pitch range and stability
  - Vocal clarity and formant variation
- **Signal Analysis Features** (77-78)
- **MFCC Features** (79-130)
  - 13 MFCCs with delta and delta-delta coefficients
- **Rhythm and Structure Features** (131-138)
  - Segment detection and analysis
  - Onset detection and analysis
- **Perceptual and Emotional Features** (139-142)
  - Emotional valence and arousal estimations

### Processing Capabilities
- Batch processing for large music collections
- Multi-threaded parallel processing
- Progress tracking with ETA estimation
- Adaptive resource usage based on system capabilities

### Supported Audio Formats
- FLAC (.flac)
- MP3 (.mp3)
- WAV (.wav)
- AAC (.aac)
- AIFF (.aiff)
- WMA (.wma)

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git (for repository cloning)
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/spectralify.git
cd spectralify
```

2. Install required dependencies:
```bash
pip install librosa numpy pandas scipy mutagen audioread matplotlib seaborn
```

### Configuration

1. Open the `Spectralify.py` file
2. Set the `music_directory` variable to your music folder path:
```python
music_directory = "C:\\Users\\[Your_Username]\\Music"  # Change this path
```

3. Adjust batch size and worker count if needed:
```python
# Batch size can be reduced for large collections with memory constraints
batch_size = 250  # Default

# Worker count uses 75% of CPU cores by default
num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))  # Default
```

### Usage

Run the analysis with:
```bash
python Spectralify.py
```

The program will:
1. Scan your music directory for supported audio files
2. Process files in parallel with progress tracking
3. Extract all 142 features from each audio file
4. Save results as CSV in an "Analysis" folder within your music directory

## Output Structure

Results are saved in an organized directory structure:
```
Music/
└── Analysis/
    └── ArtistName_analysis_20250411_123456.csv
```

## Troubleshooting

### Common Issues

- **Memory usage for large libraries**: Reduce batch size in the `run_analysis()` function
- **Performance issues**: Adjust worker count based on your CPU
- **File loading errors**: Verify file format compatibility
- **Import errors**: Ensure all dependencies are installed properly

### Performance Tips

- Process fewer files at once for large libraries
- Enable parallel processing when available (on by default)
- Monitor system resource usage during processing
- For extremely large libraries, process by artist or album

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with your improvements

## License

AGPL License - See LICENSE file for details

---

This is an active project designed for audio analysis and music library organization. 
For the latest updates and features, check our GitHub repository.
