  <p align="center">
    <img width="600" src="https://github.com/user-attachments/assets/7388b584-d8ed-4808-b5ce-7dfbe0f1c540" alt="Text Spectralify Logo">
  </p>

# What is SPECTRALIFY?
A comprehensive audio analysis toolkit for extracting spectral and musical features from your music library. Whether you're analyzing a single track or an entire music collection, Spectralify provides detailed insights through advanced visualization and statistical analysis.

* Note: All music tracks trained in the Example Analysis folder are owned on CD by the owner of this repository, Luke Cutter

## Features
Deep Audio Analysis

- Spectral characteristics (centroids, bandwidth, rolloff)
- Musical features (BPM, key detection, beat tracking)
- Rhythm and groove pattern analysis
- Energy and dynamics profiling
- Advanced MFCC (Mel-frequency cepstral coefficients) analysis



## Rich Visualization Suite

- Temporal and spectral analysis plots
- MFCC coefficient heatmaps
- Track similarity matrices
- Feature correlation heatmaps
- Energy distribution graphs
- Interactive HTML reports



## Processing Options 
- Single file analysis
- Album directory processing
- Artist directory batch analysis
- Full library scanning
- Parallel processing for improved speed

## Supported Audio Formats
- FLAC (.flac)
- MP3 (.mp3)
- WAV (.wav)
- AAC (.aac)
- AIFF (.aiff)
- WMA (.wma)

# Quick Start
### Prerequisites
You'll need these installed first:
- Python 3.8+
- Git
- pip (Python package installer)
- Jupyter Notebook (For Version 1.0, future versions will have other file formats)



# Dependencies:
```
librosa
audioread
mutagen
numpy
pandas
scipy
matplotlib
seaborn
```



# Usage
## Quick Input Format
```
mode,path,output_directory,naming,visualizations,show_stats,show_docs,html_report
```
## Standard Menu
You can also choose to use the standard menu which will take the user step-by-step through the file loading and analysis portions!
## Analysis Modes

1. Single file
2. Album directory
3. Artist directory
4. Full library scan

## Example Commands
```
# Single track analysis
1,C:\Music\song.flac

# Album analysis with visualizations
2,C:\Music\Album,,auto,yes

# Artist directory with custom output
3,C:\Music\Artist,C:\Analysis,artist_analysis,yes,yes,yes,yes
```

## Output Strcture:
```
output_dir/
├── analysis_name_timestamp/
│   ├── data/
│   │   ├── analysis_results.csv
│   │   └── analysis_info.txt
│   ├── visualizations/
│   │   ├── temporal_analysis.png
│   │   ├── spectral_analysis.png
│   │   └── ...
│   └── analysis_report.html
```



# Troubleshooting
## Common Issues

Importing:
Use the Jupyter Notebook terminal and pip import the following-
```
librosa, audioread, mutagen (all file types you need for your program), numpy, pandas, scipy, matplot lib, and seaborn
```


Memory usage for large libraries: Process fewer files at once
Performance: Utilize parallel processing options
File loading errors: Check file format compatibility

## Performance Tips

Use batch processing for large libraries
Enable parallel processing when available
Monitor system resource usage




# Contributing

Fork the repository
Create a feature branch
Submit a pull request

# License
MIT License - See LICENSE file for details

This is an active project designed for audio analysis and music library organization. For the latest updates and features, check our GitHub repository.








