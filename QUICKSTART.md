# Quick Start Guide

Get started with Audio Data Augmentation in minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/audio-data-augmentation.git
cd audio-data-augmentation

# Install dependencies
pip install -r requirements.txt
```

## Quick Examples

### 1. Basic Audio Augmentation

```python
import librosa
import soundfile as sf
import numpy as np

# Load audio
signal, sr = librosa.load("your_audio.wav", sr=22050)

# Add white noise
noise = np.random.normal(0, signal.std(), signal.size)
augmented = signal + noise * 0.05

# Save
sf.write("augmented.wav", augmented, sr)
```

### 2. Pitch Shifting

```python
import librosa

signal, sr = librosa.load("your_audio.wav", sr=22050)

# Shift pitch up by 4 semitones
pitch_shifted = librosa.effects.pitch_shift(signal, sr=sr, n_steps=4)
```

### 3. Time Stretching

```python
# Speed up by 50%
fast = librosa.effects.time_stretch(signal, rate=1.5)

# Slow down by 20%
slow = librosa.effects.time_stretch(signal, rate=0.8)
```

### 4. Extract Mel Spectrogram

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

signal, sr = librosa.load("your_audio.wav", sr=22050)

# Extract mel spectrogram
mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)

# Visualize
plt.figure(figsize=(12, 4))
librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.show()
```

## Running the Notebooks

### Audio Augmentation Notebook

```bash
jupyter notebook Audio_Augmentation_Final.ipynb
```

This notebook demonstrates:
- Various augmentation techniques
- Visualization of waveforms and spectrograms
- SNR-based noise addition
- Environmental noise mixing

### Phoneme Classification Notebook

```bash
jupyter notebook Phonemes.ipynb
```

This notebook includes:
- Phoneme data loading
- Massive augmentation (100x per phoneme)
- CNN model training
- Phoneme recognition

## Dataset Setup

### ESC-50 Dataset

1. Download ESC-50 from: https://github.com/karolpiczak/ESC-50
2. Extract to `ESC-50-master/` directory
3. The notebook will automatically use the dataset

### Phoneme Dataset

The `phonemes/` folder should contain:
```
phonemes/
├── plosives/
├── nasals/
├── affricates/
├── fricatives/
├── approximants/
├── diphthongs/
└── monophthongs/
```

## Common Issues

### ImportError: No module named 'librosa'
```bash
pip install librosa soundfile
```

### Audio file not found
- Check file paths in notebooks
- Ensure audio files are in the correct directory
- Use absolute paths if needed

### Out of memory
- Reduce batch size in model training
- Use fewer augmentation samples
- Process files in smaller batches

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the notebooks to understand each technique
- Experiment with different augmentation parameters
- Try training on your own audio data

## Need Help?

- Check the [README.md](README.md) for detailed documentation
- Open an issue on GitHub for bugs or questions
- Review the [CONTRIBUTING.md](CONTRIBUTING.md) guide

Happy augmenting! 🎵

