# Audio Data Augmentation

A comprehensive Python project for audio data augmentation and classification using deep learning. This project demonstrates various audio augmentation techniques and implements CNN-based models for audio classification tasks including environmental sound classification (ESC-50) and phoneme recognition.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Augmentation Techniques](#augmentation-techniques)
- [Models](#models)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Audio Augmentation Techniques
- **White Noise Addition**: Add white noise with configurable SNR levels
- **Pitch Shifting**: Shift audio pitch up or down by semitones
- **Time Stretching**: Speed up or slow down audio without changing pitch
- **Speed Change**: Change playback speed (resampling)
- **Volume Adjustment**: Increase or decrease volume in dB
- **Environmental Noise**: Add real-world noise (car noise, babble, etc.) with SNR control
- **SpecAugment**: Frequency and time masking on spectrograms

### Deep Learning Models
- **CNN for Environmental Sound Classification**: Classifies sounds from ESC-50 dataset (siren, car_horn)
- **CNN for Phoneme Recognition**: Classifies 44 different phonemes across 7 categories

### Visualization
- Waveform visualization
- Spectrogram visualization (linear and mel-scale)
- Comparison plots for original vs augmented audio

## 📁 Project Structure

```
Audio Data Augmentation/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
│
├── Audio_Augmentation_Final.ipynb    # Main augmentation notebook
├── Phonemes.ipynb                     # Phoneme classification notebook
│
├── phonemes/                          # Phoneme audio samples
│   ├── plosives/                      # Plosive phonemes (p, t, k, b, d, g)
│   ├── nasals/                        # Nasal phonemes (m, n, ng)
│   ├── affricates/                    # Affricate phonemes (tʃ, dʒ)
│   ├── fricatives/                    # Fricative phonemes (f, v, θ, ð, s, z, ʃ, ʒ, h)
│   ├── approximants/                  # Approximant phonemes (w, l, r, j)
│   ├── diphthongs/                    # Diphthong phonemes
│   └── monophthongs/                  # Monophthong phonemes
│
├── ESC-50-master/                     # ESC-50 dataset
│   ├── audio/                         # Audio files
│   ├── meta/                          # Metadata (esc50.csv)
│   └── README.md                      # ESC-50 dataset documentation
│
└── [augmented audio files]           # Generated augmented audio samples
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/audio-data-augmentation.git
   cd audio-data-augmentation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download ESC-50 Dataset** (if not already present)
   - The ESC-50 dataset should be placed in the `ESC-50-master/` directory
   - Download from: https://github.com/karolpiczak/ESC-50

## 🚀 Quick Start

For a quick introduction, see [QUICKSTART.md](QUICKSTART.md) which includes:
- Installation steps
- Basic code examples
- Common issues and solutions
- Quick reference for common operations

## 💻 Usage

### Audio Augmentation

Open `Audio_Augmentation_Final.ipynb` in Jupyter Notebook to explore various augmentation techniques:

```python
# Load audio file
signal, sr = librosa.load("scale.wav", sr=None)

# Apply augmentations
augmented_signal = add_white_noise(signal, noise_factor=0.05)
pitch_shifted = pitch_shift(signal, sr, n_steps=4)
time_stretched = time_stretch(signal, rate=1.5)
```

### Environmental Sound Classification

The notebook includes a complete CNN pipeline for classifying environmental sounds:

1. **Data Loading**: Loads ESC-50 dataset (siren and car_horn classes)
2. **Augmentation**: Applies deterministic augmentations to increase dataset size
3. **Feature Extraction**: Extracts mel-spectrograms from audio
4. **Model Training**: Trains a 2D CNN with regularization
5. **Evaluation**: Provides accuracy metrics, confusion matrix, and classification reports

**Key Features:**
- Deterministic augmentations for reproducibility
- Grouped train/test split to prevent data leakage
- L2 regularization and dropout to prevent overfitting
- Early stopping for optimal training

### Phoneme Classification

Open `Phonemes.ipynb` to train a model for phoneme recognition:

1. **Data Collection**: Automatically collects phoneme samples from subdirectories
2. **Massive Augmentation**: Creates 100 augmented versions per phoneme
3. **CNN Training**: Trains a deep CNN for 44-class phoneme classification
4. **Prediction**: Tests on original phoneme files

**Phoneme Categories:**
- Plosives (6): p, t, k, b, d, g
- Nasals (3): m, n, ng
- Affricates (2): tʃ, dʒ
- Fricatives (9): f, v, θ, ð, s, z, ʃ, ʒ, h
- Approximants (4): w, l, r, j
- Diphthongs (8): Various diphthong sounds
- Monophthongs (12): Various vowel sounds

## 🎵 Augmentation Techniques

### 1. White Noise Addition
```python
def add_white_noise_snr(signal, snr_db=20):
    """Add white noise with specified SNR in dB"""
    signal_power = np.mean(signal**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise
```

### 2. Pitch Shifting
```python
def pitch_shift(signal, sr, n_steps=4):
    """Shift pitch by n semitones"""
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)
```

### 3. Time Stretching
```python
def time_stretch(signal, rate=1.5):
    """Stretch time by rate factor (1.5 = 50% faster)"""
    return librosa.effects.time_stretch(signal, rate=rate)
```

### 4. Environmental Noise
```python
def add_noise(signal, noise_file, snr_db=20):
    """Add environmental noise from file with specified SNR"""
    noise, sr_noise = librosa.load(noise_file, sr=None)
    # Scale noise based on SNR
    # ... (see notebook for full implementation)
```

### 5. SpecAugment
```python
def spec_augment(mel_spectrogram, time_mask_max=50, freq_mask_max=16):
    """Apply frequency and time masking to spectrogram"""
    # Frequency masking
    # Time masking
    # ... (see notebook for full implementation)
```

## 🤖 Models

### CNN Architecture for Sound Classification

```
Input: (128, T, 1) - Mel Spectrogram
  ↓
Conv2D(32 filters, 3x3) + BatchNorm + MaxPool
  ↓
Conv2D(64 filters, 3x3) + BatchNorm + MaxPool
  ↓
GlobalAveragePooling2D
  ↓
Dense(64) + Dropout(0.5)
  ↓
Dense(num_classes) + Softmax
```

### CNN Architecture for Phoneme Classification

```
Input: (128, T, 1) - Mel Spectrogram
  ↓
Conv2D(32 filters, 3x3) + BatchNorm + MaxPool
  ↓
Conv2D(64 filters, 3x3) + BatchNorm + MaxPool
  ↓
Conv2D(128 filters, 3x3) + BatchNorm + MaxPool
  ↓
GlobalAveragePooling2D
  ↓
Dense(128) + Dropout(0.5)
  ↓
Dense(44) + Softmax  # 44 phoneme classes
```

## 📊 Dataset

### ESC-50 Dataset
- **Source**: https://github.com/karolpiczak/ESC-50
- **Description**: 2000 environmental audio recordings
- **Classes Used**: siren, car_horn (40 samples each)
- **Sample Rate**: 22050 Hz
- **Format**: WAV files

### Phoneme Dataset
- **Source**: Custom phoneme recordings
- **Total Phonemes**: 44
- **Categories**: 7 phoneme categories
- **Format**: WAV files organized by category

## 📈 Results

### Environmental Sound Classification
- **Test Accuracy**: ~75-85% (varies by configuration)
- **Classes**: 2 (siren, car_horn)
- **Augmentation Factor**: 8x (1 original + 7 augmented versions)

### Phoneme Classification
- **Test Accuracy**: ~50-76% (varies by configuration)
- **Classes**: 44 phonemes
- **Augmentation Factor**: 100x per phoneme

*Note: Results may vary based on random seed, data split, and hyperparameters.*

## 🛠️ Technologies Used

- **Python 3.x**
- **Librosa**: Audio processing and feature extraction
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Visualization
- **scikit-learn**: Data preprocessing and evaluation metrics
- **SoundFile**: Audio I/O
- **Jupyter Notebook**: Interactive development

## 📝 Key Parameters

### Audio Processing
- **Sample Rate (SR)**: 22050 Hz
- **Mel Bands (N_MELS)**: 128
- **Frame Size**: 2048
- **Hop Size**: 512

### Model Training
- **Batch Size**: 16-32
- **Learning Rate**: 1e-4
- **Epochs**: 10-30 (with early stopping)
- **L2 Regularization**: 1e-4
- **Dropout**: 0.5

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ESC-50 Dataset**: Karol J. Piczak for creating the ESC-50 dataset
- **Librosa**: The librosa library for audio analysis
- **TensorFlow**: The TensorFlow team for the deep learning framework

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

