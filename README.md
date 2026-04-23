# NeuroDecode — Brain-Computer Interface

A professional-grade Brain-Computer Interface (BCI) system that decodes human intent (left fist vs right fist motor imagery) from real EEG brain signals using advanced digital signal processing and machine learning.

## What This Does

When a person imagines moving their left hand, their brain produces a measurable electrical pattern: the **mu rhythm (8-12 Hz)** over the right side of the brain decreases in power. This phenomenon, called **Event-Related Desynchronization (ERD)**, is the mechanism that powers this BCI classifier.

This project implements an end-to-end signal processing and machine learning pipeline to detect and classify these subtle brain patterns from 64-channel EEG recordings in real-time.

## The Technology Stack

| Concept | Application |
|---|---|
| **Common Spatial Pattern (CSP)** | Industry-standard spatial filtering to maximize variance between mental states |
| **Butterworth IIR Filtering** | Zero-phase bandpass filtering isolating neural frequency bands (δ, θ, α, β, γ) |
| **Power Spectral Density** | Noise-robust power estimation via windowed periodogram averaging |
| **Continuous Wavelet Transform** | High-resolution time-frequency analysis using the Morlet wavelet |
| **Short-Time Fourier Transform** | Spectrogram generation for temporal brain wave dynamics |
| **Support Vector Machine (SVM)** | High-dimensional classification of spatial features |

## Project Structure

```text
├── main.py                 # Core analysis & training pipeline
├── data_loader.py          # Data acquisition from PhysioNet
├── band_filters.py         # IIR digital filter design
├── spectral_analysis.py    # FFT, PSD, and STFT functionality
├── wavelet_analysis.py     # CWT with Morlet wavelets
├── classifier.py           # CSP feature extraction and SVM/LDA models
├── visualize.py            # Dashboard rendering and data visualization
├── app.py                  # Streamlit web application frontend
├── requirements.txt        # Dependencies
└── output/                 # Pre-computed visualization cache
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Re-train the model and generate figures (~5 minutes)
python main.py

# 3. Launch the dashboard UI
streamlit run app.py
```

## The Neuroscience

Motor imagery activates the exact same neural pathways as actual physical movement. When you imagine clenching your **left** hand:
1. Mu (8-12 Hz) and Beta (13-30 Hz) power **decreases** over the **right** motor cortex (electrode C4).
2. By comparing the spatial distribution of this desynchronization across 64 electrodes, the CSP algorithm trains a machine learning classifier to accurately decode the intended movement.

## Data Source

**High-Resolution Cognitive EEG Dataset**
- Human subjects, 64-channel EEG, 160 Hz sampling rate
- Validated real-world neural data
