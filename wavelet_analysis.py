"""
wavelet_analysis.py
-------------------
Continuous Wavelet Transform (CWT) analysis for EEG signals.

Why wavelets instead of just FFT?
  - FFT gives frequency content but LOSES time information
  - STFT gives time-frequency but with FIXED resolution
  - CWT gives time-frequency with ADAPTIVE resolution:
      * Good time resolution at high frequencies
      * Good frequency resolution at low frequencies
  - This matches how EEG events work: fast transients (spikes)
    need fine time resolution, while slow oscillations (alpha)
    need fine frequency resolution.

We use the Morlet wavelet — a Gaussian-windowed complex sinusoid.
It's the standard choice for oscillatory EEG analysis because
its shape resembles natural neural oscillations.

Reference: Tallon-Baudry & Bertrand, TICS (1999)
"""

import numpy as np
import pywt


# ===================================================================
# CONTINUOUS WAVELET TRANSFORM (CWT)
# ===================================================================

def morlet_cwt(data, fs, freq_range=(1, 50), n_freqs=50, wavelet='cmor1.5-1.0'):
    """
    Compute CWT using complex Morlet wavelet.

    The Morlet wavelet is:
        psi(t) = pi^(-1/4) * exp(j*w0*t) * exp(-t^2/2)

    where w0 controls the tradeoff between time and frequency resolution.

    Parameters
    ----------
    data       : 1-D signal array
    fs         : sampling rate (Hz)
    freq_range : (fmin, fmax) frequency range to analyze
    n_freqs    : number of frequency bins
    wavelet    : PyWavelets wavelet name ('cmor{bandwidth}-{center}')

    Returns
    -------
    freqs      : array of frequencies analyzed
    times      : time axis (seconds)
    power      : |CWT|^2 power matrix (n_freqs x n_times)
    coeffs     : complex CWT coefficients
    """
    fmin, fmax = freq_range
    freqs = np.linspace(fmin, fmax, n_freqs)

    # CWT scales: scale = (center_freq * fs) / freq
    # for cmor1.5-1.0, center frequency = 1.0 Hz
    center_freq = pywt.central_frequency(wavelet)
    scales = (center_freq * fs) / freqs

    # compute CWT
    coeffs, returned_freqs = pywt.cwt(data, scales, wavelet,
                                       sampling_period=1.0/fs)

    # power = |coefficients|^2
    power = np.abs(coeffs) ** 2

    # time axis
    times = np.arange(len(data)) / fs

    return freqs, times, power, coeffs


def cwt_multichannel(data, fs, freq_range=(1, 50), n_freqs=50):
    """
    CWT for multi-channel EEG data.

    Parameters
    ----------
    data : (n_channels, n_samples) array

    Returns
    -------
    freqs : frequency axis
    times : time axis
    power : (n_channels, n_freqs, n_times) array
    """
    n_ch = data.shape[0]
    freqs, times, pw0, _ = morlet_cwt(data[0], fs, freq_range, n_freqs)
    power = np.zeros((n_ch, pw0.shape[0], pw0.shape[1]))
    power[0] = pw0

    for ch in range(1, n_ch):
        _, _, pw, _ = morlet_cwt(data[ch], fs, freq_range, n_freqs)
        power[ch] = pw

    return freqs, times, power


# ===================================================================
# WAVELET-BASED FEATURES
# ===================================================================

def wavelet_band_power(data, fs, band=(8, 13), n_freqs=50):
    """
    Extract average power in a frequency band from CWT.

    Unlike Welch PSD (which gives a global average), CWT band power
    gives a TIME-RESOLVED power estimate — you can see exactly
    when the mu rhythm suppresses during motor imagery.
    """
    freqs, times, power, _ = morlet_cwt(data, fs, freq_range=(1, 50),
                                         n_freqs=n_freqs)

    # select frequency band
    mask = (freqs >= band[0]) & (freqs <= band[1])

    # average power across frequencies in the band → time course
    band_power_ts = np.mean(power[mask, :], axis=0)

    return times, band_power_ts


def wavelet_decompose_epochs(epochs_data, fs, bands=None, n_freqs=40):
    """
    Extract wavelet-based band power features from epoched EEG.

    For each epoch and channel, compute average CWT power in each band.
    This produces a feature vector for classification.

    Parameters
    ----------
    epochs_data : (n_epochs, n_channels, n_samples) array
    fs          : sampling rate
    bands       : dict of {name: (fmin, fmax)}

    Returns
    -------
    features : (n_epochs, n_channels * n_bands) feature matrix
    feat_names : list of feature names
    """
    if bands is None:
        bands = {
            'mu':   (8, 12),
            'beta': (13, 30),
        }

    n_epochs, n_ch, n_samples = epochs_data.shape
    n_bands = len(bands)
    features = np.zeros((n_epochs, n_ch * n_bands))
    feat_names = []

    for bi, (bname, (fmin, fmax)) in enumerate(bands.items()):
        for ci in range(n_ch):
            col = bi * n_ch + ci
            feat_names.append(f'{bname}_ch{ci}')

            for ei in range(n_epochs):
                freqs, times, power, _ = morlet_cwt(
                    epochs_data[ei, ci], fs,
                    freq_range=(max(1, fmin-2), fmax+2),
                    n_freqs=max(10, int((fmax - fmin) * 2)),
                )
                mask = (freqs >= fmin) & (freqs <= fmax)
                features[ei, col] = np.mean(power[mask, :])

    return features, feat_names
