"""
spectral_analysis.py
--------------------
Frequency-domain analysis of EEG signals.

Key operations:
  1. FFT — raw spectral content
  2. PSD via Welch's method — smooth, reliable power estimates
  3. Spectral entropy — measure of signal complexity/irregularity
  4. Dominant frequency — peak frequency in a band

The frequency content of EEG reveals the brain's oscillatory state.
During motor imagery, the mu rhythm (8-12 Hz) over the motor cortex
undergoes Event-Related Desynchronization (ERD) — a measurable
power decrease contralateral to the imagined movement.
"""

import numpy as np
from scipy import signal as sig


def compute_fft(data, fs):
    """
    Compute single-sided amplitude spectrum via FFT.

    Returns
    -------
    freqs : frequency axis (Hz)
    mag   : |X(f)| magnitude spectrum (normalized)
    phase : phase spectrum (radians)
    """
    N = len(data) if data.ndim == 1 else data.shape[-1]
    X = np.fft.rfft(data, axis=-1)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)

    # normalize by N for single-sided spectrum
    mag = 2.0 * np.abs(X) / N
    phase = np.angle(X)

    return freqs, mag, phase


def compute_psd(data, fs, nperseg=256):
    """
    Power Spectral Density estimation using Welch's method.

    Welch's method:
      1. Split signal into overlapping segments
      2. Apply window (Hann) to each segment
      3. Compute |FFT|^2 for each segment
      4. Average across segments → smooth PSD estimate

    Parameters
    ----------
    data    : 1-D signal or (n_channels, n_samples) array
    fs      : sampling rate
    nperseg : segment length for Welch's method

    Returns
    -------
    freqs : frequency axis
    psd   : power spectral density (V^2/Hz)
    """
    nperseg = min(nperseg, data.shape[-1])

    if data.ndim == 1:
        freqs, psd = sig.welch(data, fs=fs, nperseg=nperseg,
                               window='hann', noverlap=nperseg//2,
                               scaling='density')
    else:
        # compute PSD for each channel
        n_ch = data.shape[0]
        freqs, psd_ch0 = sig.welch(data[0], fs=fs, nperseg=nperseg,
                                   window='hann', noverlap=nperseg//2,
                                   scaling='density')
        psd = np.zeros((n_ch, len(freqs)))
        psd[0] = psd_ch0
        for ch in range(1, n_ch):
            _, psd[ch] = sig.welch(data[ch], fs=fs, nperseg=nperseg,
                                  window='hann', noverlap=nperseg//2,
                                  scaling='density')

    return freqs, psd


def spectral_entropy(data, fs, nperseg=256, normalize=True):
    """
    Compute spectral entropy — a measure of signal complexity.

    Spectral entropy quantifies how "spread out" the power is across
    frequencies. A pure sine wave has very low entropy (power at one
    frequency). White noise has maximum entropy (equal power everywhere).

    H = -sum(P_norm * log2(P_norm))

    For EEG: higher entropy = more irregular brain activity.

    Returns
    -------
    H : float or array of spectral entropy values
    """
    freqs, psd = compute_psd(data, fs, nperseg)

    def _entropy_1d(p):
        p_norm = p / (np.sum(p) + 1e-30)
        p_norm = p_norm[p_norm > 0]
        H = -np.sum(p_norm * np.log2(p_norm))
        if normalize:
            H /= np.log2(len(p_norm))
        return H

    if psd.ndim == 1:
        return _entropy_1d(psd)
    else:
        return np.array([_entropy_1d(psd[ch]) for ch in range(psd.shape[0])])


def dominant_frequency(data, fs, fmin=0.5, fmax=50.0, nperseg=256):
    """
    Find the frequency with maximum power within a band.
    Useful for tracking peak alpha/beta frequency.
    """
    freqs, psd = compute_psd(data, fs, nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)

    if psd.ndim == 1:
        idx = np.argmax(psd[mask])
        return freqs[mask][idx]
    else:
        dom = []
        for ch in range(psd.shape[0]):
            idx = np.argmax(psd[ch][mask])
            dom.append(freqs[mask][idx])
        return np.array(dom)


def compute_spectrogram(data, fs, nperseg=64, noverlap=None):
    """
    Short-Time Fourier Transform (STFT) for time-frequency analysis.

    The spectrogram shows how the frequency content of the EEG
    changes over time — crucial for seeing ERD/ERS events
    that happen during motor imagery.
    """
    if noverlap is None:
        noverlap = nperseg * 3 // 4    # 75% overlap

    nperseg = min(nperseg, data.shape[-1])

    f, t, Sxx = sig.spectrogram(
        data, fs=fs, nperseg=nperseg,
        noverlap=noverlap, window='hann',
    )
    return f, t, Sxx
