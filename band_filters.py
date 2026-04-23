"""
band_filters.py
---------------
EEG frequency band decomposition using Butterworth IIR filters.

The brain produces electrical oscillations at characteristic frequencies,
each associated with different cognitive/motor states:

  Band        Freq (Hz)     Associated State
  ─────────   ──────────    ─────────────────────────────────
  Delta       0.5 – 4       Deep sleep, unconsciousness
  Theta       4 – 8         Drowsiness, meditation, memory
  Alpha/Mu    8 – 13        Relaxed wakefulness, motor idle
  Beta        13 – 30       Active thinking, focus, movement
  Gamma       30 – 100      Higher cognition, perception binding

For motor imagery BCI:
  - Mu (8-12 Hz) and Beta (13-30 Hz) are critical
  - Imagining left hand movement → ERD (power decrease) over RIGHT motor cortex (C4)
  - Imagining right hand movement → ERD over LEFT motor cortex (C3)
  - This contralateral suppression is our classification signal!

All filters are zero-phase (filtfilt) to preserve temporal structure.
"""

import numpy as np
from scipy import signal as sig


# ===================================================================
# EEG BAND DEFINITIONS
# ===================================================================
EEG_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 50.0),    # capped at 50 for 160 Hz data
}

# motor-imagery specific bands
MI_BANDS = {
    'mu':        (8.0, 12.0),     # mu rhythm (motor idle)
    'beta_low':  (13.0, 20.0),    # low beta
    'beta_high': (20.0, 30.0),    # high beta
    'broadband': (8.0, 30.0),     # full motor band
}


# ===================================================================
# FILTER DESIGN
# ===================================================================

def design_bandpass(fmin, fmax, fs, order=5):
    """
    Design a Butterworth bandpass filter.

    Returns second-order sections (SOS) for numerical stability.
    SOS form avoids the coefficient quantization issues of transfer
    function (b, a) representation at higher orders.
    """
    nyq = 0.5 * fs
    low = fmin / nyq
    high = min(fmax / nyq, 0.99)    # guard against Nyquist edge
    sos = sig.butter(order, [low, high], btype='bandpass', output='sos')
    return sos


def design_bandpass_ba(fmin, fmax, fs, order=4):
    """
    Design Butterworth bandpass as transfer function (b, a).
    Used for filter frequency response visualization.
    """
    nyq = 0.5 * fs
    low = fmin / nyq
    high = min(fmax / nyq, 0.99)
    b, a = sig.butter(order, [low, high], btype='bandpass')
    return b, a


def apply_bandpass(data, fs, fmin, fmax, order=5):
    """
    Apply zero-phase Butterworth bandpass filter.

    Uses SOS (second-order sections) + sosfiltfilt for:
      - Numerical stability at high filter orders
      - Zero phase distortion (forward-backward filtering)

    Parameters
    ----------
    data  : 1-D or 2-D array (channels x samples if 2-D)
    fs    : sampling rate
    fmin  : lower cutoff frequency
    fmax  : upper cutoff frequency
    order : filter order

    Returns
    -------
    filtered : same shape as data
    """
    sos = design_bandpass(fmin, fmax, fs, order)

    if data.ndim == 1:
        return sig.sosfiltfilt(sos, data)
    else:
        # filter each channel independently
        out = np.zeros_like(data)
        for ch in range(data.shape[0]):
            out[ch] = sig.sosfiltfilt(sos, data[ch])
        return out


def decompose_bands(data, fs, bands=None):
    """
    Decompose a signal into its constituent EEG frequency bands.

    This is essentially a filter bank: the signal is passed through
    multiple parallel bandpass filters, each extracting one frequency
    component. The sum of all bands approximates the original signal
    (though not exactly due to filter overlap).

    Parameters
    ----------
    data  : (n_channels, n_samples) or (n_samples,) array
    fs    : sampling rate
    bands : dict of {name: (fmin, fmax)}. Defaults to EEG_BANDS.

    Returns
    -------
    decomposed : dict of {band_name: filtered_data}
    """
    if bands is None:
        bands = EEG_BANDS

    decomposed = {}
    for name, (fmin, fmax) in bands.items():
        # skip bands above Nyquist
        if fmin >= fs / 2:
            continue
        fmax_actual = min(fmax, fs / 2 - 1)
        decomposed[name] = apply_bandpass(data, fs, fmin, fmax_actual)

    return decomposed


def compute_band_power(data, fs, fmin, fmax, method='welch'):
    """
    Compute average power in a specific frequency band.

    Two methods:
    1. 'welch'    — integrate PSD over band (frequency domain)
    2. 'variance' — variance of bandpass-filtered signal (Parseval's theorem)

    Both methods give equivalent results (Parseval's theorem links
    time-domain energy to frequency-domain energy).
    """
    if method == 'variance':
        filtered = apply_bandpass(data, fs, fmin, fmax)
        if data.ndim == 1:
            return np.var(filtered)
        return np.var(filtered, axis=-1)

    elif method == 'welch':
        if data.ndim == 1:
            freqs, psd = sig.welch(data, fs=fs, nperseg=min(256, len(data)),
                                   window='hann')
            mask = (freqs >= fmin) & (freqs <= fmax)
            return np.trapz(psd[mask], freqs[mask])
        else:
            powers = []
            for ch in range(data.shape[0]):
                freqs, psd = sig.welch(data[ch], fs=fs,
                                       nperseg=min(256, data.shape[-1]),
                                       window='hann')
                mask = (freqs >= fmin) & (freqs <= fmax)
                powers.append(np.trapz(psd[mask], freqs[mask]))
            return np.array(powers)


def filter_frequency_response(fmin, fmax, fs, order=5, n_points=2048):
    """
    Compute transfer function magnitude and phase for visualization.
    """
    b, a = design_bandpass_ba(fmin, fmax, fs, order=order)
    w, H = sig.freqz(b, a, worN=n_points, fs=fs)
    mag_db = 20 * np.log10(np.abs(H) + 1e-30)
    phase_deg = np.degrees(np.angle(H))
    return w, mag_db, phase_deg
