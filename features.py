"""
features.py
-----------
Feature extraction pipeline for EEG motor imagery classification.

Features are computed from the frequency-band decomposed signals.
The key insight: during motor imagery of one hand, the mu/beta
rhythms DECREASE (ERD) over the contralateral motor cortex and
INCREASE (ERS) over the ipsilateral cortex.

So the features we extract are designed to capture this
hemispheric power asymmetry.

Feature categories:
  1. Band power (log-variance of bandpass-filtered signal)
  2. Power ratios (C3 vs C4 asymmetry)
  3. Spectral entropy per channel
  4. Hjorth parameters (activity, mobility, complexity)
"""

import numpy as np
from band_filters import apply_bandpass, compute_band_power, MI_BANDS
from spectral_analysis import spectral_entropy


# ===================================================================
# HJORTH PARAMETERS
# ===================================================================
# Hjorth (1970) defined three time-domain descriptors for EEG:
#   Activity    = variance of the signal (total power)
#   Mobility    = sqrt(var(dx/dt) / var(x))   (mean frequency)
#   Complexity  = mobility(dx/dt) / mobility(x) (bandwidth)

def hjorth_params(data):
    """
    Compute Hjorth parameters for a 1-D signal.

    These are pure time-domain features that capture spectral
    properties WITHOUT computing an FFT — elegant!
    """
    var_x = np.var(data)
    dx = np.diff(data)
    var_dx = np.var(dx)
    ddx = np.diff(dx)
    var_ddx = np.var(ddx)

    activity = var_x
    mobility = np.sqrt(var_dx / (var_x + 1e-30))
    complexity = np.sqrt(var_ddx / (var_dx + 1e-30)) / (mobility + 1e-30)

    return activity, mobility, complexity


# ===================================================================
# FEATURE EXTRACTION
# ===================================================================

def extract_features(X, fs, channel_names=None):
    """
    Extract a comprehensive feature vector from epoched EEG data.

    Parameters
    ----------
    X              : (n_epochs, n_channels, n_samples) array
    fs             : sampling rate
    channel_names  : list of channel name strings

    Returns
    -------
    features       : (n_epochs, n_features) array
    feature_names  : list of feature name strings
    """
    n_epochs, n_ch, n_times = X.shape

    if channel_names is None:
        channel_names = [f'ch{i}' for i in range(n_ch)]

    all_features = []
    all_names = []

    print(f"  [*] Extracting features from {n_epochs} epochs, {n_ch} channels ...")

    # ----- 1. Log band power (variance of filtered signal) -----
    # Parseval's theorem: variance in time ≈ integral of PSD
    for bname, (fmin, fmax) in MI_BANDS.items():
        for ci, cname in enumerate(channel_names):
            powers = []
            for ei in range(n_epochs):
                bp = apply_bandpass(X[ei, ci], fs, fmin, fmax)
                # log-variance = log10(var) — standard BCI feature
                powers.append(np.log10(np.var(bp) + 1e-30))

            all_features.append(powers)
            all_names.append(f'logvar_{bname}_{cname}')

    # ----- 2. Hemispheric asymmetry (C3 vs C4 power ratio) -----
    # This directly captures ERD lateralization
    c3_idx = None
    c4_idx = None
    for i, name in enumerate(channel_names):
        if name == 'C3':
            c3_idx = i
        elif name == 'C4':
            c4_idx = i

    if c3_idx is not None and c4_idx is not None:
        for bname, (fmin, fmax) in [('mu', (8, 12)), ('beta', (13, 30))]:
            ratios = []
            for ei in range(n_epochs):
                bp_c3 = apply_bandpass(X[ei, c3_idx], fs, fmin, fmax)
                bp_c4 = apply_bandpass(X[ei, c4_idx], fs, fmin, fmax)
                p_c3 = np.var(bp_c3) + 1e-30
                p_c4 = np.var(bp_c4) + 1e-30
                ratios.append(np.log10(p_c3 / p_c4))   # log ratio

            all_features.append(ratios)
            all_names.append(f'asym_{bname}_C3vC4')

    # ----- 3. Spectral entropy per channel -----
    for ci, cname in enumerate(channel_names):
        ents = []
        for ei in range(n_epochs):
            H = spectral_entropy(X[ei, ci], fs, nperseg=min(128, n_times))
            ents.append(H)
        all_features.append(ents)
        all_names.append(f'entropy_{cname}')

    # ----- 4. Hjorth parameters -----
    for ci, cname in enumerate(channel_names):
        act_list, mob_list, comp_list = [], [], []
        for ei in range(n_epochs):
            a, m, c = hjorth_params(X[ei, ci])
            act_list.append(np.log10(a + 1e-30))
            mob_list.append(m)
            comp_list.append(c)
        all_features.append(act_list)
        all_names.append(f'hjorth_act_{cname}')
        all_features.append(mob_list)
        all_names.append(f'hjorth_mob_{cname}')
        all_features.append(comp_list)
        all_names.append(f'hjorth_cmp_{cname}')

    # stack into (n_epochs, n_features) matrix
    features = np.array(all_features).T

    print(f"  [+] Extracted {features.shape[1]} features per epoch")

    return features, all_names
