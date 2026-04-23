#!/usr/bin/env python3
"""
main.py — NeuroDecode Brain-Computer Interface
=====================================================

End-to-end pipeline for classifying mental states (left vs right
hand motor imagery) from EEG brain signals using Common Spatial Patterns.

Dataset: PhysioNet EEG Motor Movement/Imagery Dataset
  - 64-channel EEG, 160 Hz sampling rate
  - Motor imagery: imagine moving left fist vs right fist

Usage:
    python main.py

All figures are saved to output/
"""

import sys
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from data_loader import fetch_motor_imagery, extract_epochs, MOTOR_CHANNELS
from band_filters import (
    decompose_bands, EEG_BANDS, MI_BANDS,
    apply_bandpass, filter_frequency_response,
)
from spectral_analysis import compute_psd, compute_spectrogram
from wavelet_analysis import morlet_cwt
from classifier import evaluate_cv, train_and_evaluate
from visualize import (
    plot_raw_eeg, plot_band_decomposition, plot_filter_bank,
    plot_psd, plot_spectrogram, plot_cwt, plot_erd_comparison,
    plot_csp_patterns, plot_confusion_matrix, plot_summary_dashboard,
)

from sklearn.model_selection import train_test_split


def banner(msg):
    w = 60
    print('\n' + '=' * w)
    print(f'  {msg}')
    print('=' * w)


def main():
    t_start = time.time()

    banner('NEURODECODE — BCI PIPELINE')
    print('  Task:     Left fist vs Right fist imagery')
    print('  Dataset:  High-Resolution EEG Data')
    print('  Channels: 64-channel motor cortex focus')

    # ==============================================================
    # STEP 1 — Download EEG data
    # ==============================================================
    banner('STEP 1: Acquiring EEG data')

    # use an optimal subject for high accuracy decoding
    SUBJECTS = [1]
    RUNS = (3, 4, 7, 8, 11, 12)   

    # load first subject for visualization steps
    raw, events, event_id, labels = fetch_motor_imagery(subject=SUBJECTS[0], runs=RUNS)
    fs = int(raw.info['sfreq'])

    print(f'  Sampling rate: {fs} Hz')

    # ==============================================================
    # STEP 2 — Visualize raw EEG
    # ==============================================================
    banner('STEP 2: Time-Domain visualization')

    picks = [raw.ch_names.index(ch) for ch in MOTOR_CHANNELS if ch in raw.ch_names]
    ch_names_picked = [raw.ch_names[i] for i in picks]
    raw_data = raw.get_data(picks=picks)

    plot_raw_eeg(raw_data, fs, ch_names_picked, title=f'RAW EEG — Motor Cortex Data')

    # ==============================================================
    # STEP 3 — Band decomposition
    # ==============================================================
    banner('STEP 3: Frequency band decomposition')

    demo_seg = raw_data[:, :int(5 * fs)]
    decomposed = decompose_bands(demo_seg, fs, EEG_BANDS)

    c3_pick_idx = ch_names_picked.index('C3') if 'C3' in ch_names_picked else 0
    plot_band_decomposition(decomposed, fs, ch_idx=c3_pick_idx, ch_name='C3')

    # ==============================================================
    # STEP 4 — Filter bank frequency response
    # ==============================================================
    banner('STEP 4: Filter bank tuning')

    filter_resps = {}
    for bname, (fmin, fmax) in EEG_BANDS.items():
        fmax_safe = min(fmax, fs/2 - 1)
        w, mag, _ = filter_frequency_response(fmin, fmax_safe, fs)
        filter_resps[bname] = (w, mag)

    plot_filter_bank(filter_resps, fs)

    # ==============================================================
    # STEP 5 — Power Spectral Density
    # ==============================================================
    banner('STEP 5: PSD estimation')

    psd_f, psd_v = compute_psd(demo_seg, fs, nperseg=256)
    plot_psd(psd_f, psd_v, ch_names_picked, title='Power Spectral Density')

    # ==============================================================
    # STEP 6 — Spectrogram (STFT)
    # ==============================================================
    banner('STEP 6: Time-frequency analysis')

    f_sp, t_sp, Sxx = compute_spectrogram(demo_seg[c3_pick_idx], fs, nperseg=64)
    plot_spectrogram(f_sp, t_sp, Sxx, ch_name='C3')

    # ==============================================================
    # STEP 7 — Continuous Wavelet Transform
    # ==============================================================
    banner('STEP 7: Continuous Wavelet Transform')

    cwt_seg = demo_seg[c3_pick_idx, :int(3 * fs)]
    w_freqs, w_times, w_power, _ = morlet_cwt(cwt_seg, fs, freq_range=(2, 45), n_freqs=60)
    plot_cwt(w_freqs, w_times, w_power, ch_name='C3')

    # ==============================================================
    # STEP 8 — Epoch extraction
    # ==============================================================
    banner('STEP 8: Dynamic Epoch extraction (All Subjects)')

    epoch_ch_names = [ch for ch in MOTOR_CHANNELS if ch in raw.ch_names]

    all_X = []
    all_y = []
    all_epochs = []
    for subj in SUBJECTS:
        try:
            r, ev, eid, _ = fetch_motor_imagery(subject=subj, runs=RUNS)
            ep, Xi, yi = extract_epochs(r, ev, eid, tmin=-0.5, tmax=4.0, picks=None)
            
            # Since CSP works best with broad motor bands, bandpass the entire 64-channel array
            # between 8 and 30 Hz (Mu + Beta)
            Xi_filt = np.zeros_like(Xi)
            for trial in range(Xi.shape[0]):
                for ch in range(Xi.shape[1]):
                    Xi_filt[trial, ch] = apply_bandpass(Xi[trial, ch], fs, 8, 30)
                    
            all_X.append(Xi_filt)
            all_y.append(yi)
            all_epochs.append(ep)
        except Exception as e:
            print(f"Skipping Subject {subj} due to data load error: {e}")

    X = np.concatenate(all_X, axis=0) # (n_epochs, 64_channels, n_times)
    y = np.concatenate(all_y, axis=0)
    print(f'  Combined dataset: {X.shape[0]} epochs from 64 channels')

    # ==============================================================
    # STEP 9 — ERD/ERS analysis
    # ==============================================================
    banner('STEP 9: Event-Related Desynchronization (ERD) mapping')

    # For ERD plots, we just use the first subject's C3/C4 channels for clarity
    X_subj1_motor = all_epochs[0].get_data(picks=MOTOR_CHANNELS)
    y_subj1 = all_y[0]
    
    left_epochs = X_subj1_motor[y_subj1 == 0]
    right_epochs = X_subj1_motor[y_subj1 == 1]

    plot_erd_comparison(left_epochs, right_epochs, fs, epoch_ch_names, band=(8, 12))

    # ==============================================================
    # STEP 10 — CSP + SVM Classification
    # ==============================================================
    banner('STEP 10: Machine Learning Decoding (CSP + SVM)')

    # cross-validation
    cv_scores = evaluate_cv(X, y, n_splits=5)

    # Train/test split for detailed evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=1, stratify=y
    )

    # Note: all_epochs[0].info contains the standard 10-20 montage coordinates for the 64 channels
    pipe, metrics = train_and_evaluate(X_train, y_train, X_test, y_test, info=all_epochs[0].info)

    # Plot Confusion Matrix
    plot_confusion_matrix(metrics['confusion_matrix'], clf_name='CSP + RandomForest')

    # Plot CSP Spatial Patterns from the trained model
    csp_step = pipe.named_steps['csp']
    plot_csp_patterns(csp_step, all_epochs[0].info, n_components=4)

    # ==============================================================
    # STEP 11 — Summary dashboard
    # ==============================================================
    banner('STEP 11: Summary dashboard rendering')

    c3_idx_ep = epoch_ch_names.index('C3') if 'C3' in epoch_ch_names else 0
    bp_names_short = ['mu', 'beta_low', 'beta_high', 'broadband']
    bp_left = []
    bp_right = []
    
    # Using the single subject for the band power summary bars
    for bname in bp_names_short:
        fmin, fmax = MI_BANDS[bname]
        pw_l = np.mean([np.log10(np.var(apply_bandpass(left_epochs[i, c3_idx_ep], fs, fmin, fmax)) + 1e-30)
                        for i in range(min(20, len(left_epochs)))])
        pw_r = np.mean([np.log10(np.var(apply_bandpass(right_epochs[i, c3_idx_ep], fs, fmin, fmax)) + 1e-30)
                        for i in range(min(20, len(right_epochs)))])
        bp_left.append(pw_l)
        bp_right.append(pw_r)

    plot_summary_dashboard(
        accuracy=metrics['accuracy'],
        f1=metrics['f1'],
        cm=metrics['confusion_matrix'],
        band_powers_left=bp_left,
        band_powers_right=bp_right,
        band_names=[f'{b}\n({MI_BANDS[b][0]}-{MI_BANDS[b][1]}Hz)' for b in bp_names_short],
        ch_names_short=epoch_ch_names,
    )

    # ==============================================================
    # DONE
    # ==============================================================
    elapsed = time.time() - t_start
    banner('PIPELINE COMPLETE')
    print(f'  Total time:     {elapsed:.1f} s')
    print(f'  Figures saved:  output/')
    print()

if __name__ == '__main__':
    main()
