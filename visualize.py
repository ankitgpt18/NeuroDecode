"""
visualize.py
------------
Publication-quality visualizations for EEG analysis.
Dark theme with neuroscience-inspired color palette.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap


# ===================================================================
# DARK THEME + NEURO COLOR PALETTE
# ===================================================================
BG       = '#080812'
PANEL_BG = '#0e0e1c'
GRID_CLR = '#1a1a30'
TXT      = '#c8c8d8'
TICK     = '#808098'

PALETTE = {
    'left':      '#ff5566',     # left fist / C3  (warm red)
    'right':     '#44ddbb',     # right fist / C4 (teal)
    'alpha':     '#ff9933',     # alpha/mu band (orange)
    'beta':      '#6699ff',     # beta band (blue)
    'theta':     '#cc66ff',     # theta band (purple)
    'delta':     '#44cc44',     # delta band (green)
    'gamma':     '#ffcc33',     # gamma band (gold)
    'accent':    '#ff66aa',     # highlights
    'dim':       '#555577',     # muted elements
    'eeg1':      '#55aaff',     # generic EEG line 1
    'eeg2':      '#ff7755',     # generic EEG line 2
    'eeg3':      '#88dd55',     # generic EEG line 3
}

BAND_COLORS = {
    'delta': PALETTE['delta'],
    'theta': PALETTE['theta'],
    'alpha': PALETTE['alpha'],
    'mu':    PALETTE['alpha'],
    'beta':  PALETTE['beta'],
    'beta_low':  PALETTE['beta'],
    'beta_high': '#4477dd',
    'gamma': PALETTE['gamma'],
    'broadband': '#aaaacc',
}

def _apply_style():
    plt.rcParams.update({
        'figure.facecolor': BG,
        'axes.facecolor': PANEL_BG,
        'axes.edgecolor': '#333355',
        'axes.labelcolor': TXT,
        'axes.grid': True,
        'grid.color': GRID_CLR,
        'grid.alpha': 0.35,
        'text.color': TXT,
        'xtick.color': TICK,
        'ytick.color': TICK,
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
        'legend.facecolor': '#14142a',
        'legend.edgecolor': '#333355',
    })

_apply_style()

OUT_DIR = 'output'

def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f'{name}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    >> {path}")
    return path


# ===================================================================
# FIGURE 1 — Raw EEG traces
# ===================================================================
def plot_raw_eeg(data, fs, ch_names, title='Raw EEG', duration=5.0):
    n_ch = min(data.shape[0], 9)
    n_samp = int(duration * fs)
    t = np.arange(n_samp) / fs

    fig, axes = plt.subplots(n_ch, 1, figsize=(14, max(6, n_ch * 0.9)),
                             sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold', color='#ffffff', y=0.98)

    if n_ch == 1:
        axes = [axes]

    colors = [PALETTE['eeg1'], PALETTE['eeg2'], PALETTE['eeg3'],
              PALETTE['alpha'], PALETTE['beta'], PALETTE['theta'],
              PALETTE['left'], PALETTE['right'], PALETTE['gamma']]

    for i in range(n_ch):
        seg = data[i, :n_samp]
        clr = colors[i % len(colors)]
        axes[i].plot(t, seg * 1e6, color=clr, lw=0.4, alpha=0.85)
        axes[i].set_ylabel(f'{ch_names[i]}', fontsize=8, rotation=0,
                          labelpad=30, va='center')
        axes[i].set_yticks([])

    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, '01_raw_eeg')


# ===================================================================
# FIGURE 2 — Band decomposition
# ===================================================================
def plot_band_decomposition(decomposed, fs, ch_idx=0, ch_name='C3',
                            duration=3.0):
    n_bands = len(decomposed)
    n_samp = int(duration * fs)
    t = np.arange(n_samp) / fs

    fig, axes = plt.subplots(n_bands + 1, 1, figsize=(14, max(7, (n_bands+1) * 1.2)),
                             sharex=True)
    fig.suptitle(f'EEG BAND DECOMPOSITION — {ch_name}',
                 fontsize=13, fontweight='bold', color='#ffffff')

    # original (sum of bands as proxy)
    total = sum(b[ch_idx, :n_samp] if b.ndim > 1 else b[:n_samp]
                for b in decomposed.values())
    axes[0].plot(t, total * 1e6, color='#aaaacc', lw=0.4)
    axes[0].set_ylabel('Raw', fontsize=8, rotation=0, labelpad=25, va='center')
    axes[0].set_title('Reconstructed from sub-bands', fontsize=9, loc='left',
                      color='#888899')

    for i, (bname, bdata) in enumerate(decomposed.items()):
        ax = axes[i + 1]
        seg = bdata[ch_idx, :n_samp] if bdata.ndim > 1 else bdata[:n_samp]
        clr = BAND_COLORS.get(bname, '#aaaacc')
        ax.plot(t, seg * 1e6, color=clr, lw=0.5)
        ax.set_ylabel(bname.capitalize(), fontsize=8, rotation=0,
                      labelpad=25, va='center', color=clr)

    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    return _save(fig, '02_band_decomposition')


# ===================================================================
# FIGURE 3 — Filter frequency responses
# ===================================================================
def plot_filter_bank(filter_responses, fs):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle('EEG BANDPASS FILTER BANK — Frequency Responses',
                 fontsize=13, fontweight='bold', color='#ffffff')

    for bname, (freqs, mag_db) in filter_responses.items():
        clr = BAND_COLORS.get(bname, '#aaaacc')
        ax.plot(freqs, mag_db, color=clr, lw=1.5, alpha=0.85, label=bname.capitalize())

    ax.axhline(-3, color=PALETTE['accent'], ls='--', lw=0.7, alpha=0.4, label='−3 dB')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_xlim(0, 60)
    ax.set_ylim(-60, 5)
    ax.legend(ncol=3)
    fig.tight_layout()
    return _save(fig, '03_filter_bank')


# ===================================================================
# FIGURE 4 — Power Spectral Density
# ===================================================================
def plot_psd(freqs, psd, ch_names, title='PSD (Welch Method)'):
    n_ch = psd.shape[0] if psd.ndim > 1 else 1
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold', color='#ffffff')

    colors = list(PALETTE.values())
    if psd.ndim == 1:
        ax.semilogy(freqs, psd, color=colors[0], lw=1.0)
    else:
        for i in range(min(n_ch, 9)):
            ax.semilogy(freqs, psd[i], color=colors[i % len(colors)],
                       lw=0.8, alpha=0.8, label=ch_names[i])
        ax.legend(loc='upper right')

    # shade EEG bands
    bands_shade = [('δ', 0.5, 4), ('θ', 4, 8), ('α/μ', 8, 13), ('β', 13, 30), ('γ', 30, 50)]
    shade_colors = [PALETTE['delta'], PALETTE['theta'], PALETTE['alpha'],
                    PALETTE['beta'], PALETTE['gamma']]
    for (lbl, f1, f2), sc in zip(bands_shade, shade_colors):
        ax.axvspan(f1, f2, alpha=0.06, color=sc)
        ax.text((f1+f2)/2, ax.get_ylim()[1] * 0.5, lbl, ha='center',
                fontsize=8, color=sc, alpha=0.7)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (V²/Hz)')
    ax.set_xlim(0, 55)
    fig.tight_layout()
    return _save(fig, '04_psd')


# ===================================================================
# FIGURE 5 — Spectrogram
# ===================================================================
def plot_spectrogram(f, t, Sxx, ch_name='C3', event_time=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f'SPECTROGRAM (STFT) — {ch_name}',
                 fontsize=13, fontweight='bold', color='#ffffff')

    Sxx_db = 10 * np.log10(Sxx + 1e-30)
    vmin = np.percentile(Sxx_db, 5)
    vmax = np.percentile(Sxx_db, 97)

    pcm = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='inferno',
                        vmin=vmin, vmax=vmax)

    if event_time is not None:
        ax.axvline(event_time, color=PALETTE['accent'], ls='--', lw=1,
                   alpha=0.7, label='Event onset')
        ax.legend()

    ax.set_ylim(0, 50)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label('Power (dB)')
    fig.tight_layout()
    return _save(fig, f'05_spectrogram_{ch_name}')


# ===================================================================
# FIGURE 6 — CWT time-frequency map
# ===================================================================
def plot_cwt(freqs, times, power, ch_name='C3'):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f'CONTINUOUS WAVELET TRANSFORM — {ch_name}',
                 fontsize=13, fontweight='bold', color='#ffffff')

    pw_db = 10 * np.log10(power + 1e-30)
    vmin = np.percentile(pw_db, 5)
    vmax = np.percentile(pw_db, 97)

    pcm = ax.pcolormesh(times, freqs, pw_db, shading='gouraud',
                        cmap='magma', vmin=vmin, vmax=vmax)

    # mark key bands
    for bname, f_line, clr in [('μ', 10, PALETTE['alpha']),
                                ('β', 20, PALETTE['beta'])]:
        ax.axhline(f_line, color=clr, ls=':', lw=0.7, alpha=0.5)
        ax.text(times[-1] * 1.01, f_line, bname, color=clr, fontsize=9, va='center')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim(1, 45)
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label('Power (dB)')
    fig.tight_layout()
    return _save(fig, f'06_cwt_{ch_name}')


# ===================================================================
# FIGURE 7 — ERD/ERS: Left vs Right comparison
# ===================================================================
def plot_erd_comparison(epochs_left, epochs_right, fs, ch_names,
                        band=(8, 12), band_name='Mu'):
    from band_filters import apply_bandpass

    n_times = epochs_left.shape[-1]
    t = np.arange(n_times) / fs - 0.5   # assume tmin=-0.5

    # find C3 and C4 indices
    c3_i = ch_names.index('C3') if 'C3' in ch_names else 0
    c4_i = ch_names.index('C4') if 'C4' in ch_names else min(2, len(ch_names)-1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'EVENT-RELATED DESYNCHRONIZATION — {band_name} Band ({band[0]}-{band[1]} Hz)',
                 fontsize=13, fontweight='bold', color='#ffffff', y=0.98)

    configs = [
        (0, 0, epochs_left,  c3_i, 'Left Imagery → C3 (left cortex)', PALETTE['left']),
        (0, 1, epochs_left,  c4_i, 'Left Imagery → C4 (right cortex)', PALETTE['left']),
        (1, 0, epochs_right, c3_i, 'Right Imagery → C3 (left cortex)', PALETTE['right']),
        (1, 1, epochs_right, c4_i, 'Right Imagery → C4 (right cortex)', PALETTE['right']),
    ]

    for row, col, epochs, ch_i, title, clr in configs:
        ax = axes[row][col]

        # compute band power timecourse for each trial
        powers = []
        for trial in range(epochs.shape[0]):
            bp = apply_bandpass(epochs[trial, ch_i], fs, band[0], band[1])
            # hilbert envelope for instantaneous power
            from scipy.signal import hilbert
            analytic = hilbert(bp)
            inst_power = np.abs(analytic) ** 2
            powers.append(inst_power)

        powers = np.array(powers)
        mean_pw = np.mean(powers, axis=0)
        std_pw = np.std(powers, axis=0) / np.sqrt(len(powers))

        # normalize to baseline (-0.5 to 0 s)
        bl_end = int(0.5 * fs)
        baseline = np.mean(mean_pw[:bl_end])
        erd = (mean_pw - baseline) / baseline * 100  # percent change

        ax.plot(t, erd, color=clr, lw=1.0)
        ax.fill_between(t, erd - std_pw/baseline*100,
                        erd + std_pw/baseline*100,
                        color=clr, alpha=0.15)
        ax.axhline(0, color=PALETTE['dim'], ls='--', lw=0.5)
        ax.axvline(0, color='#ffffff', ls=':', lw=0.5, alpha=0.3)
        ax.set_title(title, fontsize=9, loc='left', color=clr)
        ax.set_ylabel('% Change')
        ax.set_xlabel('Time (s)')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, '07_erd_comparison')


# ===================================================================
# FIGURE 8 — CSP Spatial Patterns
# ===================================================================
def plot_csp_patterns(csp, info, n_components=4):
    """
    Plot the spatial topographies (patterns) learned by CSP.
    These show which brain regions are activated to discriminate
    between Left and Right motor imagery.
    """
    fig, axes = plt.subplots(1, n_components, figsize=(16, 4))
    fig.suptitle('COMMON SPATIAL PATTERNS (CSP) — Spatial Filters',
                 fontsize=14, fontweight='bold', color='#ffffff', y=1.05)

    # MNE's plot_patterns handles the topo plotting natively
    # We pass our axes to capture it in our figure
    csp.plot_patterns(info, components=np.arange(n_components),
                      axes=axes, show=False, colorbar=False,
                      cmap='inferno')

    # Force dark background on the MNE topomap plots
    for idx, ax in enumerate(axes):
        ax.set_facecolor(PANEL_BG)
        ax.set_title(f"Spatial Filter {idx+1}", color=TXT, fontsize=11)

    fig.tight_layout()
    return _save(fig, '08_csp_patterns')


# ===================================================================
# FIGURE 9 — Confusion Matrix
# ===================================================================
def plot_confusion_matrix(cm, labels=['Left Fist', 'Right Fist'],
                          clf_name='CSP + SVM'):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(f'CONFUSION MATRIX — {clf_name}',
                 fontsize=13, fontweight='bold', color='#ffffff')

    im = ax.imshow(cm, cmap='Blues', aspect='auto')

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(j, i, str(val), ha='center', va='center',
                   fontsize=18, fontweight='bold',
                   color='#ffffff' if val > cm.max()/2 else '#333333')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save(fig, '09_confusion_matrix')


# ===================================================================
# FIGURE 11 — Summary Dashboard
# ===================================================================
def plot_summary_dashboard(accuracy, f1, cm, band_powers_left, band_powers_right,
                            band_names, ch_names_short):
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    fig.suptitle(
        'EEG MOTOR IMAGERY CLASSIFICATION — Summary Dashboard',
        fontsize=15, fontweight='bold', color='#ffffff', y=0.98
    )

    # (0,0) Accuracy gauge
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.pie([accuracy, 1-accuracy],
            colors=[PALETTE['right'], '#222244'],
            startangle=90, counterclock=False,
            wedgeprops=dict(width=0.3))
    ax1.text(0, 0, f'{accuracy*100:.1f}%', ha='center', va='center',
            fontsize=24, fontweight='bold', color='#ffffff')
    ax1.set_title('Classification Accuracy', fontsize=11, color=TXT)

    # (0,1) Confusion matrix
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm[i,j]), ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='#ffffff' if cm[i,j] > cm.max()/2 else '#333333')
    ax2.set_xticks([0,1])
    ax2.set_yticks([0,1])
    ax2.set_xticklabels(['Left', 'Right'])
    ax2.set_yticklabels(['Left', 'Right'])
    ax2.set_title('Confusion Matrix', fontsize=11, color=TXT)

    # (0,2) Metrics
    ax3 = fig.add_subplot(gs[0, 2])
    metrics_text = (
        f"Accuracy:  {accuracy:.3f}\n"
        f"F1 Score:  {f1:.3f}\n\n"
        f"Data:      High-Res 64-Ch EEG\n"
        f"Filter:    Bandpass (8-30 Hz)\n"
        f"Featurizer: Common Spatial" + "\n" + "            Patterns (CSP)\n"
        f"Classifier: Random Forest" + "\n" + "            Engine"
    )
    ax3.text(0.1, 0.5, metrics_text, transform=ax3.transAxes,
            fontsize=11, color=TXT, va='center', family='monospace')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Pipeline Summary', fontsize=11, color=TXT)

    # (1, 0:2) Band power comparison
    ax4 = fig.add_subplot(gs[1, :])
    x = np.arange(len(band_names))
    w = 0.35
    bars1 = ax4.bar(x - w/2, band_powers_left, w, label='Left Imagery',
                    color=PALETTE['left'], alpha=0.7)
    bars2 = ax4.bar(x + w/2, band_powers_right, w, label='Right Imagery',
                    color=PALETTE['right'], alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(band_names, fontsize=9)
    ax4.set_ylabel('Log Band Power')
    ax4.set_title('Band Power: Left vs Right Motor Imagery (C3 channel)', fontsize=11, color=TXT)
    ax4.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, '11_summary_dashboard')
