"""
data_loader.py
--------------
Downloads and loads EEG Motor Movement / Imagery data from PhysioNet
via MNE-Python's built-in fetcher.

Dataset: PhysioNet EEGBCI (EEG Motor Movement/Imagery Dataset)
  - 109 subjects, 64-channel EEG at 160 Hz
  - Motor imagery tasks: imagining left fist vs right fist motion
  - The brain produces measurable Event-Related Desynchronization (ERD)
    in the mu (8-12 Hz) and beta (13-30 Hz) bands over motor cortex

Run mapping for motor imagery:
  Runs 4, 8, 12  →  left fist (T1) vs right fist (T2) imagery

Reference: Schalk et al., BCI2000, IEEE TBME 51(6), 2004
"""

import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

# suppress MNE info spam
mne.set_log_level('WARNING')

# channels of interest: motor cortex region
# C3 = left motor cortex, C4 = right motor cortex, Cz = midline
MOTOR_CHANNELS = ['C3', 'Cz', 'C4', 'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2']


def fetch_motor_imagery(subject=1, runs=(4, 8, 12)):
    """
    Download and load motor imagery EEG recordings.

    Parameters
    ----------
    subject : int (1-109)
    runs    : tuple of run numbers for left/right fist imagery

    Returns
    -------
    raw     : mne.io.Raw object with EEG data
    events  : (n_events, 3) array — [sample, 0, event_id]
    labels  : dict mapping event_id → label string
    """
    print(f"  [*] Fetching Subject S{subject:03d}, runs {runs} ...")

    # download EDF files from PhysioNet (cached after first run)
    paths = eegbci.load_data(subject, runs, path=None)

    # read and concatenate all runs
    raws = [read_raw_edf(p, preload=True) for p in paths]

    # standardize channel naming (PhysioNet uses non-standard names)
    for r in raws:
        eegbci.standardize(r)

    raw = concatenate_raws(raws)

    # set standard 10-20 montage for channel positions
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore')

    # extract event markers from annotations
    events, event_id = mne.events_from_annotations(raw)

    # the annotations map to:
    # T0 = rest, T1 = left fist, T2 = right fist
    labels = {}
    for name, eid in event_id.items():
        if 'T0' in name:
            labels[eid] = 'rest'
        elif 'T1' in name:
            labels[eid] = 'left_fist'
        elif 'T2' in name:
            labels[eid] = 'right_fist'
        else:
            labels[eid] = name

    fs = int(raw.info['sfreq'])
    n_ch = len(raw.ch_names)
    dur = raw.times[-1]

    print(f"  [+] Loaded: {n_ch} channels, {fs} Hz, {dur:.1f}s total")
    print(f"  [+] Events: {dict((labels.get(eid, name), np.sum(events[:, 2] == eid)) for name, eid in event_id.items())}")

    return raw, events, event_id, labels


def extract_epochs(raw, events, event_id, tmin=-0.5, tmax=4.0,
                   picks=None, baseline=(None, 0)):
    """
    Cut the continuous EEG into time-locked epochs around each event.

    Parameters
    ----------
    raw      : mne.io.Raw
    events   : events array from mne
    event_id : dict of event_name → event_code
    tmin     : start time relative to event onset (seconds)
    tmax     : end time relative to event onset (seconds)
    picks    : list of channel names to keep (None = all)
    baseline : tuple for baseline correction

    Returns
    -------
    epochs : mne.Epochs object
    X      : (n_epochs, n_channels, n_samples) ndarray
    y      : (n_epochs,) labels array
    """
    if picks is None:
        picks = MOTOR_CHANNELS

    # only keep left (T1) vs right (T2) events
    imagery_ids = {}
    for name, eid in event_id.items():
        if 'T1' in name:
            imagery_ids['left_fist'] = eid
        elif 'T2' in name:
            imagery_ids['right_fist'] = eid

    if not imagery_ids:
        raise ValueError("No T1/T2 events found in event_id")

    epochs = mne.Epochs(
        raw, events, event_id=imagery_ids,
        tmin=tmin, tmax=tmax,
        picks=picks,
        baseline=baseline,
        preload=True,
        verbose=False,
    )

    X = epochs.get_data()                  # (n_epochs, n_ch, n_times)
    y = epochs.events[:, 2]                # event codes

    # remap codes to 0/1
    left_code = imagery_ids['left_fist']
    y_binary = np.where(y == left_code, 0, 1)  # 0=left, 1=right

    n_left = np.sum(y_binary == 0)
    n_right = np.sum(y_binary == 1)
    print(f"  [+] Epochs: {len(y_binary)} total ({n_left} left, {n_right} right)")
    print(f"  [+] Shape:  {X.shape}  (epochs x channels x samples)")

    return epochs, X, y_binary


if __name__ == '__main__':
    raw, events, event_id, labels = fetch_motor_imagery(subject=1)
    epochs, X, y = extract_epochs(raw, events, event_id)
    print(f"\n  Data ready: X={X.shape}, y={y.shape}")
