import warnings
warnings.filterwarnings('ignore')
import numpy as np
from data_loader import fetch_motor_imagery, extract_epochs
from band_filters import apply_bandpass
from mne.decoding import CSP
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

r, ev, eid, _ = fetch_motor_imagery(subject=1, runs=(3, 4, 7, 8, 11, 12))
ep, Xi, yi = extract_epochs(r, ev, eid, tmin=-0.5, tmax=4.0, picks=None)

fs = 160
Xi_filt = np.zeros_like(Xi)
for trial in range(Xi.shape[0]):
    for ch in range(Xi.shape[1]):
        Xi_filt[trial, ch] = apply_bandpass(Xi[trial, ch], fs, 8, 30)

for rs in range(30):
    X_train, X_test, y_train, y_test = train_test_split(Xi_filt, yi, test_size=0.15, random_state=rs, stratify=yi)
    csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    pipe = Pipeline([('csp', csp), ('clf', clf)])
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)
    print(f"RS={rs}, Acc={acc:.3f}")
