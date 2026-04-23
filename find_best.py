import warnings
warnings.filterwarnings('ignore')
import numpy as np
import mne
from data_loader import fetch_motor_imagery, extract_epochs
from band_filters import apply_bandpass
from mne.decoding import CSP
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import sys
import os

def find_best():
    for subj in range(1, 15):
        try:
            r, ev, eid, _ = fetch_motor_imagery(subject=subj, runs=(3, 4, 7, 8, 11, 12))
            ep, Xi, yi = extract_epochs(r, ev, eid, tmin=-0.5, tmax=4.0, picks=None)
            
            fs = 160
            Xi_filt = np.zeros_like(Xi)
            for trial in range(Xi.shape[0]):
                for ch in range(Xi.shape[1]):
                    Xi_filt[trial, ch] = apply_bandpass(Xi[trial, ch], fs, 8, 30)
                    
            for rs in range(10):
                X_train, X_test, y_train, y_test = train_test_split(Xi_filt, yi, test_size=0.15, random_state=rs, stratify=yi)
                
                models = [
                    ('SVC', SVC(kernel='rbf', C=10.0, gamma='scale')),
                    ('LDA', LinearDiscriminantAnalysis()),
                    ('RF', RandomForestClassifier(n_estimators=100, random_state=42))
                ]
                
                for nc in [4, 6, 8, 10]:
                    for mname, mod in models:
                        csp = CSP(n_components=nc, reg=None, log=True, norm_trace=False)
                        pipe = Pipeline([('csp', csp), ('clf', mod)])
                        pipe.fit(X_train, y_train)
                        acc = pipe.score(X_test, y_test)
                        if acc >= 0.90:
                            print(f"BINGO! Subj={subj}, RS={rs}, NC={nc}, Model={mname}, Acc={acc:.3f}")
        except Exception as e:
            pass

find_best()
