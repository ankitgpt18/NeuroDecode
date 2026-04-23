import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mne.decoding import CSP

def build_csp_pipeline(n_components=6):
    """
    Build a classification pipeline using Common Spatial Patterns (CSP)
    and a Support Vector Machine (SVM).
    
    CSP finds spatial filters that maximize the variance (power) for one 
    class while minimizing it for the other. This is the gold standard
    for decoding oscillatory brain activity in Brain-Computer Interfaces.
    """
    from sklearn.ensemble import RandomForestClassifier
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    
    pipe = Pipeline([
        ('csp', csp),
        ('clf', clf)
    ])
    return pipe

def evaluate_cv(X, y, n_splits=5):
    """
    Evaluate the CSP+SVM pipeline using stratified k-fold cross-validation.
    X must be (n_epochs, n_channels, n_times).
    """
    print(f"\n  --- {n_splits}-fold Cross-Validation (CSP + SVM) ---")
    pipe = build_csp_pipeline()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    
    print(f"  Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    return scores

def train_and_evaluate(X_train, y_train, X_test, y_test, info):
    """
    Train and evaluate the CSP->SVM pipeline, returning metrics and
    model components. The 'info' object is passed to CSP to allow it
    to plot spatial topography later.
    """
    print(f"\n  --- Train/Test Split Evaluation ---")
    
    from sklearn.ensemble import RandomForestClassifier
    # We pass the MNE info to CSP so it knows channel locations
    csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    pipe = Pipeline([('csp', csp), ('clf', clf)])
    
    # Fit the pipeline
    pipe.fit(X_train, y_train)
    
    # Predict
    y_pred = pipe.predict(X_test)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"  Test Accuracy:  {acc:.3f}")
    print(f"  Test F1 Score:  {f1:.3f}")
    
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm
    }
    
    return pipe, metrics
