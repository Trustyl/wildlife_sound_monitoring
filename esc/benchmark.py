"""Nested fold evaluation: every audio clip gets one out-of-fold prediction."""
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import load_cache


def models(seed):
    return {
        'dummy': (DummyClassifier(strategy='most_frequent'), {}),
        'logistic': (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed)),
                     {'logisticregression__C': [0.1, 1.0, 10.0]}),
        'svm': (make_pipeline(StandardScaler(), SVC()),
                {'svc__C': [1.0, 10.0, 100.0], 'svc__gamma': ['scale', 0.001]}),
        'forest': (RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=seed),
                   {'max_features': ['sqrt', 0.5], 'min_samples_leaf': [1, 2]}),
        'extra_trees': (ExtraTreesClassifier(n_estimators=200, n_jobs=1, random_state=seed),
                       {'max_features': ['sqrt', 0.5], 'min_samples_leaf': [1, 2]}),
    }


def metrics(y, pred, classes):
    labels = list(range(len(classes)))
    return dict(accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
                confusion_matrix=confusion_matrix(y, pred, labels=labels).tolist(),
                per_class=classification_report(y, pred, labels=labels, target_names=classes,
                                                output_dict=True, zero_division=0))


def save_report(path, metadata, results, seed, protocol):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ['numpy', 'scikit-learn', 'librosa', 'torch']:
        try:
            packages[name] = version(name)
        except Exception:
            pass
    report = dict(protocol=protocol, seed=seed, python=platform.python_version(),
                  packages=packages, dataset=metadata, results=results)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def run(cache, output, selected=None, seed=42, jobs=1):
    data, metadata = load_cache(cache)
    X, y, folds = data['features'], data['labels'], data['folds']
    available = models(seed)
    selected = selected or list(available)
    if not set(selected) <= available.keys():
        raise ValueError('Unknown model requested')
    results = {}
    for name in selected:
        estimator, params = available[name]
        predictions = np.full(len(y), -1, dtype=int)
        fold_scores = []
        for fold in range(1, 6):
            train, test = folds != fold, folds == fold
            model = GridSearchCV(clone(estimator), params, scoring='f1_macro',
                                 cv=LeaveOneGroupOut(), n_jobs=jobs, error_score='raise')
            model.fit(X[train], y[train], groups=folds[train])
            predictions[test] = model.predict(X[test])
            score = metrics(y[test], predictions[test], metadata['classes'])
            fold_scores.append(dict(fold=fold, accuracy=score['accuracy'], macro_f1=score['macro_f1'],
                                    inner_macro_f1=float(model.best_score_), best_params=model.best_params_))
            print(f"{name} fold {fold}: accuracy={score['accuracy']:.3f}, macro-F1={score['macro_f1']:.3f}", flush=True)
        results[name] = dict(**metrics(y, predictions, metadata['classes']), folds=fold_scores,
                             fold_accuracy_std=float(np.std([f['accuracy'] for f in fold_scores], ddof=1)),
                             predictions=predictions.tolist())
    return save_report(output, metadata, results, seed,
                       'Five filename-defined outer folds; four inner group folds for tuning. Scaling is fit inside each training split.')
