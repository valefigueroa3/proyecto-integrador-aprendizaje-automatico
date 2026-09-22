from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from joblib import parallel_backend
from scipy import sparse
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts" / "sklearn"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
RESULTS_DIR = PROJECT_DIR / "results" / "sklearn"
for directory in (MODELS_DIR, PREDICTIONS_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_data():
    X_train = sparse.load_npz(ARTIFACTS_DIR / "X_train.npz").astype(np.float32)
    X_test = sparse.load_npz(ARTIFACTS_DIR / "X_test.npz").astype(np.float32)
    y_train = np.load(ARTIFACTS_DIR / "y_train.npy")
    y_test = np.load(ARTIFACTS_DIR / "y_test.npy")
    id_test = np.load(ARTIFACTS_DIR / "id_test.npy", allow_pickle=True)
    return X_train, X_test, y_train, y_test, id_test


def model_specs(n_train: int):
    reg_params = [1e-6, 1e-4]
    c_values = [1.0 / (reg_param * n_train) for reg_param in reg_params]
    return {
        "logistic_regression": (
            LogisticRegression(
                l1_ratio=0,
                solver="lbfgs",
                max_iter=150,
                random_state=RANDOM_STATE,
                tol=1e-4,
            ),
            {"C": c_values},
            False,
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"max_depth": [5, 10, 15]},
            False,
        ),
        "random_forest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
                max_features="sqrt",
            ),
            {"n_estimators": [10], "max_depth": [10, 15]},
            False,
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                early_stopping=False,
            ),
            {"max_iter": [50], "max_depth": [3, 5]},
            True,
        ),
        "linear_svc": (
            LinearSVC(
                loss="hinge",
                dual=True,
                max_iter=1000,
                random_state=RANDOM_STATE,
                tol=1e-2,
            ),
            {"C": c_values},
            False,
        ),
        "gaussian_nb": (
            GaussianNB(),
            None,
            True,
        ),
    }


def continuous_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def train_one(model_name: str):
    X_train, X_test, y_train, y_test, id_test = load_data()
    specs = model_specs(len(y_train))
    if model_name not in specs:
        raise ValueError(f"Modelo desconocido: {model_name}")

    estimator, param_grid, requires_dense = specs[model_name]
    if requires_dense:
        X_train_fit = X_train.toarray().astype(np.float32, copy=False)
        X_test_fit = X_test.toarray().astype(np.float32, copy=False)
    else:
        X_train_fit = X_train
        X_test_fit = X_test

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    fit_start = time.perf_counter()

    if param_grid is None:
        best_model = clone(estimator)
        best_model.fit(X_train_fit, y_train)
        best_params = {}
        best_cv_auc = None
    else:
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
            verbose=2,
            pre_dispatch="2*n_jobs",
            error_score="raise",
        )
        # Threads share the large design matrix and avoid one copy per CV worker.
        with parallel_backend("threading", n_jobs=-1):
            search.fit(X_train_fit, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_cv_auc = float(search.best_score_)
        pd.DataFrame(search.cv_results_).to_csv(
            RESULTS_DIR / f"cv_{model_name}.csv", index=False, encoding="utf-8-sig"
        )

    fit_seconds = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    scores = continuous_scores(best_model, X_test_fit)
    predictions = best_model.predict(X_test_fit).astype(np.int8)
    predict_seconds = time.perf_counter() - predict_start

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics = {
        "model": model_name,
        "implementation": type(best_model).__name__,
        "best_params": best_params,
        "best_cv_auc": best_cv_auc,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "average_precision": float(average_precision_score(y_test, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fit_cv_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "random_state": RANDOM_STATE,
    }

    joblib.dump(best_model, MODELS_DIR / f"{model_name}.joblib", compress=3)
    np.save(PREDICTIONS_DIR / f"scores_{model_name}.npy", scores)
    np.save(PREDICTIONS_DIR / f"predictions_{model_name}.npy", predictions)
    with open(RESULTS_DIR / f"metrics_{model_name}.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)

    prediction_frame = pd.DataFrame(
        {"id": id_test, "default": y_test, "score": scores, "prediction": predictions}
    )
    prediction_frame.to_parquet(
        PREDICTIONS_DIR / f"test_{model_name}.parquet", index=False, compression="snappy"
    )
    print(json.dumps(metrics, indent=2))


def write_hardware():
    hardware = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gib": round(psutil.virtual_memory().total / 1024**3, 3),
        "python": platform.python_version(),
    }
    with open(RESULTS_DIR / "hardware.json", "w", encoding="utf-8") as stream:
        json.dump(hardware, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(model_specs(1).keys()))
    args = parser.parse_args()
    write_hardware()
    train_one(args.model)
