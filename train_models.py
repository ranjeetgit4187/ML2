"""Train the five required classifiers and save them for the Streamlit app.

The held-out test rows are written to test_data.csv. The app scores that file
with the saved models and does not fit anything again.

Run from this folder:
    python train_models.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import (
    FEATURE_COLUMNS,
    MODEL_DIR,
    MODEL_SPECS,
    REPORT_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    evaluate_model,
    metric_row,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "grid_stability.csv"
DATA_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv",
]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        last_error = None
        for url in DATA_URLS:
            try:
                print(f"Downloading dataset from {url}", flush=True)
                urllib.request.urlretrieve(url, DATA_PATH)
                break
            except Exception as exc:  # noqa: BLE001 - report every mirror, then stop
                last_error = exc
                print(f"Download failed: {exc}", flush=True)
        else:
            raise RuntimeError(f"Could not download the UCI grid dataset: {last_error}")

    frame = pd.read_csv(DATA_PATH)
    frame.columns = [str(column).strip().strip('"') for column in frame.columns]
    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN, "stab"])
    missing = sorted(expected - set(frame.columns))
    if missing:
        raise RuntimeError(f"Dataset is missing columns: {missing}")

    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(str).str.strip().str.lower()
    before = len(frame)
    frame = frame.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN, "stab"]).copy()
    dropped = before - len(frame)
    if dropped:
        print(f"Dropped {dropped} rows with missing values.", flush=True)
    if len(frame) < 500:
        raise RuntimeError(f"Need at least 500 rows after cleaning, found {len(frame)}.")
    if len(FEATURE_COLUMNS) < 12:
        raise RuntimeError("Need at least 12 input features.")
    return frame


def scaled(estimator) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )


def choose_svm_kernel(features: pd.DataFrame, target: pd.Series) -> tuple[str, dict]:
    """Pick linear or RBF using a training subset. The test rows stay untouched."""
    probe_x, _, probe_y, _ = train_test_split(
        features,
        target,
        train_size=2500,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    probe_scores = {}
    for kernel in ("linear", "rbf"):
        candidate = scaled(SVC(kernel=kernel, C=1.0, gamma="scale", random_state=RANDOM_STATE))
        scores = cross_val_score(candidate, probe_x, probe_y, cv=3, scoring="f1", n_jobs=1)
        probe_scores[kernel] = float(np.mean(scores))
        print(
            f"SVM kernel probe on 2500 training rows, {kernel}: mean F1 {probe_scores[kernel]:.4f}",
            flush=True,
        )
    chosen = max(probe_scores, key=probe_scores.get)
    print(f"Selected SVM kernel: {chosen}", flush=True)
    return chosen, probe_scores


def fit_with_search(estimator, param_grid, features, target):
    search = GridSearchCV(
        estimator,
        param_grid,
        scoring="f1",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
        refit=True,
    )
    search.fit(features, target)
    return search.best_estimator_, plain(search.best_params_), float(search.best_score_)


def top_pairs(names, values, limit=5):
    ordered = sorted(zip(names, values), key=lambda item: abs(float(item[1])), reverse=True)
    return [{"feature": name, "value": round(float(value), 4)} for name, value in ordered[:limit]]


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def main() -> None:
    frame = ensure_dataset()
    label_counts = frame[TARGET_COLUMN].value_counts().to_dict()
    power_gap = float(frame[["p1", "p2", "p3", "p4"]].sum(axis=1).abs().max())
    stab_matches_label = float(((frame["stab"] > 0) == (frame[TARGET_COLUMN] == "unstable")).mean())
    print(f"Rows: {len(frame)}", flush=True)
    print(f"Class counts: {label_counts}", flush=True)
    print(f"Largest |p1+p2+p3+p4|: {power_gap:.3e}", flush=True)
    print(f"Share of rows where stab>0 matches unstable: {stab_matches_label:.4f}", flush=True)

    target = (frame[TARGET_COLUMN] == "unstable").astype(int)
    features = frame[FEATURE_COLUMNS]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    test_export = x_test.copy()
    test_export[TARGET_COLUMN] = y_test.map({0: "stable", 1: "unstable"}).to_numpy()
    test_export.to_csv(TEST_DATA_PATH, index=False)
    print(f"Wrote {len(test_export)} held-out rows to {TEST_DATA_PATH.name}", flush=True)

    svm_kernel, svm_probe = choose_svm_kernel(x_train, y_train)
    builders = {
        "logistic_regression": lambda: fit_with_search(
            scaled(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            {"model__C": [0.1, 1.0, 10.0]},
            x_train,
            y_train,
        ),
        "decision_tree": lambda: fit_with_search(
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"max_depth": [6, 10, 16, None], "min_samples_leaf": [5, 20]},
            x_train,
            y_train,
        ),
        "knn": lambda: fit_with_search(
            scaled(KNeighborsClassifier()),
            {"model__n_neighbors": [7, 15, 31], "model__weights": ["uniform", "distance"]},
            x_train,
            y_train,
        ),
        "naive_bayes": lambda: fit_with_search(
            scaled(GaussianNB()),
            {"model__var_smoothing": [1e-9, 1e-7, 1e-5]},
            x_train,
            y_train,
        ),
        "svm": lambda: (
            scaled(SVC(kernel=svm_kernel, C=1.0, gamma="scale", random_state=RANDOM_STATE)).fit(
                x_train, y_train
            ),
            {"kernel": svm_kernel, "C": 1.0, "gamma": "scale"},
            svm_probe[svm_kernel],
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report_models = {}
    print("\nHeld-out test metrics", flush=True)
    header = (
        f"{'Model':<24} {'Accuracy':>9} {'AUC':>8} {'Precision':>10} "
        f"{'Recall':>8} {'F1':>8} {'MCC':>8}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for spec in MODEL_SPECS:
        print(f"Fitting {spec['name']}...", flush=True)
        model, params, selection_score = builders[spec["key"]]()
        model_path = MODEL_DIR / spec["file"]
        joblib.dump(model, model_path, compress=3)

        scored = pd.read_csv(TEST_DATA_PATH)
        metrics = evaluate_model(model, scored)
        reloaded = joblib.load(model_path)
        reloaded_metrics = evaluate_model(reloaded, scored)
        for metric_name in ("accuracy", "auc", "precision", "recall", "f1", "mcc"):
            if abs(metrics[metric_name] - reloaded_metrics[metric_name]) > 1e-9:
                raise RuntimeError(f"Reloaded {spec['name']} did not reproduce {metric_name}.")

        train_predictions = np.asarray(model.predict(x_train))
        train_accuracy = float((train_predictions == y_train.to_numpy()).mean())
        extra = {
            "train_accuracy": round(train_accuracy, 4),
            "selection_f1": round(float(selection_score), 4),
        }
        if spec["key"] == "logistic_regression":
            extra["largest_scaled_coefficients"] = top_pairs(
                FEATURE_COLUMNS, model.named_steps["model"].coef_.ravel()
            )
        elif spec["key"] == "decision_tree":
            extra["top_importances"] = top_pairs(FEATURE_COLUMNS, model.feature_importances_, limit=12)
            extra["leaf_count"] = int(model.get_n_leaves())
            extra["depth"] = int(model.get_depth())
        elif spec["key"] == "svm":
            final = model.named_steps["model"]
            extra["n_support"] = [int(count) for count in final.n_support_]
            extra["kernel_probe_f1"] = {key: round(value, 4) for key, value in svm_probe.items()}
        elif spec["key"] == "naive_bayes":
            extra["class_prior"] = [
                round(float(prior), 4) for prior in model.named_steps["model"].class_prior_
            ]

        row = metric_row(spec["name"], metrics)
        print(
            f"{spec['name']:<24} {row['Accuracy']:9.4f} {row['AUC']:8.4f} "
            f"{row['Precision']:10.4f} {row['Recall']:8.4f} {row['F1']:8.4f} {row['MCC']:8.4f}",
            flush=True,
        )
        report_models[spec["key"]] = {
            "display_name": spec["name"],
            "file": spec["file"],
            "params": plain(params),
            "metrics": {key: row[key] for key in ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")},
            "extra": plain(extra),
        }

    report = {
        "dataset": "Electrical Grid Stability Simulated Data (UCI)",
        "rows": int(len(frame)),
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "dropped_column": "stab",
        "positive_class": "unstable",
        "class_counts": {str(key): int(value) for key, value in label_counts.items()},
        "power_balance_max_abs": power_gap,
        "stab_sign_matches_unstable_label": stab_matches_label,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
        },
        "metric_definitions": {
            "accuracy": "Share of test rows predicted correctly.",
            "auc": "ROC AUC for the unstable class.",
            "precision": "Precision for the unstable class.",
            "recall": "Recall for the unstable class.",
            "f1": "F1 for the unstable class.",
            "mcc": "Matthews correlation over both classes.",
        },
        "models": report_models,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved models and {REPORT_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
