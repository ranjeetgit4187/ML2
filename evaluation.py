"""Score saved classifiers without training them again.

Precision, recall, and F1 use unstable as the positive class, because a grid
operator cares first about catching an unstable case. AUC is the ROC area for
that same class. Accuracy and MCC are computed on both classes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_DATA_PATH = ROOT / "test_data.csv"
REPORT_PATH = MODEL_DIR / "training_report.json"

FEATURE_COLUMNS = [
    "tau1",
    "tau2",
    "tau3",
    "tau4",
    "p1",
    "p2",
    "p3",
    "p4",
    "g1",
    "g2",
    "g3",
    "g4",
]
TARGET_COLUMN = "stabf"
LEAKED_COLUMN = "stab"
CLASS_ORDER = [0, 1]
CLASS_NAMES = ["stable", "unstable"]
POSITIVE_LABEL = 1

MODEL_SPECS = [
    {
        "key": "logistic_regression",
        "name": "Logistic Regression",
        "file": "logistic_regression.joblib",
        "blurb": "Straight boundary on scaled reaction time, power, and elasticity.",
    },
    {
        "key": "decision_tree",
        "name": "Decision Tree",
        "file": "decision_tree.joblib",
        "blurb": "Axis-aligned splits. Depth and leaf size were chosen on the training rows only.",
    },
    {
        "key": "knn",
        "name": "KNN",
        "file": "knn.joblib",
        "blurb": "Distance-weighted vote among the nearest scaled training rows.",
    },
    {
        "key": "naive_bayes",
        "name": "Naive Bayes",
        "file": "gaussian_naive_bayes.joblib",
        "blurb": "Gaussian Naive Bayes. Features are treated as independent once the class is known.",
    },
    {
        "key": "svm",
        "name": "Support Vector Machine",
        "file": "support_vector_machine.joblib",
        "blurb": "Maximum-margin classifier. The kernel was chosen on the training rows only.",
    },
]


def load_test_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep the 12 model inputs and the stabf label. Drop the leaked stab column."""
    notes: list[str] = []
    cleaned = frame.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    lookup = {column.lower(): column for column in cleaned.columns}

    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in lookup]
    if missing:
        raise ValueError(
            "The test file is missing required columns: " + ", ".join(missing)
        )

    if LEAKED_COLUMN in lookup:
        notes.append(
            "The numeric column stab was ignored. The label stabf is derived from it, "
            "so keeping it would leak the answer into the features."
        )

    prepared = pd.DataFrame()
    for feature in FEATURE_COLUMNS:
        prepared[feature] = pd.to_numeric(cleaned[lookup[feature]], errors="coerce")
    prepared[TARGET_COLUMN] = cleaned[lookup[TARGET_COLUMN]].astype(str).str.strip().str.lower()

    if prepared[FEATURE_COLUMNS].isna().any().any():
        raise ValueError(
            "Every feature column must be numeric. Check the CSV for blank cells or text."
        )

    unexpected = sorted(set(prepared[TARGET_COLUMN].unique()) - set(CLASS_NAMES))
    if unexpected:
        raise ValueError(
            "stabf must contain only stable and unstable. Found: " + ", ".join(unexpected)
        )
    if prepared.empty:
        raise ValueError("The test file has no rows.")
    return prepared, notes


def encode_target(labels: pd.Series) -> np.ndarray:
    return (labels == "unstable").astype(int).to_numpy()


def _final_estimator(model):
    if hasattr(model, "named_steps") and "model" in getattr(model, "named_steps"):
        return model.named_steps["model"]
    return model


def positive_scores(model, features: pd.DataFrame) -> np.ndarray:
    """Higher scores mean the row looks more unstable."""
    final = _final_estimator(model)
    classes = [int(label) for label in list(model.classes_)]
    if isinstance(final, SVC) and getattr(final, "probability", False) is not True:
        scores = np.asarray(model.decision_function(features), dtype=float)
        if classes[1] != POSITIVE_LABEL:
            scores = -scores
        return scores

    probabilities = model.predict_proba(features)
    return np.asarray(probabilities[:, classes.index(POSITIVE_LABEL)], dtype=float)


def evaluate_model(model, frame: pd.DataFrame) -> dict:
    features = frame[FEATURE_COLUMNS]
    actual = encode_target(frame[TARGET_COLUMN])
    predicted = np.asarray(model.predict(features)).astype(int)
    scores = positive_scores(model, features)

    if len(np.unique(actual)) < 2:
        raise ValueError("AUC needs both stable and unstable rows in the test file.")

    matrix = confusion_matrix(actual, predicted, labels=CLASS_ORDER)
    report = classification_report(
        actual,
        predicted,
        labels=CLASS_ORDER,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "auc": float(roc_auc_score(actual, scores)),
        "precision": float(
            precision_score(actual, predicted, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predicted, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "f1": float(f1_score(actual, predicted, pos_label=POSITIVE_LABEL, zero_division=0)),
        "mcc": float(matthews_corrcoef(actual, predicted)),
        "confusion_matrix": matrix.tolist(),
        "report": report,
        "actual": actual.tolist(),
        "predicted": predicted.tolist(),
    }


def metric_row(name: str, metrics: dict) -> dict:
    return {
        "ML Model Name": name,
        "Accuracy": round(metrics["accuracy"], 4),
        "AUC": round(metrics["auc"], 4),
        "Precision": round(metrics["precision"], 4),
        "Recall": round(metrics["recall"], 4),
        "F1": round(metrics["f1"], 4),
        "MCC": round(metrics["mcc"], 4),
    }
