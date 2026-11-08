"""Grid stability desk: score a test CSV with the five saved classifiers."""

from __future__ import annotations

import json
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from evaluation import (
    CLASS_NAMES,
    FEATURE_COLUMNS,
    MODEL_DIR,
    MODEL_SPECS,
    REPORT_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    evaluate_model,
    load_test_frame,
    metric_row,
)

st.set_page_config(page_title="Grid Stability Desk", page_icon=":zap:", layout="wide")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .grid-hero {
            border: 1px solid #d5e0ea;
            background: linear-gradient(180deg, #f7fbff 0%, #eef6f2 100%);
            border-radius: 14px;
            padding: 18px 20px 14px 20px;
            margin-bottom: 8px;
        }
        .grid-kicker {
            color: #5c6d7e;
            font-size: 12px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .grid-title {
            color: #102033;
            font-size: 30px;
            font-weight: 700;
            line-height: 1.15;
            margin-bottom: 6px;
        }
        .grid-copy { color: #243446; font-size: 15.5px; margin-bottom: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_saved_models():
    import joblib

    loaded = []
    for spec in MODEL_SPECS:
        loaded.append({**spec, "model": joblib.load(MODEL_DIR / spec["file"])})
    report = {}
    if REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    return loaded, report


def remember_upload(uploaded) -> None:
    if uploaded is None:
        return
    identity = (uploaded.name, uploaded.size)
    if st.session_state.get("upload_identity") != identity:
        st.session_state.upload_identity = identity
        st.session_state.source = {"name": uploaded.name, "payload": uploaded.getvalue()}


def render_header() -> None:
    st.markdown(
        """
        <div class="grid-hero">
          <div class="grid-kicker">Decentralized smart-grid control</div>
          <div class="grid-title">Grid stability desk</div>
          <p class="grid-copy">
            Upload a test CSV, pick one of the five saved classifiers, and read the
            scores. The models are loaded from disk. Nothing is retrained on the file you upload.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(metrics: dict) -> None:
    labels = [
        ("Accuracy", "accuracy"),
        ("AUC", "auc"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("MCC", "mcc"),
    ]
    for column, (label, key) in zip(st.columns(3) + st.columns(3), labels):
        column.metric(label, f"{metrics[key]:.4f}")
    st.caption(
        "Precision, recall, and F1 are for the unstable class. "
        "AUC is the ROC area for that class. Accuracy and MCC use both classes."
    )


def render_confusion(name: str, metrics: dict) -> None:
    figure, axis = plt.subplots(figsize=(4.8, 4.0))
    sns.heatmap(
        np.array(metrics["confusion_matrix"]),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_title(name)
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)


def render_report(metrics: dict) -> None:
    rows = []
    for label in CLASS_NAMES + ["macro avg", "weighted avg"]:
        stats = metrics["report"][label]
        rows.append(
            {
                "Label": label,
                "Precision": round(stats["precision"], 4),
                "Recall": round(stats["recall"], 4),
                "F1": round(stats["f1-score"], 4),
                "Support": int(stats["support"]),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def render_misses(frame: pd.DataFrame, metrics: dict) -> None:
    view = frame[FEATURE_COLUMNS].copy()
    view["actual"] = [CLASS_NAMES[label] for label in metrics["actual"]]
    view["predicted"] = [CLASS_NAMES[label] for label in metrics["predicted"]]
    misses = view[view["actual"] != view["predicted"]].head(8)
    if misses.empty:
        st.success("Every row in this file was classified correctly by the selected model.")
        return
    st.dataframe(misses, hide_index=True)


def main() -> None:
    inject_css()
    render_header()

    missing = [spec["file"] for spec in MODEL_SPECS if not (MODEL_DIR / spec["file"]).exists()]
    if missing:
        st.error(
            "Saved models are missing: "
            + ", ".join(missing)
            + ". From this folder, run python train_models.py and refresh the page."
        )
        st.stop()

    models, report = load_saved_models()
    names = [item["name"] for item in models]
    blurbs = {item["name"]: item["blurb"] for item in models}

    st.sidebar.header("Model")
    selected_name = st.sidebar.selectbox("Classifier", names)
    st.sidebar.write(blurbs[selected_name])
    selected_key = next(item["key"] for item in models if item["name"] == selected_name)
    saved = report.get("models", {}).get(selected_key, {})
    if saved.get("params"):
        st.sidebar.caption("Parameters stored with the saved model")
        pretty = {key.split("__")[-1]: value for key, value in saved["params"].items()}
        st.sidebar.json(pretty)
    st.sidebar.divider()
    st.sidebar.caption(
        "The bundled test file is the 20 percent stratified holdout used in the README. "
        "Uploading a different CSV recalculates every metric."
    )

    uploaded = st.file_uploader(
        "Upload a test CSV with the 12 input features and the stabf column",
        type=["csv"],
    )
    remember_upload(uploaded)
    if st.button("Score the bundled test_data.csv", type="primary"):
        st.session_state.source = {
            "name": "test_data.csv (bundled holdout)",
            "payload": TEST_DATA_PATH.read_bytes(),
        }

    source = st.session_state.get("source")
    if not source:
        st.info(
            "Upload a test file, or score the bundled holdout. Required columns: "
            + ", ".join(FEATURE_COLUMNS + [TARGET_COLUMN])
            + "."
        )
        if TEST_DATA_PATH.exists():
            preview = pd.read_csv(TEST_DATA_PATH, nrows=5)
            st.caption("First five rows of the bundled test file")
            st.dataframe(preview, hide_index=True)
        return

    try:
        raw = pd.read_csv(BytesIO(source["payload"]))
        frame, notes = load_test_frame(raw)
    except Exception as exc:  # noqa: BLE001 - show the CSV problem to the user
        st.error(str(exc))
        return

    st.subheader(source["name"])
    for note in notes:
        st.warning(note)

    balance = frame[TARGET_COLUMN].value_counts()
    balance_text = ", ".join(f"{label} {int(balance.get(label, 0))}" for label in CLASS_NAMES)
    st.write(f"{len(frame)} rows. Class counts: {balance_text}.")

    with st.spinner("Scoring this file with the five saved models..."):
        scored = []
        for item in models:
            try:
                metrics = evaluate_model(item["model"], frame)
            except Exception as exc:  # noqa: BLE001 - surface metric problems per model
                st.error(f"{item['name']} could not be scored: {exc}")
                return
            scored.append({"spec": item, "metrics": metrics})

    table = pd.DataFrame([metric_row(item["spec"]["name"], item["metrics"]) for item in scored])

    def highlight_selected(row):
        if row["ML Model Name"] == selected_name:
            return ["background-color: #dbeafe"] * len(row)
        return [""] * len(row)

    st.subheader("All five models on this test file")
    st.dataframe(
        table.style.apply(highlight_selected, axis=1).format(
            {
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "MCC": "{:.4f}",
            }
        ),
        hide_index=True,
    )
    st.caption("The selected classifier is highlighted. These numbers were just calculated on the uploaded file.")

    chosen = next(item for item in scored if item["spec"]["name"] == selected_name)
    st.subheader(selected_name)
    render_metrics(chosen["metrics"])

    chart_col, report_col = st.columns(2)
    with chart_col:
        st.markdown("**Confusion matrix**")
        render_confusion(selected_name, chosen["metrics"])
    with report_col:
        st.markdown("**Classification report**")
        render_report(chosen["metrics"])

    st.markdown("**Rows the selected model missed**")
    render_misses(frame, chosen["metrics"])


main()
