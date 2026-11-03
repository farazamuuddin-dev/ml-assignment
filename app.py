"""
AirSense CO Alert — Streamlit front-end for ML Assignment 2.

Upload a labelled test CSV (13 sensor/weather/time features + `high_co`), pick
one of the five pre-trained classifiers and inspect its metrics, confusion
matrix and classification report. Models are loaded from model/*.joblib;
nothing is retrained on the uploaded data.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from train_models import prepare_raw, read_raw

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}
HAZE = "#B5446E"    # high-CO accent
CLEAR = "#2B7A9E"   # normal-air accent
MUTED = "#B0B7BF"

st.set_page_config(page_title="AirSense CO Alert", page_icon="🌫️", layout="wide")


# ------------------------------------------------------------------ loading
@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}


@st.cache_data
def load_info():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    saved = json.loads((MODEL_DIR / "metrics.json").read_text())
    return meta, saved


def high_co_scores(model, X):
    """Probability of High CO, or SVM decision values (AUC only needs a ranking)."""
    try:
        return model.predict_proba(X)[:, 1]
    except AttributeError:
        return model.decision_function(X)


def score_model(model, X, y):
    y_pred = model.predict(X)
    y_score = high_co_scores(model, X)
    both = y.nunique() == 2
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_score) if both else np.nan,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred, y_score


def check_upload(df, features, target):
    issues = []
    missing = [c for c in features if c not in df.columns]
    if missing:
        issues.append(f"missing feature column(s): {', '.join(missing)}")
    if target not in df.columns:
        issues.append(f"target column `{target}` not found")
    elif not set(df[target].dropna().unique()) <= {0, 1}:
        issues.append(f"`{target}` must contain only 0 (normal) and 1 (high CO)")
    elif df[target].isna().any():
        issues.append(f"`{target}` has empty values")
    if not missing:
        non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            issues.append(f"non-numeric feature column(s): {', '.join(non_numeric)}")
    return issues


# ------------------------------------------------------------------ plots
def confusion_fig(cm, labels):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    row_pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    annot = np.array([[f"{cm[i, j]}\n{row_pct[i, j]:.1%}" for j in range(2)] for i in range(2)])
    sns.heatmap(row_pct, annot=annot, fmt="", cmap="PuRd", cbar=False, vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, linewidths=1, linecolor="white", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def roc_fig(y, scores, selected):
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        sel = name == selected
        ax.plot(fpr, tpr, lw=2.4 if sel else 1, alpha=1 if sel else 0.5,
                color=HAZE if sel else MUTED, label=name if sel else None)
    ax.plot([0, 1], [0, 1], ls="--", color="#DDDDDD", lw=1)
    ax.set_xlabel("False alarm rate")
    ax.set_ylabel("High-CO hours caught")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ UI
models = load_models()
meta, saved = load_info()
FEATURES, TARGET = meta["features"], meta["target"]
LABELS = [meta["class_labels"]["0"], meta["class_labels"]["1"]]
THRESH = meta["co_threshold_mg_m3"]

st.markdown(
    f"<h1 style='margin-bottom:0'>🌫️ AirSense CO Alert</h1>"
    f"<p style='color:grey;margin-top:4px'>Can five low-cost gas sensors plus weather tell us "
    f"when carbon monoxide is <span style='color:{HAZE}'><b>high (≥ {THRESH} mg/m³)</b></span> "
    f"rather than <span style='color:{CLEAR}'><b>normal</b></span>? Five classifiers trained "
    f"on the UCI Air Quality dataset (an Italian city, Mar 2004 – Apr 2005).</p>",
    unsafe_allow_html=True)

with st.sidebar:
    st.header("① Upload test data")
    uploaded = st.file_uploader("Test CSV", type="csv",
                                help=f"Either test_data.csv (the {len(FEATURES)} feature columns + "
                                     f"`{TARGET}`) or the raw UCI AirQualityUCI.csv.")
    sample = ROOT / "test_data.csv"
    use_sample = False
    if sample.exists():
        st.download_button("⬇️ Download sample test_data.csv", sample.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")
        use_sample = st.checkbox("Use bundled test_data.csv", value=uploaded is None)

    st.header("② Pick a model")
    model_name = st.selectbox("Classifier", list(MODEL_FILES))
    params = saved["best_params"].get(model_name, {})
    if params:
        st.caption("Tuned: " + ", ".join(f"`{k}={v}`" for k, v in params.items()))

    with st.expander("Expected columns"):
        st.code("\n".join(FEATURES + [TARGET]), language=None)

raw_note = None
if uploaded is not None:
    source = uploaded.name
    first_line = uploaded.getvalue().split(b"\n", 1)[0].decode("utf-8", "ignore")
    if ";" in first_line and "CO(GT)" in first_line:
        # Raw UCI AirQualityUCI.csv -> apply the same preprocessing as training.
        try:
            raw = read_raw(uploaded)
            df = prepare_raw(raw)
        except Exception as exc:
            st.error(f"Looks like the raw UCI file, but it could not be parsed: {exc}")
            st.stop()
        raw_note = (f"Detected the raw UCI file: {len(raw):,} rows converted to "
                    f"{len(df):,} labelled rows (target = CO(GT) ≥ {THRESH} mg/m³). "
                    "Note: this includes the 80% the models were trained on, so scores "
                    "will be optimistic — use test_data.csv for a fair hold-out evaluation.")
    else:
        df = pd.read_csv(uploaded)
elif use_sample:
    df = pd.read_csv(sample)
    source = "bundled test_data.csv"
else:
    st.info("⬅️ Upload a test CSV, or tick *Use bundled test_data.csv*, to begin.")
    st.stop()

if raw_note:
    st.warning(raw_note)

issues = check_upload(df, FEATURES, TARGET)
if issues:
    st.error("This file can't be scored:\n\n- " + "\n- ".join(issues))
    st.stop()

# The raw UCI file marks missing readings as -200; treat them the same way here.
X = df[FEATURES].replace(-200, np.nan)
y = df[TARGET].astype(int)
st.caption(f"Scoring **{len(df):,} hourly records** from *{source}* · "
           f"high-CO share {y.mean():.1%} · missing cells imputed: {int(X.isna().sum().sum())}")

results = {name: score_model(m, X, y) for name, m in models.items()}
metrics, y_pred, y_score = results[model_name]

tab_one, tab_all, tab_rows = st.tabs(
    [f"🔎 {model_name}", "📊 Compare all models", "🧾 Predictions"])

with tab_one:
    for col, (k, v) in zip(st.columns(6), metrics.items()):
        col.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        st.pyplot(confusion_fig(cm, LABELS))
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"**{fn}** high-CO hours missed · **{fp}** false alarms on normal hours")
    with right:
        st.subheader("ROC curve")
        if y.nunique() == 2:
            st.pyplot(roc_fig(y, {n: r[2] for n, r in results.items()}, model_name))
            st.caption("Grey: the other four models for reference.")
        else:
            st.warning("ROC/AUC needs both classes in the uploaded file.")

    st.subheader("Classification report")
    rep = classification_report(y, y_pred, labels=[0, 1], target_names=LABELS,
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(4), width="stretch")

with tab_all:
    table = pd.DataFrame({n: r[0] for n, r in results.items()}).T
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#F3D9E3"),
                 width="stretch")
    winner = table["MCC"].idxmax()
    st.success(f"Best on this file by MCC: **{winner}** ({table.loc[winner, 'MCC']:.4f})")

    pick = st.radio("Metric to chart", list(table.columns), horizontal=True, index=5)
    order = table[pick].sort_values()
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(order.index, order.values,
            color=[HAZE if n == model_name else MUTED for n in order.index])
    for i, v in enumerate(order.values):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(max(0, order.min() - 0.1), 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(pick)
    fig.tight_layout()
    st.pyplot(fig)

with tab_rows:
    out = pd.DataFrame({
        "row": range(1, len(df) + 1),
        "actual": y.map({0: LABELS[0], 1: LABELS[1]}),
        "predicted": pd.Series(y_pred).map({0: LABELS[0], 1: LABELS[1]}).values,
        "high_co_score": np.round(y_score, 4),
    })
    out["correct"] = out["actual"] == out["predicted"]
    only_wrong = st.toggle("Show only misclassified hours")
    st.dataframe(out[~out["correct"]] if only_wrong else out, width="stretch", hide_index=True)
    st.download_button("Download predictions CSV", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                       mime="text/csv")
