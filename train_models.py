"""
Train and evaluate five classifiers on the UCI Air Quality dataset.

Task: predict whether the hourly CO concentration measured by the reference
analyser, CO(GT), is HIGH (>= 2.0 mg/m^3) using only the low-cost metal-oxide
sensor readings, weather and time-of-day features.

Run:  python train_models.py

Outputs
-------
model/<name>.joblib      one fitted sklearn Pipeline per classifier (imputation/scaling included)
model/metrics.json       hold-out metrics + tuned hyper-parameters (used in the README)
model/metadata.json      feature list, target name, threshold, split details
test_data.csv            the 20% hold-out split (features + target) to upload into the app
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
RAW_CSV = ROOT / "data" / "AirQualityUCI.csv"
MODEL_DIR = ROOT / "model"
TARGET = "high_co"
CO_THRESHOLD = 2.0          # mg/m^3, hourly CO(GT)
SEED = 42

SENSOR_COLS = {             # raw name -> clean name
    "PT08.S1(CO)": "sensor_co_tin_oxide",
    "PT08.S2(NMHC)": "sensor_nmhc_titania",
    "PT08.S3(NOx)": "sensor_nox_tungsten",
    "PT08.S4(NO2)": "sensor_no2_tungsten",
    "PT08.S5(O3)": "sensor_o3_indium",
    "T": "temperature_c",
    "RH": "relative_humidity",
    "AH": "absolute_humidity",
}
TIME_COLS = ["hour_sin", "hour_cos", "day_of_week", "is_weekend", "month"]
FEATURES = list(SENSOR_COLS.values()) + TIME_COLS


def read_raw(path_or_buffer):
    """Read the raw UCI file: semicolon separated, comma decimals, -200 = missing,
    two empty trailing columns and blank trailing rows."""
    raw = pd.read_csv(path_or_buffer, sep=";", decimal=",", na_values=[-200, "-200"])
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
    return raw.dropna(subset=["Date", "Time"]).reset_index(drop=True)


def prepare_raw(raw, verbose=False):
    """Turn raw UCI rows into the 13 model features + `high_co` target."""
    ts = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H.%M.%S")
    df = raw.rename(columns=SENSOR_COLS)[list(SENSOR_COLS.values())].copy()
    hour = ts.dt.hour
    # Encode hour on a circle so 23:00 and 00:00 are neighbours.
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).round(6)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).round(6)
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["month"] = ts.dt.month

    co = raw["CO(GT)"]
    keep = co.notna() & df[list(SENSOR_COLS.values())].notna().any(axis=1)
    df, co = df[keep].reset_index(drop=True), co[keep].reset_index(drop=True)
    df[TARGET] = (co >= CO_THRESHOLD).astype(int)
    if verbose:
        print(f"Rows with a CO label and at least one sensor reading: {len(df)}")
        print(f"High-CO share: {df[TARGET].mean():.3f}")
        print(f"Remaining missing feature cells: {int(df[FEATURES].isna().sum().sum())}")
    return df


def load_and_prepare():
    raw = read_raw(RAW_CSV)
    print(f"Raw rows: {len(raw)}")
    return prepare_raw(raw, verbose=True)


def prep(scale=True):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return steps


def build_searches():
    """name -> (pipeline, small hyper-parameter grid)."""
    return {
        "Logistic Regression": (
            Pipeline(prep() + [("clf", LogisticRegression(max_iter=5000))]),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "Decision Tree": (
            Pipeline(prep(scale=False) + [("clf", DecisionTreeClassifier(random_state=SEED))]),
            {"clf__max_depth": [4, 6, 8, 12, None], "clf__min_samples_leaf": [1, 5, 20]},
        ),
        "KNN": (
            Pipeline(prep() + [("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [5, 11, 21, 31], "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes": (
            Pipeline(prep(scale=False) + [("clf", GaussianNB())]),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]},
        ),
        "SVM": (
            Pipeline(prep() + [("clf", SVC(kernel="rbf", random_state=SEED))]),
            {"clf__C": [1, 10, 30], "clf__gamma": ["scale", 0.05]},
        ),
    }


def positive_scores(model, X):
    """Score for the high-CO class. SVC has no predict_proba by default, so use
    its decision_function (AUC only needs a ranking)."""
    try:
        return model.predict_proba(X)[:, 1]
    except AttributeError:
        return model.decision_function(X)


def evaluate(model, X, y):
    y_pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    scores = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, positive_scores(model, X)),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return {k: round(float(v), 4) for k, v in scores.items()}, \
        {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def main():
    df = load_and_prepare()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, confusions, best_params = {}, {}, {}

    for name, (pipe, grid) in build_searches().items():
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        results[name], confusions[name] = evaluate(best, X_test, y_test)
        best_params[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        joblib.dump(best, MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib"), compress=3)
        print(f"{name:20s} {best_params[name]}  {results[name]}  {confusions[name]}")

    pd.concat([X_test, y_test], axis=1).to_csv(ROOT / "test_data.csv", index=False)

    (MODEL_DIR / "metrics.json").write_text(json.dumps(
        {"test_metrics": results, "confusion": confusions, "best_params": best_params},
        indent=2, default=str))
    (MODEL_DIR / "metadata.json").write_text(json.dumps({
        "target": TARGET,
        "co_threshold_mg_m3": CO_THRESHOLD,
        "features": FEATURES,
        "class_labels": {"0": "Normal CO", "1": "High CO"},
        "n_rows": len(df), "n_train": len(X_train), "n_test": len(X_test),
        "high_co_share": round(float(y.mean()), 4), "random_state": SEED,
    }, indent=2))

    # Feature importances for the README observations
    tree = joblib.load(MODEL_DIR / "decision_tree.joblib").named_steps["clf"]
    lr = joblib.load(MODEL_DIR / "logistic_regression.joblib").named_steps["clf"]
    print("\nTree importances:", dict(sorted(zip(FEATURES, tree.feature_importances_.round(3)),
                                         key=lambda kv: -kv[1])))
    print("LR coefficients:", dict(sorted(zip(FEATURES, lr.coef_[0].round(3)),
                                          key=lambda kv: -abs(kv[1]))))
    print("\n" + pd.DataFrame(results).T.to_string())


if __name__ == "__main__":
    main()
