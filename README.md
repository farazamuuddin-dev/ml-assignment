# AirSense CO Alert — ML Assignment 2

**Live Streamlit app:** `<paste your Streamlit Community Cloud URL here>`
**GitHub repository:** `<paste your GitHub repository URL here>`

## a. Problem statement

Certified reference analysers for carbon monoxide (CO) are expensive, so cities
want to know whether cheap metal-oxide gas sensors can stand in for them. This
project frames that as a **binary classification** problem: using only the five
low-cost sensor responses, temperature/humidity and the time of day, predict
whether the hourly CO concentration measured by the reference analyser is
**High (CO ≥ 2.0 mg/m³, class 1)** or **Normal (class 0)**.

Five classifiers — Logistic Regression, Decision Tree, K-Nearest Neighbours,
Gaussian Naive Bayes and an RBF-kernel Support Vector Machine — are trained on
the same data and compared on Accuracy, AUC, Precision, Recall, F1 and the
Matthews Correlation Coefficient (MCC). The trained models are served through
an interactive Streamlit app where a user uploads labelled test data and
inspects each model's performance without any retraining.

For an air-quality alert, a **false negative** (a high-CO hour reported as
normal) is the more harmful error, so Recall on the High-CO class is watched
alongside overall accuracy.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Air Quality |
| Source | UCI Machine Learning Repository (De Vito et al., ENEA, 2008) — https://archive.ics.uci.edu/dataset/360/air+quality |
| Collected | Hourly averages, March 2004 – April 2005, a road-level station in an Italian city |
| Raw instances | 9,357 hourly records, 15 columns |
| Instances used | 7,344 (rows with a CO(GT) label and sensor readings) |
| Features used | 13 (listed below) |
| Target | `high_co` — 1 if `CO(GT)` ≥ 2.0 mg/m³, else 0 |
| Class balance | 45.8% High CO, 54.2% Normal |
| Missing values | Encoded as `-200` in the raw file |

**Features (13)**

| Feature | Raw column | Description |
|---|---|---|
| `sensor_co_tin_oxide` | PT08.S1(CO) | Tin-oxide sensor response (nominally CO-targeted) |
| `sensor_nmhc_titania` | PT08.S2(NMHC) | Titania sensor response (nominally NMHC-targeted) |
| `sensor_nox_tungsten` | PT08.S3(NOx) | Tungsten-oxide sensor response (NOx-targeted) |
| `sensor_no2_tungsten` | PT08.S4(NO2) | Tungsten-oxide sensor response (NO2-targeted) |
| `sensor_o3_indium` | PT08.S5(O3) | Indium-oxide sensor response (O3-targeted) |
| `temperature_c` | T | Temperature, °C |
| `relative_humidity` | RH | Relative humidity, % |
| `absolute_humidity` | AH | Absolute humidity |
| `hour_sin`, `hour_cos` | Time | Hour of day encoded on a circle (23:00 is next to 00:00) |
| `day_of_week` | Date | 0 = Monday … 6 = Sunday |
| `is_weekend` | Date | 1 on Saturday/Sunday |
| `month` | Date | 1–12 |

**Why the other analyser columns are excluded:** `C6H6(GT)`, `NOx(GT)`,
`NO2(GT)` and `NMHC(GT)` come from the same expensive reference station as the
target. Using them would defeat the purpose (predicting the reference reading
from cheap sensors) and `NMHC(GT)` is also ~90% missing.

**Preprocessing**

1. Parsed the raw file (`;` separator, `,` decimals), dropped the two empty
   trailing columns and blank rows, and converted every `-200` to missing.
2. Dropped 1,683 rows with no `CO(GT)` label and 330 rows where every sensor
   was offline (no information to classify) → 7,344 rows.
3. Built the binary target from `CO(GT)` with a 2.0 mg/m³ threshold (close to
   the median, so classes are roughly balanced) and engineered the time features
   from `Date`/`Time`.
4. Stratified 80/20 train/test split (`random_state=42`): 5,875 train rows,
   1,469 test rows, saved as `test_data.csv`.
5. Every model is an sklearn `Pipeline`: median `SimpleImputer` (so uploaded
   files with gaps still work) → `StandardScaler` (for LR, KNN, SVM) →
   classifier. Decision Tree and Naive Bayes are not scaled.
6. Hyper-parameters chosen by 5-fold stratified `GridSearchCV` on the training
   set only, optimising F1.

Caveat: hourly readings are autocorrelated, so a random split puts neighbouring
hours in train and test and makes the scores somewhat optimistic compared with
forecasting a completely unseen future period.

## c. GitHub repository

`<paste your GitHub repository URL here>`

```
ml-assignment2/
├── app.py                 # Streamlit app
├── train_models.py        # preprocessing, tuning, evaluation, saving
├── requirements.txt
├── README.md
├── test_data.csv          # 1,469-row hold-out set (13 features + high_co)
├── data/
│   └── AirQualityUCI.csv  # raw UCI file
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── metrics.json       # hold-out metrics, confusion counts, tuned hyper-parameters
    └── metadata.json      # feature order, target, threshold, class labels
```

Reproduce locally:

```bash
pip install -r requirements.txt
python train_models.py      # retrains and overwrites model/ and test_data.csv
streamlit run app.py
```

## d. Models used

| Model | Configuration chosen by grid search |
|---|---|
| Logistic Regression | Impute + StandardScaler, `C=10`, L2 penalty |
| Decision Tree | Impute, `max_depth=4`, `min_samples_leaf=20` |
| K-Nearest Neighbours | Impute + StandardScaler, `k=11`, distance-weighted |
| Naive Bayes | Impute, GaussianNB, `var_smoothing=1e-5` |
| Support Vector Machine | Impute + StandardScaler, RBF kernel, `C=30`, `gamma='scale'` |

AUC for the SVM is computed from its `decision_function`, since AUC only needs
a ranking of the test hours, not calibrated probabilities.

### Comparison table (hold-out test set, 1,469 hours)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9183 | 0.9774 | 0.9107 | 0.9107 | 0.9107 | 0.8354 |
| Decision Tree | 0.9135 | 0.9656 | 0.9037 | 0.9077 | 0.9057 | 0.8259 |
| KNN | 0.9285 | 0.9798 | 0.9276 | 0.9152 | 0.9213 | 0.8559 |
| Naive Bayes | 0.8938 | 0.9592 | 0.8817 | 0.8869 | 0.8843 | 0.7862 |
| Support Vector Machine | **0.9353** | **0.9850** | **0.9351** | **0.9226** | **0.9288** | **0.8696** |

Precision, Recall and F1 are for the High-CO class (`1`).

Confusion counts on the test set (797 Normal, 672 High CO):

| Model | False alarms (FP) | High-CO hours missed (FN) |
|---|---|---|
| Logistic Regression | 60 | 60 |
| Decision Tree | 65 | 62 |
| KNN | 48 | 57 |
| Naive Bayes | 80 | 76 |
| SVM | 43 | 52 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Solid baseline (accuracy 0.918, MCC 0.835) with a high AUC of 0.977, so a linear boundary already separates the classes well after scaling. Its coefficients are easy to read: the titania sensor (`PT08.S2`) dominates (+5.1), followed by the tin-oxide CO sensor (+1.3); temperature (−0.9) and absolute humidity (−0.6) are negative, i.e. the model corrects for the sensors' known drift with weather. Grid search chose weak regularisation (`C=10`), suggesting there is little overfitting with 13 features and ~5,900 rows. Errors are perfectly symmetric (60 FP, 60 FN). |
| Decision Tree | Surprisingly close to Logistic Regression (accuracy 0.914, MCC 0.826) despite being only 4 levels deep — grid search preferred a shallow, heavily pruned tree (`min_samples_leaf=20`) over deeper ones, which overfit. It relies almost entirely on one feature: `PT08.S2` carries 89% of the importance and temperature 7%, so it is essentially a set of thresholds on a single sensor. Its AUC (0.966) is the second lowest because a depth-4 tree outputs at most 16 distinct probabilities, giving a coarse ranking. |
| KNN | Second best overall (accuracy 0.929, MCC 0.856, AUC 0.980). Hours with similar sensor readings, weather and time of day tend to have similar CO, which is exactly what neighbour voting exploits; distance weighting with k=11 smooths sensor noise. Scaling is essential here because raw sensor responses (~600–2,000) would otherwise swamp the time features. Its downsides are the largest model file (it stores the training set) and slower prediction. |
| Naive Bayes | Weakest model (accuracy 0.894, MCC 0.786) with the most errors of both kinds (80 FP, 76 FN). The independence assumption is clearly violated: the five metal-oxide sensors respond to the same traffic emissions and are strongly correlated with each other, so the same evidence is counted several times and probabilities become over-confident. The time features (`month`, `day_of_week`) are also not Gaussian. Still, an AUC of 0.959 shows its ranking is reasonable. |
| Support Vector Machine | Best on all six metrics: accuracy 0.935, AUC 0.985, precision 0.935, recall 0.923, F1 0.929, MCC 0.870. It has the fewest false alarms (43) and the fewest missed high-CO hours (52). The RBF kernel captures the non-linear sensor-vs-CO response and its interaction with temperature/humidity that a linear model can only approximate, and the large `C=30` shows the data supports a fairly flexible boundary. |
| **Overall, winner for your dataset?** | **SVM (RBF kernel).** It is the best on every metric, including MCC (0.870), which is used as the deciding metric because it accounts for all four confusion-matrix cells. KNN is a close second (MCC 0.856). If interpretability mattered more — for example, explaining to a city agency which sensor drives the alert — Logistic Regression would be the practical choice at a cost of ~0.035 MCC. |

## Streamlit app features

1. **Dataset upload (CSV)** — upload a labelled test file; the app checks that
   all 13 feature columns and `high_co` are present and valid (`-200` values are
   treated as missing and imputed). The bundled `test_data.csv` can be
   downloaded or used directly.
2. **Model selection dropdown** — choose any of the five saved models.
3. **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1 and MCC,
   computed on the uploaded data with the saved model (no retraining).
4. **Confusion matrix and classification report** — heat-map with counts and
   row percentages, plus the per-class sklearn report.
5. Extras: ROC curve of the selected model against the other four, a
   side-by-side comparison of all models on the uploaded file, and a
   predictions table (filterable to misclassified hours) with CSV download.
