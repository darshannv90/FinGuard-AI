# FinGuard AI

FinGuard AI is a local Streamlit dashboard for exploring credit-card fraud and consumer-loan default risk. It provides data-quality review, leakage-aware feature engineering, model comparison, saved-model scoring, and model-insights views. Scores are decision-support signals, not guarantees or lending decisions.

## Abstract

This project applies supervised classification and anomaly detection to two public tabular datasets. It validates and cleans data without overwriting raw files, fits preprocessing only on training rows, evaluates models on fixed-seed stratified holdouts, and saves the selected model pipelines with their evaluation results. The dashboard reads real local data and saved artifacts; it does not display fabricated metrics or retrain automatically when opened.

## Problem statement

Fraud and loan-default data are class-imbalanced, and seemingly strong accuracy can conceal missed positive cases. FinGuard AI compares models with precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion counts so that fraud misses, false alarms, and default errors are visible.

## Objectives

- Inspect, clean, and summarize the datasets actually available.
- Engineer repeatable fraud and loan features without using target labels as model inputs.
- Prevent preprocessing and exact-duplicate leakage across training and holdout sets.
- Compare relevant models using fixed random seeds and appropriate imbalance-aware metrics.
- Persist model bundles and measured evaluation results for dashboard inference.
- Provide interactive data-quality, prediction, and model-insights workflows.

## Tools and technologies

Python, pandas, NumPy, scikit-learn, Joblib, Streamlit, Plotly, Matplotlib, Seaborn, Jupyter/nbformat, and the Python standard library. The models use class weighting; SMOTE is not used.

## Datasets and verified observations

The application expects these CSV files in `data/raw/`:

| File | Public source | Observed shape | Target and observed counts |
|---|---|---:|---|
| `creditcard.csv` | [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | 284,807 rows × 31 columns | `Class`: 284,315 non-fraud (0), 492 fraud (1) |
| `loan_data.csv` | [Kaggle: Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) | 32,581 rows × 12 columns | `loan_status`: 25,473 non-default (0), 7,108 default (1) |

These are the local source files inspected during the final stability review. The fraud columns are `Time`, `Amount`, `V1`–`V28`, and `Class`; the dataset contains no customer, merchant, or location identifiers. `Time` is elapsed seconds from the first observed transaction, not a calendar timestamp. The loan columns are `person_age`, `person_income`, `person_home_ownership`, `person_emp_length`, `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_status`, `loan_percent_income`, `cb_person_default_on_file`, and `cb_person_cred_hist_length`.

Observed data-quality facts:

- The fraud file has no missing cells and 1,081 exact duplicate rows. Training removes those rows before splitting, leaving 283,726 unique labeled rows (226,980 train; 56,746 test).
- The loan file has 4,011 missing cells, primarily in `loan_int_rate` (3,116) and `person_emp_length` (895), plus 165 exact duplicate rows. The review cleaning workflow imputes missing feature values on a non-destructive working copy and removes exact duplicates; it does not overwrite the original file.
- Loan review identified 5 ages over 100, 2 employment lengths greater than age, and 283 rows where the supplied loan-to-income percentage differs from the recomputed ratio by more than two percentage points. These are review flags, not automatic corrections. The highest observed age is 144 and highest employment length is 123 years.
- The observed loan default rate is 21.82%. Income and loan amount have Pearson correlation 0.267 and Spearman correlation 0.407 in this dataset; association does not establish causation.

Dataset schemas and target meanings are validated by the loader and analysis workflows. Training excludes rows with missing target labels and rejects unsupported target encodings. To set up local data, download the source CSVs and place them at the expected paths. If the loan download is named `credit_risk_dataset.csv`, rename it to `loan_data.csv`.

## Data wrangling and exploratory analysis

`src/data_loader.py` validates file presence, CSV headers, duplicate column names, data rows, and required target columns. Uploaded files are processed in memory. `src/cleaning.py` and `clean_loan_data()` provide non-destructive quality reports, numeric coercion, training-derived numeric and categorical imputation, category normalization, IQR outlier flags, and optional capping/encoding. Outlier capping is opt-in; target labels are never imputed. Loan consistency checks flag domain and ratio discrepancies for review.

The notebooks `notebooks/01_fraud_eda.ipynb` and `notebooks/02_loan_eda.ipynb` contain the dataset exploration. The Streamlit Data Quality page reports missingness, duplicates, outliers, invalid values, consistency flags, and cleaning logs. On the observed loan data, the mean/median income is 66,074.85/55,000 and mean/median loan amount is 9,589.37/8,000. These summaries and the dashboard charts are calculated from the local data, not sample values.

## Feature engineering

- **Fraud:** `src/fraud_features.py` retains available numeric inputs and derives log amount, deviation and standard/robust z-scores, a high-amount indicator based on the training 99th percentile, elapsed seconds/hours, and an elapsed-hour bucket. It fits medians and amount statistics on training rows. `Class` is excluded. No customer velocity or customer rolling features are created because there is no customer ID or event timestamp suitable for grouping.
- **Loan:** `src/loan_features.py` retains available numeric information with missing indicators, encodes observed categories, creates age groups and training-income quartile indicators, loan-to-income and inverse ratios, a supplied-versus-recomputed ratio gap, a training-quantile high-ratio flag, an employment stability ratio, and an interest-burden proxy when its inputs exist. Zero/invalid denominators are handled explicitly and fitted values are reused at prediction time. There is no total-debt field, so the project does not invent a total debt-to-income ratio.

## Models and evaluation

Training is explicit: open the Overview page's **Dedicated model training controls**, or call the functions in `src/train_fraud.py` and `src/train_loan.py`. Both workflows use `random_state=42`, a stratified 80/20 holdout, class weighting where supported, train-only preprocessing, and remove exact duplicate labeled rows before splitting. Fraud compares Logistic Regression, Random Forest, and Isolation Forest; loan compares Logistic Regression, Decision Tree, and Random Forest. Both select by PR-AUC (average precision), with recall and F1-score as tie-breakers. Reported classification counts use each classifier's default 0.5 threshold. Accuracy is not used as the main selection metric.

Fresh results saved under `models/` after the duplicate-leakage fix:

| Dataset / selected model | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fraud / Random Forest | 0.9714 | 0.7158 | 0.8242 | 0.9350 | 0.7998 | 56,649 | 2 | 27 | 68 |
| Loan / Random Forest | 0.9770 | 0.7193 | 0.8286 | 0.9343 | 0.8923 | 5,042 | 24 | 398 | 1,020 |

PR-AUC is the chosen selection metric because the positive classes are minorities. The selected fraud model missed 27 of 95 fraud cases in its holdout; a high precision does not mean all fraud is caught. The selected loan model missed 398 defaults in its holdout. Changing a decision threshold trades false positives against false negatives and should reflect operational costs. Model scores are not guaranteed to be calibrated probabilities; Isolation Forest produces an anomaly score rather than a probability.

Evaluation CSVs contain the full model comparisons and confusion counts. JSON summaries record the split, selected model, target meaning, and duplicate counts:

- `models/fraud_best_model.joblib`, `models/fraud_evaluation_results.csv`, `models/fraud_training_summary.json`
- `models/loan_best_model.joblib`, `models/loan_model_evaluation_results.csv`, `models/loan_training_summary.json`

## System architecture

```text
data/raw/*.csv
      |
src/data_loader.py -> validation and dataset inspection
      |                           |
src/cleaning.py + feature modules | notebooks/ EDA
      |
src/train_fraud.py / src/train_loan.py
      |                         \
Joblib model bundles       CSV + JSON evaluation artifacts
      |                         |
      +------> Streamlit dashboard <------+
             Overview, Fraud, Loan,        |
             Data Quality, Model Insights-+
```

The app entry point is `app.py`; pages live under `pages/`, reusable code under `src/`, and unit tests under `tests/`. Opening the dashboard does not trigger training. The Overview page has explicit training controls. Uploads are read in memory, targets are removed from scoring inputs, and CSV downloads omit the pandas index.

```text
FinGuard-AI/
|-- app.py
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- notebooks/
|-- pages/
|-- src/
|   |-- cleaning.py
|   |-- dashboard.py
|   |-- data_loader.py
|   |-- evaluation.py
|   |-- fraud_features.py
|   |-- loan_features.py
|   |-- train_fraud.py
|   `-- train_loan.py
|-- tests/
|-- docs/
|-- README.md
`-- requirements.txt
```

## Installation and running (Windows PowerShell)

Use Python 3.10 or later; the final local validation was performed with Python 3.13.5. From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Put the two source CSVs in `data/raw/`, then launch:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Run the complete test suite and dependency consistency check:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

Model training can be triggered from the Overview controls. To invoke it from Python without starting Streamlit:

```powershell
.\.venv\Scripts\python.exe -c "from src.train_fraud import train_fraud_models; print(train_fraud_models().results.to_string(index=False))"
.\.venv\Scripts\python.exe -c "from src.train_loan import train_loan_models; print(train_loan_models().results.to_string(index=False))"
```

## Screenshots

Add current dashboard screenshots here:

- `![Overview dashboard](docs/screenshots/overview.png)`
- `![Fraud detection page](docs/screenshots/fraud-detection.png)`
- `![Loan risk page](docs/screenshots/loan-risk.png)`
- `![Data quality page](docs/screenshots/data-quality.png)`
- `![Model insights page](docs/screenshots/model-insights.png)`

## Limitations

- Both models are trained on the supplied public datasets; performance may not generalize to other populations, time periods, institutions, or fraud patterns.
- The fraud dataset has no customer/merchant/location identity or usable wall-clock timestamp, limiting behavioral velocity features and context.
- The loan dataset lacks total debt, repayment term, and some applicant information; the interest-burden feature is only a simple annual-interest-to-income proxy.
- Holdout results are a single fixed-seed evaluation, not external validation. The loan model's default threshold is not optimized for a lender's costs.
- Predicted scores are not calibrated and must not be used as sole grounds for blocking transactions or making credit decisions.
- Local CSV and Joblib artifacts must be present for the full dashboard workflow. Data provenance and licensing should be checked before redistribution.

## Future scope

- Validate on later-period and external datasets and monitor drift.
- Calibrate scores and select operating thresholds with stakeholder-defined false-positive/false-negative costs.
- Add explainability and fairness assessment appropriate to an approved deployment context.
- Add customer-level fraud velocity only when reliable customer identifiers and timestamps are available.
- Improve data provenance, versioned artifacts, and deployment monitoring.

## Conclusion

FinGuard AI provides an end-to-end, reproducible workflow for data inspection, feature engineering, imbalanced classification, evaluation, and interactive review. Its saved results are measured on the local holdouts, including the corrected duplicate-free fraud split. The dashboard communicates error trade-offs and limitations rather than presenting risk scores as certainty.

## Final status checklist

- [x] Five dashboard destinations, real CSV ingestion, input validation, saved-model scoring, and CSV report downloads.
- [x] Fraud and loan feature engineering; model comparison, fixed-seed holdout evaluation, and saved bundles/results.
- [x] Exact duplicate removal before both training splits; training-only fitted preprocessing; target exclusion.
- [x] Full unit test suite, notebook schema validation, page rendering, real saved-model inference, and dependency checks completed.
- [ ] Production monitoring, external validation, probability calibration, and policy-specific decision thresholds remain outside this project.

See [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) for the verified project report and [docs/PRESENTATION_GUIDE.md](docs/PRESENTATION_GUIDE.md) for a presentation script and viva preparation.
