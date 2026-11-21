# FinGuard AI: Financial Risk Analytics Dashboard

## Mini-project report

### Abstract

FinGuard AI is a Streamlit application for exploratory analysis and risk scoring on public credit-card transaction and consumer-loan datasets. The project implements non-destructive data-quality workflows, reusable feature engineering, Logistic Regression, Random Forest, Decision Tree, and Isolation Forest comparisons, fixed-seed stratified holdout evaluation, and saved Joblib inference bundles. Model selection uses PR-AUC rather than accuracy because the target classes are imbalanced. A stability review identified exact duplicate fraud transactions crossing the earlier holdout split; training was corrected to remove exact duplicate labeled rows before splitting, and the saved fraud evaluation artifacts were regenerated. This report uses only the resulting observed dataset and model values.

### 1. Introduction and problem statement

Fraud monitoring and loan-default risk assessment involve rare or imbalanced outcomes. A model can have high apparent accuracy while missing positive cases, and a model that catches more cases may also create costly false alarms. The project therefore presents precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion counts, with particular attention to fraud misses and false alarms. Its dashboard is an analytical prototype, not an automated payment or lending decision system.

### 2. Objectives

1. Validate and explore the available datasets and their target encodings.
2. Clean and transform feature data without overwriting source CSVs.
3. Engineer deterministic features using only columns actually present.
4. Prevent target leakage and fit preprocessing only on training data.
5. Compare model families using a fixed-seed stratified holdout and imbalance-aware metrics.
6. Persist selected models and their measured evaluation results.
7. Expose data quality, scoring, and model insights through a navigable Streamlit dashboard.

### 3. Dataset description and observed data

The inspected fraud dataset contains 284,807 rows and 31 columns: `Time`, `Amount`, `V1` through `V28`, and `Class`. `Class=0` denotes non-fraud (284,315 records) and `Class=1` denotes fraud (492 records). There are no missing cells and 1,081 exact duplicate rows. `Time` is elapsed seconds, not a transaction calendar timestamp; the data has no customer, merchant, or location identifiers.

The inspected loan dataset contains 32,581 rows and 12 columns. The target is `loan_status`, verified as 0=non-default (25,473) and 1=default (7,108). It contains 4,011 missing cells, including 3,116 in `loan_int_rate` and 895 in `person_emp_length`, and 165 exact duplicate rows. No target labels were missing in the observed training files.

The observed loan default rate is 21.82%. Mean/median applicant income is 66,074.85/55,000; mean/median loan amount is 9,589.37/8,000. Pearson and Spearman income-to-loan amount correlations are 0.267 and 0.407. The largest observed age is 144 and the largest employment-length value is 123 years. Quality analysis flags 5 ages over 100, 2 employment lengths greater than age, and 283 loan-percent-income values differing from the recomputed loan/income ratio by more than two percentage points. These are review flags, not automatic factual corrections.

### 4. Data wrangling and exploratory analysis

`src/data_loader.py` checks the expected file, CSV syntax and headers, duplicate column names, data rows, and required target. Uploads are read in memory. Cleaning functions create working copies and report missing values, exact duplicates, data types, corrupted categories, IQR outliers, and domain-consistency flags.

Numeric feature values are coerced where appropriate; missing numeric values use training-derived medians and missing categories use training-derived modes or explicit unknown indicators. Categories are normalized consistently. Outliers are flagged by default rather than silently removed or capped; optional capping is explicit. Target labels are not imputed. Loan cleaning removes exact source duplicates before imputation and yields 32,416 rows with no remaining missing cells in its cleaned working copy.

The notebooks `notebooks/01_fraud_eda.ipynb` and `notebooks/02_loan_eda.ipynb` cover class/outcome distribution, transaction amount and time, loan amount and income, employment, credit-history fields, correlations, and outliers. Summary values are computed from the actual local datasets.

### 5. Feature engineering

Fraud features retain numeric input columns and derive log amount, deviation from training mean, standard and robust amount z-scores, a high-amount indicator based on the training 99th percentile, elapsed seconds/hours, and elapsed-hour bucket. All statistics and imputations are fit on training rows only. `Class` is excluded. Customer-level frequency or rolling features are not created because no customer identifier is present.

Loan features include source numeric fields with missingness indicators, category indicators, fixed age groups, training-income quartile bands, loan-to-income and inverse ratios, a supplied-versus-derived ratio gap and consistency flag, a high loan-to-income flag based on the training 75th percentile, an employment-stability ratio, and an interest-burden proxy where input columns exist. Invalid and zero denominators are handled; the fitted feature schema is reused for inference. No total debt-to-income ratio is created because total debt is not present.

### 6. Modeling and evaluation methodology

Both training workflows remove exact duplicate labeled rows before splitting, use stratified 80/20 train/test splits with random state 42, fit preprocessing on training rows only, and use class weights where supported. The target is not included in engineered model inputs. The selected model, evaluation CSV, and JSON run summary are saved under `models/`.

Fraud candidates: Logistic Regression, Random Forest, and Isolation Forest. Isolation Forest is unsupervised and returns an anomaly score, not a calibrated fraud probability. Loan candidates: Logistic Regression, Decision Tree, and Random Forest. Candidate comparisons report precision, recall, F1-score, ROC-AUC, average precision (reported as PR-AUC), and TN/FP/FN/TP confusion counts. Model selection uses the highest PR-AUC, with recall and F1-score as tie-breakers. Classification confusion counts use each classifier's default threshold (0.5 for the supervised classifiers).

### 7. Measured results

#### Fraud comparison

Unique labeled records after removing exact duplicates: 283,726 (226,980 train; 56,746 test). No unlabeled rows were dropped. The saved holdout results are:

| Model | Precision | Recall | F1-score | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0560 | 0.8737 | 0.1053 | 0.9683 | 0.6762 | 55,252 | 1,399 | 12 | 83 |
| Random Forest (selected) | 0.9714 | 0.7158 | 0.8242 | 0.9350 | 0.7998 | 56,649 | 2 | 27 | 68 |
| Isolation Forest | 0.0295 | 0.8211 | 0.0569 | 0.9432 | 0.0586 | 54,082 | 2,569 | 17 | 78 |

Random Forest has the highest PR-AUC and was selected. In its test split, it detected 68 of 95 fraud rows and missed 27; it produced 2 false alarms. Logistic Regression found 83 fraud cases but generated 1,399 false alarms at its default threshold. Isolation Forest found 78 but generated 2,569 false alarms. These are holdout-specific results and the preferred operating point depends on review capacity and the cost of missed fraud.

#### Loan comparison

After dropping 165 exact duplicate labeled rows, 32,416 records were split into 25,932 train and 6,484 test rows:

| Model | Precision | Recall | F1-score | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5618 | 0.7884 | 0.6561 | 0.8850 | 0.7482 | 4,194 | 872 | 300 | 1,118 |
| Decision Tree | 0.7541 | 0.7722 | 0.7631 | 0.8509 | 0.6322 | 4,709 | 357 | 323 | 1,095 |
| Random Forest (selected) | 0.9770 | 0.7193 | 0.8286 | 0.9343 | 0.8923 | 5,042 | 24 | 398 | 1,020 |

Random Forest was selected by PR-AUC. On this holdout it generated 24 false default flags and missed 398 actual defaults. The default 0.5 classification threshold is not a lending policy; threshold selection requires the institution's risk appetite and error costs.

### 8. System architecture and implementation

The Streamlit entry point is `app.py` and registers five destinations: Overview, Credit Card Fraud Detection, Loan Default Risk Analysis, Data Quality & Wrangling, and Model Insights. Reusable loader, cleaner, feature, evaluation, and training modules live under `src/`. Notebooks live under `notebooks/`; automated unit tests live under `tests/`. Saved Joblib bundles and corresponding CSV/JSON evaluation artifacts live under `models/`.

Dashboard startup loads datasets and existing model artifacts but does not train models. The Overview page has explicit training controls. Fraud and loan pages support data upload/validation and prediction workflows; outputs can be downloaded as CSV. The Model Insights page shows saved comparisons, confusion counts, false positives/negatives, and supported feature importance.

### 9. Verification

The final stability checks completed with Python 3.13.5:

- 59 unit tests passed using `python -m unittest discover -s tests -v`.
- All eight application modules imported successfully; `pip check` reported no broken requirements.
- Both notebooks passed `nbformat.validate`.
- AppTest rendered `app.py` and each of the five dashboard destinations without exceptions.
- CSV validation and CSV download round-trip tests passed.
- Both saved model bundles scored 100 actual rows from their respective local datasets with finite scores.
- Reconstructing the corrected fraud split found zero exact duplicate input rows shared between train and test.
- `compileall` completed successfully for `src` and `tests`.

### 10. Limitations and future work

Results are single-holdout estimates on public datasets and may not transfer to new populations or time periods. Fraud lacks identity and wall-clock context; loan data lacks total debt and loan term. Scores are not calibrated, and threshold policies are not optimized for real operational costs. Deployment would require external validation, monitoring, fairness review, probability calibration, cost-based threshold setting, artifact/data versioning, and approved governance.

### 11. Conclusion

FinGuard AI delivers an end-to-end analysis and scoring prototype that uses available fields, records actual holdout results, and surfaces error trade-offs. The final review corrected exact-duplicate leakage in the fraud split and regenerated the persisted fraud metrics. The resulting models should be interpreted as research prototypes, not production fraud controls or credit decisions.

### References

- Credit-card data: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Loan data: [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- Implementation and saved evaluation artifacts: this repository's `src/`, `notebooks/`, and `models/` directories.
