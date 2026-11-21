# FinGuard AI Presentation Guide

## Suggested 10-slide presentation

### Slide 1 — Title and motivation
- **Content:** FinGuard AI; fraud detection and loan default-risk analytics.
- **Speaking script:** “FinGuard AI is a data-driven prototype that explores two financial risk problems and makes its model errors visible rather than relying on accuracy alone.”

### Slide 2 — Problem and objectives
- **Content:** Imbalanced targets, missed fraud/defaults, false alarms; objectives: clean data, engineer repeatable features, compare models, communicate risks.
- **Speaking script:** “The central challenge is balancing missed positive cases against false alarms. We built a repeatable workflow to measure that trade-off.”

### Slide 3 — Data and target meaning
- **Content:** Fraud: 284,807 rows, 31 columns, 492 fraud labels. Loan: 32,581 rows, 12 columns, 7,108 defaults. `Class=1` means fraud; verified `loan_status=1` means default.
- **Speaking script:** “The fraud target is extremely rare. Loan labels were checked in the source data before training; they were not assumed from the column name.”

### Slide 4 — Data quality and EDA
- **Content:** Fraud has 1,081 exact duplicate rows and no missing cells. Loan has 4,011 missing cells and 165 exact duplicates; age, employment, and ratio inconsistencies are flagged for review.
- **Speaking script:** “The quality page separates missingness, duplicates, statistical outliers, and domain inconsistencies. Flags are for review, not invented corrections.”

### Slide 5 — Feature engineering
- **Content:** Fraud amount and elapsed-time features; loan ratios, age/income groups, employment and credit indicators. Targets excluded; customer and total-debt features omitted because the source data lacks those fields.
- **Speaking script:** “Every feature comes from an observed column. Training-derived statistics are fitted on training data and then reused for holdout and prediction rows.”

### Slide 6 — Modeling design
- **Content:** Fraud candidates: Logistic Regression, Random Forest, Isolation Forest. Loan: Logistic Regression, Decision Tree, Random Forest. Stratified 80/20 split, fixed seed 42, class weights, duplicate removal before split.
- **Speaking script:** “We compare supervised baselines and an anomaly detector for fraud, then three classifiers for loan default. The dashboard loads saved models; it does not retrain on startup.”

### Slide 7 — Evaluation metrics
- **Content:** Define precision, recall, F1, ROC-AUC, PR-AUC; PR-AUC chosen for imbalanced target classes; show confusion matrix terms.
- **Speaking script:** “Recall measures how many positive cases we catch. Precision measures how many flagged cases are truly positive. PR-AUC helps compare ranking quality when positives are rare.”

### Slide 8 — Verified results
- **Content:** Fraud Random Forest PR-AUC 0.7998, precision 0.9714, recall 0.7158; confusion counts: FP=2, FN=27. Loan Random Forest PR-AUC 0.8923, precision 0.9770, recall 0.7193; FP=24, FN=398.
- **Speaking script:** “These are measured holdout results, not promises. The fraud model still missed 27 fraud rows, and the loan model missed 398 defaults, so the operating threshold and costs matter.”

### Slide 9 — Dashboard demo and limitations
- **Content:** Five pages, uploads, single/batch scoring, data-quality review, model comparisons and error analysis. Limitations: one public dataset per problem, no calibration, no external validation.
- **Speaking script:** “The dashboard is designed for review and explanation. It is not a production decision engine, and the scores should not be used alone to block a transaction or deny credit.”

### Slide 10 — Conclusion and future work
- **Content:** Reproducible, tested pipeline; corrected duplicate leakage; future: external/time validation, calibration, cost-sensitive thresholds, monitoring and fairness review.
- **Speaking script:** “The project demonstrates a full, measurable workflow and makes its limitations explicit. The next step is validation and governance, not simply deploying the current threshold.”

## Demo flow

1. Start the app from the repository root with `.\.venv\Scripts\python.exe -m streamlit run app.py`.
2. On **Overview**, show dataset counts, model availability, saved PR-AUC values, and the fact that training is an explicit action.
3. Open **Credit Card Fraud Detection**. Use the local fraud CSV or upload a valid CSV, show the preview/validation, run predictions, review probability/score and flagged rows, then download the CSV report.
4. Open **Loan Default Risk Analysis**. Show the data quality summary, use an applicant form with actual feature inputs or batch-upload rows, inspect risk scores and flagged applicants, then download the report.
5. Open **Data Quality & Wrangling**. Point out missing values, exact duplicates, outlier flags, consistency flags, and non-destructive cleaning log.
6. Open **Model Insights**. Compare actual saved model results and confusion matrices; explain false positives and false negatives.
7. If time allows, open the Overview training expander and explain that clicking a dedicated control explicitly retrains and refreshes saved artifacts. Avoid retraining during a short demo unless prepared to wait for the full dataset.

## Likely faculty viva questions and answers

**1. Why is accuracy not the primary metric?**

Fraud and default labels are imbalanced. A model can predict mostly the majority class and still report high accuracy while missing positive cases. Precision, recall, F1, ROC-AUC, PR-AUC, and confusion counts show more useful behavior.

**2. How do you know `loan_status=1` means default?**

The target encoding was inspected in the actual dataset and validated in the loan analysis/training code: 0 is non-default and 1 is default.

**3. What is the difference between precision and recall?**

Precision is the proportion of flagged positives that are actually positive. Recall is the proportion of all actual positives that the model flags. Increasing one can reduce the other depending on the threshold.

**4. Why select with PR-AUC?**

PR-AUC (average precision) emphasizes positive-class retrieval and precision across score thresholds, which is informative when the positive class is the minority. Recall and F1 are tie-breakers; it does not automatically define a deployment threshold.

**5. How did you prevent data leakage?**

Exact duplicate labeled rows are removed before the train/test split. Feature engineering and imputation statistics are fitted on training rows only, then reused for test and prediction rows. Target columns are removed from model inputs.

**6. Why use Isolation Forest for fraud?**

It provides an unsupervised anomaly-detection comparison. Its anomaly score is not a calibrated fraud probability, and on this dataset its measured false-alarm/PR-AUC results were worse than the selected Random Forest.

**7. What do the fraud results mean operationally?**

At the default classifier threshold, the selected Random Forest had 68 true positives, 27 missed fraud cases, and 2 false positives on the holdout. A real system must choose an operating threshold based on review capacity and the cost of missed fraud.

**8. Why are there no customer velocity features?**

The fraud CSV has no customer ID and `Time` is elapsed seconds rather than an identified customer's timestamp stream. Grouping by customers or fabricating identity would be invalid.

**9. Is `loan_percent_income` total debt-to-income ratio?**

No. It represents the available loan-to-income measure. The dataset does not contain total debt, so the code does not claim or create total debt-to-income.

**10. Are the model scores calibrated or ready for automatic decisions?**

No. The project does not calibrate scores, conduct external validation, or optimize thresholds against institution-specific costs. The dashboard is a prototype for analytical review, not an automated lending or fraud-blocking system.

**11. What happens when a dataset has missing values?**

The quality workflow reports them. The training workflow uses imputers fitted on training rows, and the saved inference bundle applies those same fitted transformations to later rows. Missing targets are excluded rather than imputed.

**12. What is the most important next step?**

Evaluate on external and later-period data, then calibrate scores and choose thresholds based on explicit false-positive and false-negative costs, with fairness, governance, and monitoring reviews.

# FinGuard AI Presentation Guide

## Suggested 10-slide presentation

### Slide 1 — Title and motivation
- **Content:** FinGuard AI; fraud detection and loan default-risk analytics.
- **Speaking script:** “FinGuard AI is a data-driven prototype that explores two financial risk problems and makes its model errors visible rather than relying on accuracy alone.”

### Slide 2 — Problem and objectives
- **Content:** Imbalanced targets, missed fraud/defaults, false alarms; objectives: clean data, engineer repeatable features, compare models, communicate risks.
- **Speaking script:** “The central challenge is balancing missed positive cases against false alarms. We built a repeatable workflow to measure that trade-off.”

### Slide 3 — Data and target meaning
- **Content:** Fraud: 284,807 rows, 31 columns, 492 fraud labels. Loan: 32,581 rows, 12 columns, 7,108 defaults. `Class=1` means fraud; verified `loan_status=1` means default.
- **Speaking script:** “The fraud target is extremely rare. Loan labels were checked in the source data before training; they were not assumed from the column name.”

### Slide 4 — Data quality and EDA
- **Content:** Fraud has 1,081 exact duplicate rows and no missing cells. Loan has 4,011 missing cells and 165 exact duplicates; age, employment, and ratio inconsistencies are flagged for review.
- **Speaking script:** “The quality page separates missingness, duplicates, statistical outliers, and domain inconsistencies. Flags are for review, not invented corrections.”

### Slide 5 — Feature engineering
- **Content:** Fraud amount and elapsed-time features; loan ratios, age/income groups, employment and credit indicators. Targets excluded; customer and total-debt features omitted because the source data lacks those fields.
- **Speaking script:** “Every feature comes from an observed column. Training-derived statistics are fitted on training data and then reused for holdout and prediction rows.”

### Slide 6 — Modeling design
- **Content:** Fraud candidates: Logistic Regression, Random Forest, Isolation Forest. Loan: Logistic Regression, Decision Tree, Random Forest. Stratified 80/20 split, fixed seed 42, class weights, duplicate removal before split.
- **Speaking script:** “We compare supervised baselines and an anomaly detector for fraud, then three classifiers for loan default. The dashboard loads saved models; it does not retrain on startup.”

### Slide 7 — Evaluation metrics
- **Content:** Define precision, recall, F1, ROC-AUC, PR-AUC; PR-AUC chosen for imbalanced target classes; show confusion matrix terms.
- **Speaking script:** “Recall measures how many positive cases we catch. Precision measures how many flagged cases are truly positive. PR-AUC helps compare ranking quality when positives are rare.”

### Slide 8 — Verified results
- **Content:** Fraud Random Forest PR-AUC 0.7998, precision 0.9714, recall 0.7158; confusion counts: FP=2, FN=27. Loan Random Forest PR-AUC 0.8923, precision 0.9770, recall 0.7193; FP=24, FN=398.
- **Speaking script:** “These are measured holdout results, not promises. The fraud model still missed 27 fraud rows, and the loan model missed 398 defaults, so the operating threshold and costs matter.”

### Slide 9 — Dashboard demo and limitations
- **Content:** Five pages, uploads, single/batch scoring, data-quality review, model comparisons and error analysis. Limitations: one public dataset per problem, no calibration, no external validation.
- **Speaking script:** “The dashboard is designed for review and explanation. It is not a production decision engine, and the scores should not be used alone to block a transaction or deny credit.”

### Slide 10 — Conclusion and future work
- **Content:** Reproducible, tested pipeline; corrected duplicate leakage; future: external/time validation, calibration, cost-sensitive thresholds, monitoring and fairness review.
- **Speaking script:** “The project demonstrates a full, measurable workflow and makes its limitations explicit. The next step is validation and governance, not simply deploying the current threshold.”

## Demo flow

1. Start the app from the repository root with `.\.venv\Scripts\python.exe -m streamlit run app.py`.
2. On **Overview**, show dataset counts, model availability, saved PR-AUC values, and the fact that training is an explicit action.
3. Open **Credit Card Fraud Detection**. Use the local fraud CSV or upload a valid CSV, show the preview/validation, run predictions, review probability/score and flagged rows, then download the CSV report.
4. Open **Loan Default Risk Analysis**. Show the data quality summary, use an applicant form with actual feature inputs or batch-upload rows, inspect risk scores and flagged applicants, then download the report.
5. Open **Data Quality & Wrangling**. Point out missing values, exact duplicates, outlier flags, consistency flags, and non-destructive cleaning log.
6. Open **Model Insights**. Compare actual saved model results and confusion matrices; explain false positives and false negatives.
7. If time allows, open the Overview training expander and explain that clicking a dedicated control explicitly retrains and refreshes saved artifacts. Avoid retraining during a short demo unless prepared to wait for the full dataset.

## Likely faculty viva questions and answers

**1. Why is accuracy not the primary metric?**  
Fraud and default labels are imbalanced. A model can predict mostly the majority class and still report high accuracy while missing positive cases. Precision, recall, F1, ROC-AUC, PR-AUC, and confusion counts show more useful behavior.

**2. How do you know `loan_status=1` means default?**  
The target encoding was inspected in the actual dataset and validated in the loan analysis/training code: 0 is non-default and 1 is default.

**3. What is the difference between precision and recall?**  
Precision is the proportion of flagged positives that are actually positive. Recall is the proportion of all actual positives that the model flags. Increasing one can reduce the other depending on the threshold.

**4. Why select with PR-AUC?**  
PR-AUC (average precision) emphasizes positive-class retrieval and precision across score thresholds, which is informative when the positive class is the minority. Recall and F1 are tie-breakers; it does not automatically define a deployment threshold.

**5. How did you prevent data leakage?**  
Exact duplicate labeled rows are removed before the train/test split. Feature engineering and imputation statistics are fitted on training rows only, then reused for test and prediction rows. Target columns are removed from model inputs.

**6. Why use Isolation Forest for fraud?**  
It provides an unsupervised anomaly-detection comparison. Its anomaly score is not a calibrated fraud probability, and on this dataset its measured false-alarm/PR-AUC results were worse than the selected Random Forest.

**7. What do the fraud results mean operationally?**  
At the default classifier threshold, the selected Random Forest had 68 true positives, 27 missed fraud cases, and 2 false positives on the holdout. A real system must choose an operating threshold based on review capacity and the cost of missed fraud.

**8. Why are there no customer velocity features?**  
The fraud CSV has no customer ID and `Time` is elapsed seconds rather than an identified customer's timestamp stream. Grouping by customers or fabricating identity would be invalid.

**9. Is `loan_percent_income` total debt-to-income ratio?**  
No. It represents the available loan-to-income measure. The dataset does not contain total debt, so the code does not claim or create total debt-to-income.

**10. Are the model scores calibrated or ready for automatic decisions?**  
No. The project does not calibrate scores, conduct external validation, or optimize thresholds against institution-specific costs. The dashboard is a prototype for analytical review, not an automated lending or fraud-blocking system.

**11. What happens when a dataset has missing values?**  
The quality workflow reports them. The training workflow uses imputers fitted on training rows, and the saved inference bundle applies those same fitted transformations to later rows. Missing targets are excluded rather than imputed.

**12. What is the most important next step?**  
Evaluate on external and later-period data, then calibrate scores and choose thresholds based on explicit false-positive and false-negative costs, with fairness, governance, and monitoring reviews.
