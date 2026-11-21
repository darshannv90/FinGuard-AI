"""Train, compare, and persist loan-default classifiers without data leakage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	average_precision_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import DatasetLoadError, load_loan_dataset
from src.loan_features import (
	LOAN_TARGET_COLUMN,
	LOAN_TARGET_MEANINGS,
	LoanFeatureEngineer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
SELECTION_METRIC = "PR-AUC"


@dataclass
class LoanModelBundle:
	"""Persistable inference pipeline with fitted loan feature preprocessing."""

	model_name: str
	feature_engineer: LoanFeatureEngineer
	preprocessing_pipeline: Pipeline
	estimator: Any
	random_state: int

	def _transform(self, frame: pd.DataFrame) -> np.ndarray:
		engineered = self.feature_engineer.transform(frame)
		return self.preprocessing_pipeline.transform(engineered)

	def predict(self, frame: pd.DataFrame) -> np.ndarray:
		"""Predict 1 for default and 0 for non-default."""
		return self.estimator.predict(self._transform(frame)).astype("int8")

	def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
		"""Return default-class probabilities for risk ranking."""
		transformed = self._transform(frame)
		default_index = list(self.estimator.classes_).index(1)
		return self.estimator.predict_proba(transformed)[:, default_index]


@dataclass
class LoanTrainingResult:
	"""Evaluation results and paths for a completed loan-model comparison."""

	results: pd.DataFrame
	best_model_name: str
	best_model: LoanModelBundle
	best_pr_auc: float
	training_rows: int
	test_rows: int
	dropped_unlabeled_rows: int
	dropped_duplicate_rows: int
	artifact_paths: dict[str, Path]
	selection_reason: str


def _prepare_labeled_data(
	frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, int, int]:
	if not isinstance(frame, pd.DataFrame):
		raise TypeError("frame must be a pandas DataFrame")
	if frame.columns.has_duplicates:
		raise ValueError("Dataset contains duplicate column names")
	if frame.empty:
		raise ValueError("Loan dataset is empty")
	if LOAN_TARGET_COLUMN not in frame.columns:
		raise ValueError(
			f"Loan dataset is missing target column '{LOAN_TARGET_COLUMN}'"
		)

	raw_target = frame[LOAN_TARGET_COLUMN]
	target = pd.to_numeric(raw_target, errors="coerce")
	if (raw_target.notna() & target.isna()).any():
		raise ValueError(f"'{LOAN_TARGET_COLUMN}' contains non-numeric target values")
	unsupported = target.notna() & ~target.isin(LOAN_TARGET_MEANINGS)
	if unsupported.any():
		values = sorted(target.loc[unsupported].unique().tolist())
		raise ValueError(
			f"'{LOAN_TARGET_COLUMN}' contains values outside the verified "
			f"0/1 encoding: {values}"
		)

	labeled = target.notna()
	dropped_unlabeled_rows = int((~labeled).sum())
	labeled_frame = frame.loc[labeled].copy()
	labels = target.loc[labeled].astype("int8")
	unique_rows = ~labeled_frame.duplicated(keep="first")
	dropped_duplicate_rows = int((~unique_rows).sum())
	labeled_frame = labeled_frame.loc[unique_rows].copy()
	labels = labels.loc[unique_rows]
	if labels.nunique() != 2 or int(labels.value_counts().min()) < 2:
		raise ValueError(
			"Need at least two labeled rows from each target class for a stratified split"
		)
	features = labeled_frame.drop(columns=[LOAN_TARGET_COLUMN]).copy()
	return features, labels, dropped_unlabeled_rows, dropped_duplicate_rows


def _make_preprocessing_pipeline(*, scale: bool) -> Pipeline:
	steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
	if scale:
		steps.append(("scaler", StandardScaler()))
	return Pipeline(steps)


def _evaluate_predictions(
	model_name: str,
	actual: pd.Series,
	predictions: np.ndarray,
	scores: np.ndarray,
) -> dict[str, Any]:
	true_nondefault, false_default, missed_default, detected_default = confusion_matrix(
		actual, predictions, labels=[0, 1]
	).ravel()
	return {
		"Model": model_name,
		"Precision": float(precision_score(actual, predictions, zero_division=0)),
		"Recall": float(recall_score(actual, predictions, zero_division=0)),
		"F1-score": float(f1_score(actual, predictions, zero_division=0)),
		"ROC-AUC": float(roc_auc_score(actual, scores)),
		"PR-AUC": float(average_precision_score(actual, scores)),
		"TN": int(true_nondefault),
		"FP": int(false_default),
		"FN": int(missed_default),
		"TP": int(detected_default),
		"Confusion Matrix": (
			f"[[{true_nondefault}, {false_default}], "
			f"[{missed_default}, {detected_default}]]"
		),
	}


def train_loan_models(
	frame: pd.DataFrame | None = None,
	*,
	data_dir: str | Path | None = None,
	models_dir: str | Path | None = None,
	test_size: float = 0.2,
	random_state: int = 42,
) -> LoanTrainingResult:
	"""Train and compare three default models on a stratified holdout.

	The verified target is ``loan_status`` (0 non-default, 1 default). Missing
	target labels are excluded; malformed or unsupported target values raise a
	clear ``ValueError``. The split happens before feature engineering and
	preprocessing are fit. Exact duplicate labeled rows are removed before the
	split to prevent identical records crossing train and test. Logistic
	Regression, Decision Tree, and Random Forest use class weights to account
	for the observed default imbalance.

	Average precision is reported as PR-AUC and is the primary selection
	metric because default is the minority class. Recall and F1-score break
	ties. Metrics use each estimator's default 0.5 decision threshold; lenders
	should choose an operating threshold using their own default/approval costs.
	"""
	if not 0 < test_size < 1:
		raise ValueError("test_size must be between 0 and 1")
	if frame is None:
		try:
			data = load_loan_dataset(data_dir=data_dir)
		except DatasetLoadError as error:
			raise DatasetLoadError(
				f"Cannot train loan models because the required loan dataset is unavailable: {error}"
			) from error
	else:
		data = frame.copy(deep=True)

	(
		features,
		target,
		dropped_unlabeled_rows,
		dropped_duplicate_rows,
	) = _prepare_labeled_data(data)
	if len(features) < 4:
		raise ValueError("At least four labeled loan rows are required")
	x_train, x_test, y_train, y_test = train_test_split(
		features,
		target,
		test_size=test_size,
		random_state=random_state,
		stratify=target,
	)
	if y_train.nunique() != 2 or y_test.nunique() != 2:
		raise ValueError(
			"The chosen test_size must leave both target classes in train and test splits"
		)

	feature_engineer = LoanFeatureEngineer().fit(x_train)
	x_train_features = feature_engineer.transform(x_train)
	x_test_features = feature_engineer.transform(x_test)
	if LOAN_TARGET_COLUMN in x_train_features.columns:
		raise ValueError("Target leakage detected: loan_status is present in features")
	if x_train_features.empty or not x_train_features.columns.tolist():
		raise ValueError("No loan features remain after feature engineering")

	model_configs: list[tuple[str, Any, bool]] = [
		(
			"Logistic Regression",
			LogisticRegression(
				class_weight="balanced",
				max_iter=1000,
				random_state=random_state,
			),
			True,
		),
		(
			"Decision Tree",
			DecisionTreeClassifier(
				class_weight="balanced",
				random_state=random_state,
			),
			False,
		),
		(
			"Random Forest",
			RandomForestClassifier(
				n_estimators=100,
				class_weight="balanced_subsample",
				n_jobs=-1,
				random_state=random_state,
			),
			False,
		),
	]
	evaluation_rows: list[dict[str, Any]] = []
	bundles: dict[str, LoanModelBundle] = {}
	for model_name, estimator, scale in model_configs:
		preprocessing_pipeline = _make_preprocessing_pipeline(scale=scale)
		x_train_ready = preprocessing_pipeline.fit_transform(x_train_features)
		x_test_ready = preprocessing_pipeline.transform(x_test_features)
		estimator.fit(x_train_ready, y_train)
		predictions = estimator.predict(x_test_ready).astype("int8")
		default_index = list(estimator.classes_).index(1)
		scores = estimator.predict_proba(x_test_ready)[:, default_index]
		evaluation_rows.append(
			_evaluate_predictions(model_name, y_test, predictions, scores)
		)
		bundles[model_name] = LoanModelBundle(
			model_name=model_name,
			feature_engineer=feature_engineer,
			preprocessing_pipeline=preprocessing_pipeline,
			estimator=estimator,
			random_state=random_state,
		)

	results = pd.DataFrame(evaluation_rows)
	best_index = results.sort_values(
		["PR-AUC", "Recall", "F1-score"],
		ascending=[False, False, False],
		kind="stable",
	).index[0]
	best_model_name = str(results.loc[best_index, "Model"])
	best_pr_auc = float(results.loc[best_index, "PR-AUC"])
	results["Selected"] = results["Model"].eq(best_model_name)
	selection_reason = (
		"Selected the model with the highest PR-AUC (average precision), using "
		"recall and F1-score as tie-breakers because default is the minority class. "
		"Metrics use the default 0.5 threshold; changing the threshold trades "
		"precision against recall and requires a lender-specific cost policy."
	)
	best_model = bundles[best_model_name]

	output_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
	output_dir.mkdir(parents=True, exist_ok=True)
	model_path = output_dir / "loan_best_model.joblib"
	results_path = output_dir / "loan_model_evaluation_results.csv"
	summary_path = output_dir / "loan_training_summary.json"
	joblib.dump(best_model, model_path)
	results.to_csv(results_path, index=False)
	summary = {
		"target_column": LOAN_TARGET_COLUMN,
		"target_meanings": {str(key): value for key, value in LOAN_TARGET_MEANINGS.items()},
		"best_model": best_model_name,
		"selection_metric": SELECTION_METRIC,
		"best_pr_auc": best_pr_auc,
		"selection_reason": selection_reason,
		"random_state": random_state,
		"test_size": test_size,
		"training_rows": int(len(x_train)),
		"test_rows": int(len(x_test)),
		"dropped_unlabeled_rows": dropped_unlabeled_rows,
		"dropped_duplicate_rows": dropped_duplicate_rows,
		"classification_threshold": 0.5,
	}
	with summary_path.open("w", encoding="utf-8") as output_file:
		json.dump(summary, output_file, indent=2)
		output_file.write("\n")

	return LoanTrainingResult(
		results=results,
		best_model_name=best_model_name,
		best_model=best_model,
		best_pr_auc=best_pr_auc,
		training_rows=len(x_train),
		test_rows=len(x_test),
		dropped_unlabeled_rows=dropped_unlabeled_rows,
		dropped_duplicate_rows=dropped_duplicate_rows,
		artifact_paths={
			"model": model_path,
			"results": results_path,
			"summary": summary_path,
		},
		selection_reason=selection_reason,
	)