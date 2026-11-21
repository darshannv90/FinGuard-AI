"""Train, compare, and persist fraud classifiers without test-set leakage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
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

from src.data_loader import load_fraud_dataset
from src.fraud_features import FraudFeatureEngineer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
TARGET_COLUMN = "Class"
SELECTION_METRIC = "PR-AUC"
ProgressCallback = Callable[[float, str], None]


@dataclass
class FraudModelBundle:
	"""Persistable inference pipeline, including fitted feature transformations."""

	model_name: str
	feature_engineer: FraudFeatureEngineer
	preprocessing_pipeline: Pipeline
	estimator: Any
	random_state: int

	def _transform(self, frame: pd.DataFrame) -> np.ndarray:
		engineered = self.feature_engineer.transform(frame)
		return self.preprocessing_pipeline.transform(engineered)

	def predict(self, frame: pd.DataFrame) -> np.ndarray:
		"""Predict 1 for fraud and 0 for legitimate transactions."""
		predictions = self.estimator.predict(self._transform(frame))
		if self.model_name == "Isolation Forest":
			return (predictions == -1).astype("int8")
		return predictions.astype("int8")

	def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
		"""Return a score where larger values indicate stronger fraud evidence."""
		transformed = self._transform(frame)
		if self.model_name == "Isolation Forest":
			return -self.estimator.decision_function(transformed)
		fraud_index = list(self.estimator.classes_).index(1)
		return self.estimator.predict_proba(transformed)[:, fraud_index]


@dataclass
class FraudTrainingResult:
	"""Evaluation results and paths for a completed fraud model comparison."""

	results: pd.DataFrame
	best_model_name: str
	best_model: FraudModelBundle
	best_pr_auc: float
	training_rows: int
	test_rows: int
	dropped_unlabeled_rows: int
	dropped_duplicate_rows: int
	artifact_paths: dict[str, Path]
	selection_reason: str


def _notify(
	callback: ProgressCallback | None,
	progress: float,
	message: str,
) -> None:
	if callback is not None:
		callback(progress, message)


def _prepare_labeled_data(
	frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, int, int]:
	if not isinstance(frame, pd.DataFrame):
		raise TypeError("frame must be a pandas DataFrame")
	if frame.columns.has_duplicates:
		raise ValueError("Dataset contains duplicate column names")
	if TARGET_COLUMN not in frame.columns:
		raise ValueError(f"Dataset must contain the '{TARGET_COLUMN}' target column")

	labels = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
	invalid_labels = labels.notna() & ~labels.isin([0, 1])
	if invalid_labels.any():
		raise ValueError(f"'{TARGET_COLUMN}' must contain only binary 0/1 labels")

	labeled = labels.isin([0, 1])
	dropped_unlabeled_rows = int((~labeled).sum())
	labeled_frame = frame.loc[labeled].copy()
	target = labels.loc[labeled].astype("int8")
	unique_rows = ~labeled_frame.duplicated(keep="first")
	dropped_duplicate_rows = int((~unique_rows).sum())
	labeled_frame = labeled_frame.loc[unique_rows].copy()
	target = target.loc[unique_rows]
	features = labeled_frame.drop(columns=[TARGET_COLUMN]).copy()
	if features.empty:
		raise ValueError("Dataset contains no rows with valid fraud labels")
	if target.nunique() != 2 or int(target.value_counts().min()) < 2:
		raise ValueError("Need at least two labeled rows from each class for a stratified split")
	if not any(
		pd.api.types.is_numeric_dtype(features[column].dtype)
		or pd.to_numeric(features[column], errors="coerce").notna().any()
		for column in features
	):
		raise ValueError("Dataset contains no usable numeric input features")
	return features, target, dropped_unlabeled_rows, dropped_duplicate_rows


def _make_preprocessing_pipeline(*, scale: bool) -> Pipeline:
	steps: list[tuple[str, Any]] = [
		("imputer", SimpleImputer(strategy="median")),
	]
	if scale:
		steps.append(("scaler", StandardScaler()))
	return Pipeline(steps)


def _evaluate_predictions(
	model_name: str,
	y_true: pd.Series,
	predictions: np.ndarray,
	scores: np.ndarray,
) -> dict[str, Any]:
	tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
	return {
		"Model": model_name,
		"Precision": float(precision_score(y_true, predictions, zero_division=0)),
		"Recall": float(recall_score(y_true, predictions, zero_division=0)),
		"F1-score": float(f1_score(y_true, predictions, zero_division=0)),
		"ROC-AUC": float(roc_auc_score(y_true, scores)),
		"PR-AUC": float(average_precision_score(y_true, scores)),
		"TN": int(tn),
		"FP": int(fp),
		"FN": int(fn),
		"TP": int(tp),
		"Confusion Matrix": f"[[{tn}, {fp}], [{fn}, {tp}]]",
	}


def train_fraud_models(
	frame: pd.DataFrame | None = None,
	*,
	models_dir: str | Path | None = None,
	test_size: float = 0.2,
	random_state: int = 42,
	progress_callback: ProgressCallback | None = None,
) -> FraudTrainingResult:
	"""Fit and evaluate three models; persist the PR-AUC-selected pipeline.

	The input is split before feature engineering or preprocessing is fitted.
	Exact duplicate labeled rows are removed before the split so identical
	records cannot occur in both train and test.
	Logistic Regression and Random Forest use class weights rather than
	resampling. Isolation Forest is fit without labels and maps outliers to
	fraud; its anomaly score is negated so larger scores consistently indicate
	fraud. Average precision is reported as PR-AUC, an appropriate primary
	selection metric for this highly imbalanced dataset.
	"""
	if not 0 < test_size < 1:
		raise ValueError("test_size must be between 0 and 1")
	_notify(progress_callback, 0.02, "Loading and validating the labeled fraud dataset")
	data = load_fraud_dataset() if frame is None else frame.copy(deep=True)
	(
		features,
		target,
		dropped_unlabeled_rows,
		dropped_duplicate_rows,
	) = _prepare_labeled_data(data)
	if len(features) < 4:
		raise ValueError("At least four labeled rows are required for a train/test split")

	_notify(progress_callback, 0.08, "Creating a stratified train/test split")
	x_train, x_test, y_train, y_test = train_test_split(
		features,
		target,
		test_size=test_size,
		random_state=random_state,
		stratify=target,
	)
	if y_train.nunique() != 2 or y_test.nunique() != 2:
		raise ValueError("The chosen test_size must leave both classes in train and test splits")

	_notify(progress_callback, 0.15, "Fitting fraud features on training rows only")
	feature_engineer = FraudFeatureEngineer().fit(x_train)
	x_train_features = feature_engineer.transform(x_train)
	x_test_features = feature_engineer.transform(x_test)
	if x_train_features.empty or not x_train_features.columns.tolist():
		raise ValueError("No usable input features remain after fraud feature engineering")
	if TARGET_COLUMN in x_train_features.columns:
		raise ValueError("Target leakage detected: Class is present in engineered inputs")

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
			"Random Forest",
			RandomForestClassifier(
				n_estimators=100,
				class_weight="balanced_subsample",
				n_jobs=-1,
				random_state=random_state,
			),
			False,
		),
		(
			"Isolation Forest",
			IsolationForest(
				contamination="auto",
				n_estimators=100,
				n_jobs=-1,
				random_state=random_state,
			),
			True,
		),
	]

	evaluation_rows: list[dict[str, Any]] = []
	bundles: dict[str, FraudModelBundle] = {}
	for index, (model_name, estimator, scale) in enumerate(model_configs):
		progress = 0.2 + index * 0.24
		_notify(progress_callback, progress, f"Training {model_name}")
		preprocessing_pipeline = _make_preprocessing_pipeline(scale=scale)
		x_train_ready = preprocessing_pipeline.fit_transform(x_train_features)
		x_test_ready = preprocessing_pipeline.transform(x_test_features)
		if model_name == "Isolation Forest":
			estimator.fit(x_train_ready)
			predictions = (estimator.predict(x_test_ready) == -1).astype("int8")
			scores = -estimator.decision_function(x_test_ready)
		else:
			estimator.fit(x_train_ready, y_train)
			predictions = estimator.predict(x_test_ready).astype("int8")
			fraud_index = list(estimator.classes_).index(1)
			scores = estimator.predict_proba(x_test_ready)[:, fraud_index]

		evaluation_rows.append(
			_evaluate_predictions(model_name, y_test, predictions, scores)
		)
		bundles[model_name] = FraudModelBundle(
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
		"Selected the model with the highest PR-AUC (average precision), "
		"using recall and F1-score as tie-breakers. PR-AUC emphasizes fraud "
		"detection quality under severe class imbalance."
	)
	best_model = bundles[best_model_name]

	output_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
	output_dir.mkdir(parents=True, exist_ok=True)
	model_path = output_dir / "fraud_best_model.joblib"
	results_path = output_dir / "fraud_evaluation_results.csv"
	summary_path = output_dir / "fraud_training_summary.json"
	_notify(progress_callback, 0.94, "Saving the selected model and evaluation results")
	joblib.dump(best_model, model_path)
	results.to_csv(results_path, index=False)
	summary = {
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
	}
	with summary_path.open("w", encoding="utf-8") as output_file:
		json.dump(summary, output_file, indent=2)
		output_file.write("\n")
	_notify(progress_callback, 1.0, "Training and evaluation complete")

	return FraudTrainingResult(
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