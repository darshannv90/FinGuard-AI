"""Load and validate saved model evaluations and expose model importance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_ARTIFACTS = {
	"fraud": ("fraud_evaluation_results.csv", "fraud_best_model.joblib"),
	"loan": ("loan_model_evaluation_results.csv", "loan_best_model.joblib"),
}
BASE_METRICS = ("Precision", "Recall", "F1-score")
OPTIONAL_METRICS = ("ROC-AUC", "PR-AUC")
CONFUSION_COLUMNS = ("TN", "FP", "FN", "TP")


def calculate_confusion_metrics(
	tn: int,
	fp: int,
	fn: int,
	tp: int,
) -> dict[str, float | int]:
	"""Calculate standard binary metrics and error rates from confusion counts."""
	counts = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
	for name, value in counts.items():
		if isinstance(value, (bool, np.bool_)):
			raise ValueError(f"{name} must be a nonnegative integer")
		try:
			integer_value = int(value)
		except (TypeError, ValueError, OverflowError) as error:
			raise ValueError(f"{name} must be a nonnegative integer") from error
		if integer_value < 0 or integer_value != value:
			raise ValueError(f"{name} must be a nonnegative integer")
		counts[name] = integer_value

	precision = counts["TP"] / (counts["TP"] + counts["FP"]) if counts["TP"] + counts["FP"] else 0.0
	recall = counts["TP"] / (counts["TP"] + counts["FN"]) if counts["TP"] + counts["FN"] else 0.0
	f1_score = (
		2 * precision * recall / (precision + recall)
		if precision + recall
		else 0.0
	)
	false_positive_rate = (
		counts["FP"] / (counts["FP"] + counts["TN"])
		if counts["FP"] + counts["TN"]
		else 0.0
	)
	false_negative_rate = (
		counts["FN"] / (counts["FN"] + counts["TP"])
		if counts["FN"] + counts["TP"]
		else 0.0
	)
	return {
		**counts,
		"Precision": precision,
		"Recall": recall,
		"F1-score": f1_score,
		"False positive rate": false_positive_rate,
		"False negative rate": false_negative_rate,
	}


def load_evaluation_results(
	dataset: str,
	*,
	models_dir: str | Path | None = None,
) -> pd.DataFrame:
	"""Load one saved comparison table and verify its metric/count consistency."""
	normalized_dataset = dataset.strip().lower()
	if normalized_dataset not in DATASET_ARTIFACTS:
		raise ValueError("dataset must be either 'fraud' or 'loan'")
	output_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
	results_path = output_dir / DATASET_ARTIFACTS[normalized_dataset][0]
	if not results_path.is_file():
		raise FileNotFoundError(f"Saved {normalized_dataset} evaluation was not found: {results_path}")
	try:
		results = pd.read_csv(results_path)
	except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
		raise ValueError(f"Could not read saved evaluation file {results_path}") from error
	if results.empty:
		raise ValueError(f"Saved {normalized_dataset} evaluation contains no model rows")
	if results.columns.has_duplicates:
		raise ValueError("Evaluation file contains duplicate column names")

	required_columns = {"Model", *BASE_METRICS, *CONFUSION_COLUMNS}
	missing_columns = sorted(required_columns.difference(results.columns))
	if missing_columns:
		raise ValueError(
			"Evaluation file is missing required columns: "
			+ ", ".join(missing_columns)
		)
	results = results.copy()
	results["Model"] = results["Model"].astype("string").str.strip()
	if results["Model"].isna().any() or results["Model"].eq("").any():
		raise ValueError("Evaluation file contains a blank model name")
	if results["Model"].duplicated().any():
		raise ValueError("Evaluation file contains duplicate model names")

	for column in (*BASE_METRICS, *OPTIONAL_METRICS, *CONFUSION_COLUMNS):
		if column not in results:
			results[column] = np.nan
		raw_values = results[column]
		numeric_values = pd.to_numeric(raw_values, errors="coerce")
		if (raw_values.notna() & numeric_values.isna()).any():
			raise ValueError(f"{column} contains a non-numeric value")
		results[column] = numeric_values

	for index, row in results.iterrows():
		for column in BASE_METRICS:
			value = row[column]
			if pd.isna(value) or not np.isfinite(value) or not 0 <= value <= 1:
				raise ValueError(f"{column} for {row['Model']} must be a finite value in [0, 1]")
		for column in OPTIONAL_METRICS:
			value = row[column]
			if pd.notna(value) and (not np.isfinite(value) or not 0 <= value <= 1):
				raise ValueError(f"{column} for {row['Model']} must be in [0, 1] when present")
		for column in CONFUSION_COLUMNS:
			value = row[column]
			if pd.isna(value) or not np.isfinite(value) or value < 0 or value != int(value):
				raise ValueError(f"{column} for {row['Model']} must be a nonnegative integer")

		calculated = calculate_confusion_metrics(
			*(int(row[column]) for column in CONFUSION_COLUMNS)
		)
		for column in BASE_METRICS:
			if not np.isclose(row[column], calculated[column], rtol=0, atol=1e-6):
				raise ValueError(
					f"{column} for {row['Model']} does not match its saved confusion counts"
				)

	if "Selected" not in results:
		results["Selected"] = False
	else:
		selected = results["Selected"]
		if not pd.api.types.is_bool_dtype(selected):
			normalized = selected.astype("string").str.casefold()
			if not normalized.isin(["true", "false"]).all():
				raise ValueError("Selected must contain only true/false values")
			selected = normalized.eq("true")
		results["Selected"] = selected.astype(bool)
	if int(results["Selected"].sum()) > 1:
		raise ValueError("Evaluation file marks more than one selected model")

	results.insert(0, "Dataset", normalized_dataset.title())
	results["Confusion Matrix"] = results.apply(
		lambda row: f"[[{int(row['TN'])}, {int(row['FP'])}], "
		f"[{int(row['FN'])}, {int(row['TP'])}]]",
		axis=1,
	)
	return results


def load_best_model_bundle(
	dataset: str,
	*,
	models_dir: str | Path | None = None,
) -> Any:
	"""Load the saved selected-model bundle for feature-importance inspection."""
	normalized_dataset = dataset.strip().lower()
	if normalized_dataset not in DATASET_ARTIFACTS:
		raise ValueError("dataset must be either 'fraud' or 'loan'")
	output_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
	model_path = output_dir / DATASET_ARTIFACTS[normalized_dataset][1]
	if not model_path.is_file():
		raise FileNotFoundError(f"Saved {normalized_dataset} model was not found: {model_path}")
	bundle = joblib.load(model_path)
	if not hasattr(bundle, "estimator") or not hasattr(bundle, "feature_engineer"):
		raise ValueError(f"Saved {normalized_dataset} model bundle has an unsupported format")
	return bundle


def feature_importance_from_bundle(
	bundle: Any,
) -> tuple[pd.DataFrame, str | None]:
	"""Return supported native importances or absolute linear coefficients."""
	estimator = bundle.estimator
	if hasattr(estimator, "feature_importances_"):
		importances = np.asarray(estimator.feature_importances_, dtype=float)
		method = "Tree impurity-based feature importance"
	elif hasattr(estimator, "coef_"):
		coefficients = np.asarray(estimator.coef_, dtype=float)
		importances = np.abs(coefficients).mean(axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
		method = "Absolute model coefficients"
	else:
		return pd.DataFrame(columns=["Feature", "Importance"]), None

	feature_engineer = bundle.feature_engineer
	names = getattr(feature_engineer, "feature_names_out_", None)
	if names is None:
		if not hasattr(feature_engineer, "numeric_columns_"):
			raise ValueError("Saved feature engineer does not expose its output feature names")
		empty_input = pd.DataFrame(columns=feature_engineer.numeric_columns_)
		names = feature_engineer.transform(empty_input).columns.tolist()
	if len(names) != len(importances):
		raise ValueError(
			"Saved model importance count does not match its engineered feature names"
		)
	if not np.isfinite(importances).all():
		raise ValueError("Saved model contains non-finite feature importance values")
	feature_table = pd.DataFrame({"Feature": list(names), "Importance": importances})
	feature_table = feature_table.sort_values(
		"Importance", ascending=False, kind="stable"
	).reset_index(drop=True)
	return feature_table, method