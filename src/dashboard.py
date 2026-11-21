"""Shared data validation and scoring helpers for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import (
	DATASETS,
	RAW_DATA_DIR,
	DatasetLoadError,
	load_dataset,
	validate_csv_headers,
)
from src.evaluation import DATASET_ARTIFACTS, MODELS_DIR, load_best_model_bundle


TARGET_COLUMNS = {"fraud": "Class", "loan": "loan_status"}


@st.cache_data(show_spinner=False)
def _cached_dataset(
	dataset: str,
	path: str,
	modified_ns: int,
	file_size: int,
) -> pd.DataFrame:
	del modified_ns, file_size
	return load_dataset(dataset, data_dir=Path(path).parent)


def load_dashboard_dataset(dataset: str) -> pd.DataFrame:
	"""Load and cache a local dataset until its file changes."""
	normalized_dataset = dataset.strip().lower()
	if normalized_dataset not in DATASETS:
		raise DatasetLoadError(f"Unknown dataset '{dataset}'.")
	path = RAW_DATA_DIR / DATASETS[normalized_dataset].filename
	if not path.is_file():
		return load_dataset(normalized_dataset)
	file_stat = path.stat()
	return _cached_dataset(
		normalized_dataset,
		str(path),
		file_stat.st_mtime_ns,
		file_stat.st_size,
	)


@st.cache_resource(show_spinner=False)
def _cached_model(
	dataset: str,
	models_directory: str,
	modified_ns: int,
	file_size: int,
) -> Any:
	del modified_ns, file_size
	return load_best_model_bundle(dataset, models_dir=models_directory)


def load_dashboard_model(dataset: str) -> Any:
	"""Load and cache the selected model, refreshing after the artifact changes."""
	normalized_dataset = dataset.strip().lower()
	if normalized_dataset not in {"fraud", "loan"}:
		raise ValueError("dataset must be either 'fraud' or 'loan'")
	model_filename = (
		"fraud_best_model.joblib"
		if normalized_dataset == "fraud"
		else "loan_best_model.joblib"
	)
	model_path = MODELS_DIR / model_filename
	if not model_path.is_file():
		return load_best_model_bundle(normalized_dataset)
	file_stat = model_path.stat()
	return _cached_model(
		normalized_dataset,
		str(MODELS_DIR),
		file_stat.st_mtime_ns,
		file_stat.st_size,
	)


def clear_dashboard_model_cache() -> None:
	"""Release cached bundles after a model has been explicitly retrained."""
	_cached_model.clear()


def model_artifact_revision(dataset: str) -> tuple[int, int]:
	"""Return a file revision token for invalidating displayed predictions."""
	normalized_dataset = dataset.strip().lower()
	if normalized_dataset not in DATASET_ARTIFACTS:
		raise ValueError("dataset must be either 'fraud' or 'loan'")
	model_path = MODELS_DIR / DATASET_ARTIFACTS[normalized_dataset][1]
	file_stat = model_path.stat()
	return file_stat.st_mtime_ns, file_stat.st_size


def read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
	"""Read an uploaded CSV in memory and raise actionable format errors."""
	if uploaded_file is None:
		raise ValueError("Choose a CSV file to continue.")
	validate_csv_headers(uploaded_file, "uploaded CSV")
	try:
		uploaded_file.seek(0)
		frame = pd.read_csv(uploaded_file, low_memory=False)
	except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as error:
		raise DatasetLoadError("The uploaded file could not be read as a valid CSV.") from error
	if frame.empty:
		raise DatasetLoadError("The uploaded CSV contains no data rows.")
	if frame.columns.has_duplicates:
		raise DatasetLoadError("The uploaded CSV contains duplicate column names.")
	return frame


def csv_download_bytes(frame: pd.DataFrame) -> bytes:
	"""Serialize a prediction or cleaning report as downloadable UTF-8 CSV bytes."""
	if not isinstance(frame, pd.DataFrame):
		raise TypeError("frame must be a pandas DataFrame")
	return frame.to_csv(index=False).encode("utf-8")


def prepare_model_inputs(
	frame: pd.DataFrame,
	bundle: Any,
	dataset: str,
) -> tuple[pd.DataFrame, list[str]]:
	"""Validate required trained-model fields and remove any label from inputs."""
	normalized_dataset = dataset.strip().lower()
	if normalized_dataset not in TARGET_COLUMNS:
		raise ValueError("dataset must be either 'fraud' or 'loan'")
	if not isinstance(frame, pd.DataFrame):
		raise TypeError("frame must be a pandas DataFrame")
	if frame.empty:
		raise ValueError("The dataset contains no rows to score.")
	if frame.columns.has_duplicates:
		raise ValueError("The dataset contains duplicate column names.")

	engineer = getattr(bundle, "feature_engineer", None)
	if engineer is None:
		raise ValueError("The saved model bundle does not include its fitted feature engineer.")
	required_columns = list(getattr(engineer, "numeric_columns_", []))
	if normalized_dataset == "loan":
		required_columns.extend(getattr(engineer, "categorical_columns_", []))
	required_columns = list(dict.fromkeys(required_columns))
	missing_columns = [column for column in required_columns if column not in frame.columns]
	if missing_columns:
		raise ValueError(
			"Dataset is missing trained model input columns: "
			+ ", ".join(map(str, missing_columns))
		)

	target_column = TARGET_COLUMNS[normalized_dataset]
	model_inputs = frame.drop(columns=[target_column], errors="ignore").copy(deep=True)
	ignored_columns = [
		column for column in model_inputs.columns if column not in required_columns
	]
	return model_inputs, ignored_columns


def inspect_input_quality(
	frame: pd.DataFrame,
	numeric_columns: list[str] | None = None,
) -> dict[str, Any]:
	"""Count missing, duplicate, invalid numeric, and IQR-outlier input records."""
	columns = numeric_columns or frame.select_dtypes(include="number").columns.tolist()
	invalid_by_column: dict[str, int] = {}
	outliers_by_column: dict[str, int] = {}
	invalid_row_indexes: set[Any] = set()
	outlier_row_indexes: set[Any] = set()

	for column in columns:
		if column not in frame.columns:
			continue
		original = frame[column]
		numeric = pd.to_numeric(original, errors="coerce").replace(
			[float("inf"), float("-inf")], np.nan
		)
		invalid = original.notna() & numeric.isna()
		invalid_count = int(invalid.sum())
		invalid_by_column[column] = invalid_count
		invalid_row_indexes.update(frame.index[invalid].tolist())

		finite = numeric.dropna()
		if finite.nunique() > 2:
			q1, q3 = finite.quantile([0.25, 0.75])
			spread = float(q3 - q1)
			outlier_mask = numeric.lt(q1 - 1.5 * spread) | numeric.gt(q3 + 1.5 * spread)
			outlier_mask = outlier_mask.fillna(False)
			outlier_count = int(outlier_mask.sum())
			outliers_by_column[column] = outlier_count
			outlier_row_indexes.update(frame.index[outlier_mask].tolist())
		else:
			outliers_by_column[column] = 0

	return {
		"rows": int(len(frame)),
		"columns": int(frame.shape[1]),
		"missing_cells": int(frame.isna().sum().sum()),
		"duplicate_rows": int(frame.duplicated().sum()),
		"invalid_numeric_cells": int(sum(invalid_by_column.values())),
		"invalid_rows": len(invalid_row_indexes),
		"outlier_rows": len(outlier_row_indexes),
		"invalid_by_column": invalid_by_column,
		"outliers_by_column": outliers_by_column,
	}


def categorize_risk_scores(
	scores: np.ndarray | pd.Series,
	medium_cutoff: float,
	high_cutoff: float,
) -> np.ndarray:
	"""Bucket probability scores using explicit low/medium/high boundaries."""
	if not 0 <= medium_cutoff < high_cutoff <= 1:
		raise ValueError("Risk cutoffs must satisfy 0 <= medium < high <= 1")
	values = np.asarray(scores, dtype=float)
	if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
		raise ValueError("Risk scores must be finite probabilities between 0 and 1")
	return np.where(
		values >= high_cutoff,
		"High",
		np.where(values >= medium_cutoff, "Medium", "Low"),
	)
