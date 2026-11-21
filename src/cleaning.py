"""Reusable, non-destructive data cleaning and preprocessing utilities."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd


OutlierAction = Literal["flag", "cap", "none"]
CategoricalImputation = Literal["most_frequent", "unknown"]
ScaleMethod = Literal["standard", "minmax"]


@dataclass(frozen=True)
class CleaningConfig:
	"""Options for a fitted cleaning pipeline.

	Learned values are calculated only in ``fit``. Fit on training data, then
	call ``transform`` for validation, test, or later production data.
	"""

	remove_duplicates: bool = True
	numeric_imputation: Literal["median"] = "median"
	categorical_imputation: CategoricalImputation = "most_frequent"
	unknown_category: str = "Unknown"
	normalize_categories: bool = True
	categorical_case: Literal["lower", "upper", "preserve"] = "lower"
	numeric_coercion_threshold: float = 0.6
	dtype_overrides: dict[str, Literal["numeric", "categorical"]] = field(default_factory=dict)
	outlier_action: OutlierAction = "flag"
	outlier_multiplier: float = 1.5
	scale_numeric: bool = False
	scale_method: ScaleMethod = "standard"
	encode_categorical: bool = False
	target_columns: tuple[str, ...] = ()

	def __post_init__(self) -> None:
		if not 0 <= self.numeric_coercion_threshold <= 1:
			raise ValueError("numeric_coercion_threshold must be between 0 and 1")
		if self.outlier_multiplier < 0:
			raise ValueError("outlier_multiplier must be non-negative")
		if self.outlier_action not in {"flag", "cap", "none"}:
			raise ValueError("outlier_action must be 'flag', 'cap', or 'none'")
		if self.categorical_case not in {"lower", "upper", "preserve"}:
			raise ValueError("categorical_case must be 'lower', 'upper', or 'preserve'")
		if self.scale_method not in {"standard", "minmax"}:
			raise ValueError("scale_method must be 'standard' or 'minmax'")
		if self.categorical_imputation not in {"most_frequent", "unknown"}:
			raise ValueError("categorical_imputation must be 'most_frequent' or 'unknown'")
		invalid_overrides = set(self.dtype_overrides.values()) - {"numeric", "categorical"}
		if invalid_overrides:
			raise ValueError("dtype_overrides values must be 'numeric' or 'categorical'")


@dataclass
class CleaningResult:
	"""Cleaned copy and the reports, flags, and log that explain its changes."""

	data: pd.DataFrame
	before_report: dict[str, Any]
	after_report: dict[str, Any]
	outlier_flags: pd.DataFrame
	corrupted_categories: pd.DataFrame
	log: list[str]


def normalize_category_value(
	value: Any,
	case: Literal["lower", "upper", "preserve"] = "lower",
) -> Any:
	"""Trim and collapse whitespace in one category, optionally normalize case."""
	if pd.isna(value):
		return value
	text = re.sub(r"\s+", " ", str(value)).strip()
	if case == "lower":
		return text.casefold()
	if case == "upper":
		return text.upper()
	return text


def detect_corrupted_categories(
	frame: pd.DataFrame,
	allowed_categories: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
) -> pd.DataFrame:
	"""Find suspicious category values and optional values outside an allow-list."""
	allowed_categories = allowed_categories or {}
	findings: list[dict[str, Any]] = []
	columns = frame.select_dtypes(include=["object", "category", "string"]).columns
	for column in columns:
		allowed = allowed_categories.get(str(column))
		allowed_values = set(allowed) if allowed is not None else None
		for value in frame[column].dropna().unique():
			text = str(value)
			reasons: list[str] = []
			if not text.strip():
				reasons.append("blank category")
			if "\ufffd" in text:
				reasons.append("replacement character")
			if any(unicodedata.category(char) == "Cc" for char in text):
				reasons.append("control character")
			if allowed_values is not None and text not in allowed_values:
				reasons.append("not in allowed categories")
			if reasons:
				findings.append(
					{"column": column, "value": value, "reason": "; ".join(reasons)}
				)
	return pd.DataFrame(findings, columns=["column", "value", "reason"])


def build_quality_report(frame: pd.DataFrame) -> dict[str, Any]:
	"""Summarize shape, types, missingness, duplicates, and column families."""
	missing_counts = frame.isna().sum().astype(int)
	return {
		"rows": int(len(frame)),
		"columns": int(frame.shape[1]),
		"column_names": frame.columns.tolist(),
		"data_types": frame.dtypes.astype(str).to_dict(),
		"missing_counts": missing_counts.to_dict(),
		"missing_percentages": (frame.isna().mean() * 100).round(2).to_dict(),
		"missing_cells": int(missing_counts.sum()),
		"duplicate_rows": int(frame.duplicated().sum()),
		"numeric_columns": frame.select_dtypes(include="number").columns.tolist(),
		"categorical_columns": frame.select_dtypes(
			include=["object", "category", "string", "bool"]
		).columns.tolist(),
	}


class DataCleaningPipeline:
	"""Fit reusable cleaning rules on training data and apply them to copies."""

	def __init__(self, config: CleaningConfig | None = None) -> None:
		self.config = config or CleaningConfig()
		self.is_fitted = False
		self.columns_: list[Any] = []
		self.numeric_columns_: list[Any] = []
		self.categorical_columns_: list[Any] = []
		self.numeric_medians_: dict[Any, float] = {}
		self.categorical_fills_: dict[Any, str] = {}
		self.categorical_fallback_columns_: set[Any] = set()
		self.outlier_bounds_: dict[Any, tuple[float, float]] = {}
		self.scaling_parameters_: dict[Any, tuple[float, float]] = {}
		self.category_values_: dict[Any, list[str]] = {}
		self.fit_log_: list[str] = []

	def _validate_frame(self, frame: pd.DataFrame, *, fitting: bool) -> None:
		if not isinstance(frame, pd.DataFrame):
			raise TypeError("frame must be a pandas DataFrame")
		if frame.columns.has_duplicates:
			raise ValueError("DataFrame contains duplicate column names")
		if fitting and frame.empty:
			raise ValueError("Cannot fit the cleaning pipeline on an empty DataFrame")
		unknown_overrides = set(self.config.dtype_overrides) - set(frame.columns)
		if fitting and unknown_overrides:
			self.fit_log_.append(
				"Ignored dtype overrides for missing columns: "
				+ ", ".join(sorted(unknown_overrides))
			)

	def _infer_column_families(self, frame: pd.DataFrame) -> None:
		self.numeric_columns_ = []
		self.categorical_columns_ = []
		for column in frame.columns:
			series = frame[column]
			override = self.config.dtype_overrides.get(str(column))
			if override == "numeric":
				self.numeric_columns_.append(column)
			elif override == "categorical":
				self.categorical_columns_.append(column)
			elif pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
				self.numeric_columns_.append(column)
			elif pd.api.types.is_bool_dtype(series.dtype) or isinstance(
				series.dtype, pd.CategoricalDtype
			):
				self.categorical_columns_.append(column)
			elif pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(
				series.dtype
			):
				converted = pd.to_numeric(series, errors="coerce")
				non_missing = int(series.notna().sum())
				numeric_ratio = float(converted.notna().sum() / non_missing) if non_missing else 0.0
				if non_missing and numeric_ratio >= self.config.numeric_coercion_threshold:
					self.numeric_columns_.append(column)
					self.fit_log_.append(
						f"Validated '{column}' as numeric ({numeric_ratio:.0%} parseable values)."
					)
				else:
					self.categorical_columns_.append(column)
			else:
				self.categorical_columns_.append(column)

	def fit(self, training_frame: pd.DataFrame) -> DataCleaningPipeline:
		"""Learn imputers, bounds, categories, and scalers from training rows only."""
		self.fit_log_ = []
		self._validate_frame(training_frame, fitting=True)
		self.columns_ = training_frame.columns.tolist()
		self._infer_column_families(training_frame)
		self.numeric_medians_ = {}
		self.categorical_fills_ = {}
		self.categorical_fallback_columns_ = set()
		self.outlier_bounds_ = {}
		self.scaling_parameters_ = {}
		self.category_values_ = {}

		for column in self.numeric_columns_:
			if str(column) in self.config.target_columns:
				continue
			values = pd.to_numeric(training_frame[column], errors="coerce").replace(
				[float("inf"), float("-inf")], np.nan
			)
			median = values.median()
			self.numeric_medians_[column] = float(median) if pd.notna(median) else 0.0
			if pd.isna(median):
				self.fit_log_.append(
					f"'{column}' has no numeric training values; numeric imputation will use 0."
				)

			if (
				self.config.outlier_action != "none"
				and str(column) not in self.config.target_columns
			):
				finite = values.dropna()
				if finite.nunique() > 2:
					q1, q3 = finite.quantile([0.25, 0.75])
					spread = float(q3 - q1)
					self.outlier_bounds_[column] = (
						float(q1 - self.config.outlier_multiplier * spread),
						float(q3 + self.config.outlier_multiplier * spread),
					)

		for column in self.categorical_columns_:
			if str(column) in self.config.target_columns:
				continue
			values = training_frame[column].dropna().map(self._normalize_value)
			if self.config.categorical_imputation == "unknown":
				fill = self._normalize_value(self.config.unknown_category)
				self.categorical_fallback_columns_.add(column)
			elif values.empty:
				fill = self._normalize_value(self.config.unknown_category)
				self.categorical_fallback_columns_.add(column)
				self.fit_log_.append(
					f"'{column}' has no non-missing training categories; imputation will use "
					f"'{fill}'."
				)
			else:
				fill = str(values.mode(dropna=True).iloc[0])
			self.categorical_fills_[column] = fill
			if self.config.encode_categorical and str(column) not in self.config.target_columns:
				self.category_values_[column] = sorted(values.unique().tolist())

		if self.config.scale_numeric:
			for column in self.numeric_columns_:
				if str(column) in self.config.target_columns:
					continue
				values = pd.to_numeric(training_frame[column], errors="coerce").replace(
					[float("inf"), float("-inf")], np.nan
				)
				values = values.fillna(self.numeric_medians_[column])
				if self.config.outlier_action == "cap" and column in self.outlier_bounds_:
					values = values.clip(*self.outlier_bounds_[column])
				if self.config.scale_method == "standard":
					center, scale = float(values.mean()), float(values.std(ddof=0))
				else:
					center = float(values.min())
					scale = float(values.max() - values.min())
				self.scaling_parameters_[column] = (center, scale if scale else 1.0)

		self.is_fitted = True
		self.fit_log_.append(
				f"Fitted preprocessing values on {len(training_frame):,} rows. For model workflows, "
				"fit on training rows only and transform validation or test rows."
			)
		return self

	def _normalize_value(self, value: Any) -> Any:
		if not self.config.normalize_categories:
			return value
		return normalize_category_value(value, self.config.categorical_case)

	def fit_transform(self, training_frame: pd.DataFrame) -> CleaningResult:
		"""Fit on a frame and clean a copy; use only for training or EDA workflows."""
		return self.fit(training_frame).transform(training_frame)

	def transform(self, frame: pd.DataFrame) -> CleaningResult:
		"""Clean a new frame using fitted training statistics; never mutate input."""
		if not self.is_fitted:
			raise RuntimeError("Call fit(training_frame) before transform(frame)")
		self._validate_frame(frame, fitting=False)
		before = build_quality_report(frame)
		cleaned = frame.copy(deep=True)
		log = list(self.fit_log_)
		corrupted = detect_corrupted_categories(frame)
		if len(corrupted):
			log.append(f"Detected {len(corrupted):,} suspicious categorical values before cleaning.")
		else:
			log.append("No corrupted categorical values were detected by automatic checks.")

		missing_columns = [column for column in self.columns_ if column not in cleaned.columns]
		for column in missing_columns:
			cleaned[column] = np.nan
		if missing_columns:
			log.append(
				"Added missing training columns as null values: "
				+ ", ".join(map(str, missing_columns))
			)
		extra_columns = [column for column in cleaned.columns if column not in self.columns_]
		if extra_columns:
			log.append(
				"Preserved unexpected columns without fitted transformations: "
				+ ", ".join(map(str, extra_columns))
			)

		for column in self.numeric_columns_:
			if column not in cleaned.columns:
				continue
			original_non_missing = int(cleaned[column].notna().sum())
			converted = pd.to_numeric(cleaned[column], errors="coerce").replace(
				[float("inf"), float("-inf")], np.nan
			)
			coercion_failures = original_non_missing - int(converted.notna().sum())
			if coercion_failures:
				log.append(f"Converted {coercion_failures:,} invalid values to missing in '{column}'.")
			cleaned[column] = converted

		for column in self.categorical_columns_:
			if column not in cleaned.columns:
				continue
			cleaned[column] = cleaned[column].astype("object")
			if self.config.normalize_categories and str(column) not in self.config.target_columns:
				original_values = cleaned[column]
				normalized_values = original_values.map(self._normalize_value).astype("object")
				changed_values = int(
					(
						original_values.notna()
						& normalized_values.notna()
						& original_values.astype("string").ne(normalized_values.astype("string"))
					).sum()
				)
				cleaned[column] = normalized_values
				log.append(
					f"Normalized category spacing/case in '{column}'; "
					f"{changed_values:,} values changed."
				)
			if str(column) not in self.config.target_columns:
				missing_count = int(cleaned[column].isna().sum())
				cleaned[column] = cleaned[column].fillna(self.categorical_fills_[column])
				if missing_count:
					strategy = (
						"Unknown fallback"
						if column in self.categorical_fallback_columns_
						else "most frequent training category"
					)
					log.append(
						f"Imputed {missing_count:,} missing categorical values in '{column}' "
						f"with the {strategy}."
					)

		for column in self.numeric_columns_:
			if column in cleaned.columns and str(column) not in self.config.target_columns:
				missing_count = int(cleaned[column].isna().sum())
				cleaned[column] = cleaned[column].fillna(self.numeric_medians_[column])
				if missing_count:
					log.append(
						f"Imputed {missing_count:,} missing numeric values in '{column}' "
						"with its training median."
					)

		missing_target_columns = [
			column
			for column in self.config.target_columns
			if column in cleaned.columns and cleaned[column].isna().any()
		]
		if missing_target_columns:
			log.append(
				"Preserved missing target labels without imputation: "
				+ ", ".join(missing_target_columns)
			)

		outlier_flags = pd.DataFrame(index=cleaned.index)
		for column, (lower, upper) in self.outlier_bounds_.items():
			if column not in cleaned.columns:
				continue
			flags = cleaned[column].lt(lower) | cleaned[column].gt(upper)
			outlier_flags[column] = flags.fillna(False)
			count = int(outlier_flags[column].sum())
			if count:
				log.append(
					f"Flagged {count:,} IQR outliers in '{column}' using training bounds "
					f"[{lower:.6g}, {upper:.6g}]."
				)
			if self.config.outlier_action == "cap" and count:
				cleaned[column] = cleaned[column].clip(lower, upper)
				log.append(f"Capped flagged values in '{column}' to its IQR bounds; no rows were deleted.")
		if not any(outlier_flags.columns):
			log.append("No numeric outlier columns were eligible for IQR flagging.")

		if self.config.remove_duplicates:
			keep_rows = ~cleaned.duplicated()
			duplicate_count = int((~keep_rows).sum())
			if duplicate_count:
				cleaned = cleaned.loc[keep_rows].copy()
				outlier_flags = outlier_flags.iloc[keep_rows.to_numpy()].copy()
			log.append(f"Removed {duplicate_count:,} duplicate rows after normalization and imputation.")
		else:
			log.append("Duplicate rows were detected but retained by configuration.")

		if self.config.scale_numeric:
			for column, (center, scale) in self.scaling_parameters_.items():
				if column in cleaned.columns:
					cleaned[column] = (cleaned[column] - center) / scale
			log.append(f"Scaled {len(self.scaling_parameters_)} numeric feature(s) using {self.config.scale_method} parameters fitted on training data.")

		if self.config.encode_categorical:
			encoded_frames: list[pd.DataFrame] = []
			for column, categories in self.category_values_.items():
				if column not in cleaned.columns:
					continue
				values = cleaned[column].astype("string")
				for category in categories:
					encoded_frames.append(
						pd.DataFrame(
							{f"{column}__{category}": values.eq(category).fillna(False).astype("int8")},
							index=cleaned.index,
						)
					)
				cleaned = cleaned.drop(columns=[column])
			if encoded_frames:
				cleaned = pd.concat([cleaned, *encoded_frames], axis=1)
			log.append(
				f"One-hot encoded {len(self.category_values_)} categorical feature(s) using "
				"training categories; unseen values map to all-zero indicators."
			)

		if not self.config.remove_duplicates:
			log.append(f"{before['duplicate_rows']:,} duplicate rows were present before cleaning.")
		after = build_quality_report(cleaned)
		log.append(
			f"Cleaning complete: {before['rows']:,} input rows, {after['rows']:,} output rows, "
			f"{before['missing_cells']:,} missing cells before, {after['missing_cells']:,} after."
		)
		return CleaningResult(
			data=cleaned,
			before_report=before,
			after_report=after,
			outlier_flags=outlier_flags,
			corrupted_categories=corrupted,
			log=log,
		)


def clean_dataset(
	frame: pd.DataFrame,
	config: CleaningConfig | None = None,
) -> CleaningResult:
	"""Convenience cleaner for an EDA-only frame.

	For model development, do not call this on a complete train/test dataset:
	fit a ``DataCleaningPipeline`` on training rows and transform held-out rows.
	"""
	return DataCleaningPipeline(config).fit_transform(frame)
