"""Loan dataset cleaning diagnostics and exploratory-analysis summaries.

The inspected ``loan_data.csv`` has 12 columns and uses ``loan_status`` as its
binary target: the dataset publisher documents 0 as non-default and 1 as
default. No applicant identifier is present, so duplicate checks are exact
row checks rather than applicant-level identity resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.cleaning import (
	CleaningConfig,
	CleaningResult,
	DataCleaningPipeline,
	build_quality_report,
	detect_corrupted_categories,
	normalize_category_value,
)


LOAN_TARGET_COLUMN = "loan_status"
LOAN_TARGET_MEANINGS = {0: "Non-default", 1: "Default"}
LOAN_SOURCE_URL = "https://www.kaggle.com/datasets/laotse/credit-risk-dataset"
LOAN_ALLOWED_CATEGORIES = {
	"person_home_ownership": {"rent", "mortgage", "own", "other"},
	"loan_intent": {
		"education",
		"medical",
		"venture",
		"personal",
		"debtconsolidation",
		"homeimprovement",
	},
	"loan_grade": {"a", "b", "c", "d", "e", "f", "g"},
	"cb_person_default_on_file": {"n", "y"},
}
LOAN_NUMERIC_COLUMNS = (
	"person_age",
	"person_income",
	"person_emp_length",
	"loan_amnt",
	"loan_int_rate",
	"loan_percent_income",
	"cb_person_cred_hist_length",
)


@dataclass
class LoanDataQualityResult:
	"""Cleaned working copy plus row-level credit consistency review flags."""

	cleaning: CleaningResult
	consistency_flags: pd.DataFrame
	target_column: str = LOAN_TARGET_COLUMN


class LoanFeatureEngineer:
	"""Fit reusable loan features on training rows and transform new applications.

	``loan_status`` is excluded from all inputs. Numeric medians, categorical
	modes/vocabularies, derived-ratio imputations, income quartile cutoffs, and
	the high-loan-to-income threshold are learned only in ``fit``. Age groups
	have fixed boundaries; unseen prediction categories map to an explicit
	unknown indicator, keeping the fitted output schema unchanged.

	Feature formulas:
	* ``loan_to_income_ratio`` = loan amount / annual income; both source values
	  must be positive. ``income_to_loan_ratio`` is its inverse.
	  Invalid denominators use a training-derived median and have a flag.
	* ``loan_percent_income_gap`` is the absolute gap between the recomputed
	  ratio and the supplied ``loan_percent_income``; a separate flag marks
	  gaps greater than 0.02 (two percentage points). ``high_loan_to_income``
	  marks ratios above the training-set 75th percentile.
	* ``interest_burden_proxy`` = loan amount * annual interest rate / (100 *
	  annual income). It is a simple annual-interest-to-income proxy, not an
	  amortized repayment estimate; no loan term is available in this dataset.
	* ``employment_stability_ratio`` = employment years / (age - 18), only
	  when age is over 18 and employment length is nonnegative. Values above 1
	  are retained as evidence of potentially inconsistent source data.
	* ``age_group_*`` are fixed one-hot bands: 18-24, 25-34, 35-44, 45-54,
	  55+, and unknown.
	* ``income_band_*`` are one-hot training-income quartile bands. Cutoffs
	  come from the positive income values in training data only.
	* Prior-default, home-ownership, intent, and grade categories are
	  one-hot encoded from the inspected schema; missing and unseen categories
	  get separate indicators.
	* Numeric source fields are retained after training-median imputation;
	  ``<column>__missing`` indicators mark values that were missing, invalid,
	  non-finite, or absent in a prediction frame.

	The dataset has no total-debt field, so this transformer does not create a
	debt-to-income ratio. ``loan_percent_income`` is a single-loan/income
	measure, not total debt divided by income.
	"""

	def __init__(self) -> None:
		self.is_fitted = False
		self.numeric_columns_: list[str] = []
		self.categorical_columns_: list[str] = []
		self.numeric_medians_: dict[str, float] = {}
		self.categorical_modes_: dict[str, str] = {}
		self.categorical_values_: dict[str, list[str]] = {}
		self.derived_medians_: dict[str, float] = {}
		self.income_quantile_edges_: tuple[float, float, float] | None = None
		self.high_loan_to_income_threshold_: float = 0.0
		self.feature_names_out_: list[str] = []

	@staticmethod
	def _validate_frame(frame: pd.DataFrame, *, fitting: bool) -> None:
		if not isinstance(frame, pd.DataFrame):
			raise TypeError("frame must be a pandas DataFrame")
		if frame.columns.has_duplicates:
			raise ValueError("DataFrame contains duplicate column names")
		if fitting and frame.empty:
			raise ValueError("Cannot fit loan features on an empty DataFrame")

	@staticmethod
	def _as_numeric(series: pd.Series) -> pd.Series:
		return pd.to_numeric(series, errors="coerce").replace(
			[float("inf"), float("-inf")], np.nan
		)

	@staticmethod
	def _normalize_category(value: Any) -> Any:
		if pd.isna(value):
			return pd.NA
		return normalize_category_value(value, "lower")

	def _numeric_values(
		self,
		frame: pd.DataFrame,
	) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
		values: dict[str, pd.Series] = {}
		missing: dict[str, pd.Series] = {}
		for column in self.numeric_columns_:
			source = frame[column] if column in frame.columns else pd.Series(
				np.nan, index=frame.index, dtype=float
			)
			converted = self._as_numeric(source)
			missing[column] = converted.isna()
			values[column] = converted.fillna(self.numeric_medians_[column]).astype(float)
		return values, missing

	def _derive_values(
		self,
		values: dict[str, pd.Series],
	) -> dict[str, pd.Series]:
		derived: dict[str, pd.Series] = {}

		if {"person_income", "loan_amnt"}.issubset(values):
			income = values["person_income"]
			amount = values["loan_amnt"]
			valid_denominators = income.gt(0) & amount.gt(0)
			loan_to_income = amount.div(income.where(valid_denominators))
			income_to_loan = income.div(amount.where(valid_denominators))
			derived["loan_to_income_ratio"] = loan_to_income.replace(
				[float("inf"), float("-inf")], np.nan
			)
			derived["income_to_loan_ratio"] = income_to_loan.replace(
				[float("inf"), float("-inf")], np.nan
			)
			derived["loan_to_income_invalid_denominator"] = (
				~valid_denominators
			).astype("int8")
			derived["income_to_loan_invalid_denominator"] = (
				~valid_denominators
			).astype("int8")

			if "loan_percent_income" in values:
				reported_ratio = values["loan_percent_income"]
				derived["loan_percent_income_gap"] = (
					loan_to_income - reported_ratio
				).abs().replace([float("inf"), float("-inf")], np.nan)
				derived["loan_percent_income_differs_by_over_2pp"] = (
					derived["loan_percent_income_gap"].gt(0.02).fillna(False).astype("int8")
				)

			if "loan_int_rate" in values:
				rate = values["loan_int_rate"]
				valid_interest_inputs = valid_denominators & rate.ge(0)
				interest_burden = amount.mul(rate).div(
					income.where(valid_interest_inputs).mul(100)
				)
				derived["interest_burden_proxy"] = interest_burden.replace(
					[float("inf"), float("-inf")], np.nan
				)
				derived["interest_burden_invalid_inputs"] = (
					~valid_interest_inputs
				).astype("int8")

		if {"person_age", "person_emp_length"}.issubset(values):
			age = values["person_age"]
			employment = values["person_emp_length"]
			valid_employment_inputs = age.gt(18) & employment.ge(0)
			stability = employment.div((age - 18).where(valid_employment_inputs))
			derived["employment_stability_ratio"] = stability.replace(
				[float("inf"), float("-inf")], np.nan
			)
			derived["employment_stability_invalid_inputs"] = (
				~valid_employment_inputs
			).astype("int8")

		return derived

	def fit(self, training_frame: pd.DataFrame) -> LoanFeatureEngineer:
		"""Learn imputations and income-based thresholds from training rows."""
		self._validate_frame(training_frame, fitting=True)
		self.numeric_columns_ = []
		self.categorical_columns_ = []
		self.numeric_medians_ = {}
		self.categorical_modes_ = {}
		self.categorical_values_ = {}
		self.derived_medians_ = {}
		self.income_quantile_edges_ = None
		self.high_loan_to_income_threshold_ = 0.0

		for column in LOAN_NUMERIC_COLUMNS:
			if column not in training_frame.columns:
				continue
			series = training_frame[column]
			converted = self._as_numeric(series)
			non_missing = int(series.notna().sum())
			parse_ratio = float(converted.notna().sum() / non_missing) if non_missing else 0.0
			if pd.api.types.is_numeric_dtype(series.dtype) or parse_ratio >= 0.6:
				self.numeric_columns_.append(column)
				median = converted.median()
				self.numeric_medians_[column] = (
					float(median) if pd.notna(median) else 0.0
				)

		for column in LOAN_ALLOWED_CATEGORIES:
			if column not in training_frame.columns:
				continue
			self.categorical_columns_.append(column)
			normalized = training_frame[column].map(self._normalize_category)
			allowed = sorted(LOAN_ALLOWED_CATEGORIES[column])
			known_values = normalized.loc[normalized.isin(allowed)]
			self.categorical_values_[column] = sorted(known_values.unique().tolist())
			mode = known_values.mode(dropna=True)
			self.categorical_modes_[column] = (
				str(mode.iloc[0]) if not mode.empty else "unknown"
			)

		if not self.numeric_columns_ and not self.categorical_columns_:
			raise ValueError("No supported loan predictor columns were found")

		numeric_values, _ = self._numeric_values(training_frame)
		if "person_income" in numeric_values:
			training_income = self._as_numeric(training_frame["person_income"])
			positive_income = training_income.loc[training_income.gt(0)]
			if not positive_income.empty:
				quantiles = positive_income.quantile([0.25, 0.5, 0.75]).to_numpy()
				self.income_quantile_edges_ = tuple(
					float(value) for value in quantiles
				)

		training_derived = self._derive_values(numeric_values)
		for name, series in training_derived.items():
			finite = series.replace([float("inf"), float("-inf")], np.nan).dropna()
			self.derived_medians_[name] = (
				float(finite.median()) if not finite.empty else 0.0
			)
		if "loan_to_income_ratio" in training_derived:
			valid_ratios = training_derived["loan_to_income_ratio"].dropna()
			if not valid_ratios.empty:
				self.high_loan_to_income_threshold_ = float(valid_ratios.quantile(0.75))

		self.is_fitted = True
		self.feature_names_out_ = self.transform(training_frame).columns.tolist()
		return self

	def fit_transform(self, training_frame: pd.DataFrame) -> pd.DataFrame:
		"""Fit on training rows and return the resulting model-ready features."""
		return self.fit(training_frame).transform(training_frame)

	def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
		"""Apply fitted transformations without reading or returning loan_status."""
		if not self.is_fitted:
			raise RuntimeError("Call fit(training_frame) before transform(frame)")
		self._validate_frame(frame, fitting=False)
		features = pd.DataFrame(index=frame.index)
		numeric_values, numeric_missing = self._numeric_values(frame)

		for column in self.numeric_columns_:
			features[column] = numeric_values[column]
			features[f"{column}__missing"] = numeric_missing[column].astype("int8")

		for column in self.categorical_columns_:
			if column in frame.columns:
				source = frame[column]
				normalized = source.map(self._normalize_category).astype("string")
				missing = normalized.isna()
			else:
				normalized = pd.Series(pd.NA, index=frame.index, dtype="string")
				missing = pd.Series(True, index=frame.index)
			imputed = normalized.fillna(self.categorical_modes_[column])
			allowed = self.categorical_values_[column]
			for value in allowed:
				features[f"{column}__{value}"] = imputed.eq(value).fillna(False).astype("int8")
			features[f"{column}__unknown"] = (
				~imputed.isin(allowed)
			).astype("int8")
			features[f"{column}__missing"] = missing.astype("int8")

		if "person_age" in numeric_values:
			age = numeric_values["person_age"]
			age_missing = numeric_missing["person_age"] | age.lt(18)
			age_groups = {
				"18_24": age.ge(18) & age.lt(25),
				"25_34": age.ge(25) & age.lt(35),
				"35_44": age.ge(35) & age.lt(45),
				"45_54": age.ge(45) & age.lt(55),
				"55_plus": age.ge(55),
			}
			for group_name, mask in age_groups.items():
				features[f"age_group_{group_name}"] = (
					mask & ~age_missing
				).astype("int8")
			features["age_group_unknown"] = age_missing.astype("int8")

		if "person_income" in numeric_values:
			income = numeric_values["person_income"]
			income_unknown = numeric_missing["person_income"] | income.le(0)
			band_names = (
				"bottom_quartile",
				"second_quartile",
				"third_quartile",
				"top_quartile",
			)
			band_indexes = np.zeros(len(frame), dtype="int8")
			if self.income_quantile_edges_ is not None:
				band_indexes = np.searchsorted(
					np.asarray(self.income_quantile_edges_),
					income.to_numpy(dtype=float),
					side="right",
				).astype("int8")
			for band_index, band_name in enumerate(band_names):
				features[f"income_band_{band_name}"] = (
					(band_indexes == band_index) & ~income_unknown.to_numpy()
				).astype("int8")
			features["income_band_unknown"] = income_unknown.astype("int8")

		derived = self._derive_values(numeric_values)
		for name, series in derived.items():
			if name.endswith("invalid_denominator") or name.endswith("invalid_inputs"):
				features[name] = series.astype("int8")
			elif name == "loan_percent_income_differs_by_over_2pp":
				features[name] = series.astype("int8")
			else:
				values = series.replace([float("inf"), float("-inf")], np.nan)
				features[name] = values.fillna(self.derived_medians_[name]).astype(float)

		if "loan_to_income_ratio" in derived:
			features["high_loan_to_income"] = (
				derived["loan_to_income_ratio"]
				.gt(self.high_loan_to_income_threshold_)
				.fillna(False)
				.astype("int8")
			)

		if LOAN_TARGET_COLUMN in features.columns:
			raise ValueError("Target leakage detected: loan_status is present in features")
		if not np.isfinite(features.to_numpy(dtype=float)).all():
			raise ValueError("Loan feature transformation produced non-finite values")
		return features


def _validate_target(frame: pd.DataFrame) -> pd.Series:
	if not isinstance(frame, pd.DataFrame):
		raise TypeError("frame must be a pandas DataFrame")
	if frame.columns.has_duplicates:
		raise ValueError("DataFrame contains duplicate column names")
	if frame.empty:
		raise ValueError("Cannot analyze an empty loan dataset")
	if LOAN_TARGET_COLUMN not in frame.columns:
		raise ValueError(f"Loan dataset must contain '{LOAN_TARGET_COLUMN}'")

	raw_target = frame[LOAN_TARGET_COLUMN]
	target = pd.to_numeric(raw_target, errors="coerce")
	if (raw_target.notna() & target.isna()).any():
		raise ValueError(f"'{LOAN_TARGET_COLUMN}' contains non-numeric target values")
	unsupported = target.notna() & ~target.isin(LOAN_TARGET_MEANINGS)
	if unsupported.any():
		values = sorted(target.loc[unsupported].unique().tolist())
		raise ValueError(
			f"'{LOAN_TARGET_COLUMN}' contains values outside the observed 0/1 encoding: {values}"
		)
	return target


def _build_consistency_flags(frame: pd.DataFrame) -> pd.DataFrame:
	flags = pd.DataFrame(index=frame.index)
	for column in LOAN_NUMERIC_COLUMNS:
		if column not in frame.columns:
			continue
		source_values = frame[column]
		numeric_values = pd.to_numeric(source_values, errors="coerce")
		flags[f"nonnumeric_value_{column}"] = (
			source_values.notna() & numeric_values.isna()
		)
		flags[f"nonfinite_value_{column}"] = (
			numeric_values.notna() & ~np.isfinite(numeric_values)
		).fillna(False)
	age = pd.to_numeric(frame.get("person_age"), errors="coerce")
	income = pd.to_numeric(frame.get("person_income"), errors="coerce")
	employment_length = pd.to_numeric(frame.get("person_emp_length"), errors="coerce")
	loan_amount = pd.to_numeric(frame.get("loan_amnt"), errors="coerce")
	loan_to_income = pd.to_numeric(frame.get("loan_percent_income"), errors="coerce")
	credit_history_length = pd.to_numeric(
		frame.get("cb_person_cred_hist_length"), errors="coerce"
	)
	interest_rate = pd.to_numeric(frame.get("loan_int_rate"), errors="coerce")

	if age is not None:
		flags["age_over_100_review"] = age.gt(100).fillna(False)
		flags["age_below_18_invalid"] = age.lt(18).fillna(False)
	if employment_length is not None and age is not None:
		flags["negative_employment_length"] = employment_length.lt(0).fillna(False)
		flags["employment_length_exceeds_age"] = (
			employment_length.gt(age).fillna(False)
		)
	if credit_history_length is not None and age is not None:
		flags["negative_credit_history_length"] = (
			credit_history_length.lt(0).fillna(False)
		)
		flags["credit_history_exceeds_age"] = (
			credit_history_length.gt(age).fillna(False)
		)
	if income is not None:
		flags["nonpositive_income"] = income.le(0).fillna(False)
	if loan_amount is not None:
		flags["nonpositive_loan_amount"] = loan_amount.le(0).fillna(False)
	if loan_to_income is not None:
		flags["loan_percent_income_out_of_range"] = (
			loan_to_income.lt(0) | loan_to_income.gt(1)
		).fillna(False)
	if loan_amount is not None and income is not None and loan_to_income is not None:
		expected_ratio = loan_amount.div(income.where(income.gt(0)))
		flags["loan_percent_income_differs_by_over_2pp"] = (
			expected_ratio.sub(loan_to_income).abs().gt(0.02).fillna(False)
		)
	if interest_rate is not None:
		flags["nonpositive_interest_rate"] = interest_rate.le(0).fillna(False)
	if "cb_person_default_on_file" in frame.columns:
		history = frame["cb_person_default_on_file"].map(
			lambda value: (
				normalize_category_value(value, "lower")
				if pd.notna(value)
				else value
			)
		)
		flags["invalid_prior_default_category"] = (
			history.notna() & ~history.isin(LOAN_ALLOWED_CATEGORIES["cb_person_default_on_file"])
		)
	return flags.astype(bool)


def clean_loan_data(
	frame: pd.DataFrame,
	*,
	remove_duplicates: bool = True,
	cap_outliers: bool = False,
	encode_categorical: bool = False,
) -> LoanDataQualityResult:
	"""Clean a copy of the inspected loan schema and flag domain inconsistencies.

	This helper is for data-quality review and EDA. Its default behavior
	removes exact duplicate rows, imputes numeric/categorical missing values,
	normalizes category spacing/case, and flags rather than caps outliers.
	Target labels are never imputed or category-normalized. Age-over-100 and
	loan/income discrepancies are review flags, not automatic corrections.
	"""
	_validate_target(frame)
	before_report = build_quality_report(frame)
	working_frame = frame.copy(deep=True)
	removed_duplicates = 0
	if remove_duplicates:
		unique_mask = ~working_frame.duplicated(keep="first")
		removed_duplicates = int((~unique_mask).sum())
		working_frame = working_frame.loc[unique_mask].copy()
	config = CleaningConfig(
		remove_duplicates=False,
		categorical_imputation="most_frequent",
		normalize_categories=True,
		categorical_case="lower",
		outlier_action="cap" if cap_outliers else "flag",
		encode_categorical=encode_categorical,
		target_columns=(LOAN_TARGET_COLUMN,),
	)
	cleaning = DataCleaningPipeline(config).fit_transform(working_frame)
	cleaning.before_report = before_report
	cleaning.after_report = build_quality_report(cleaning.data)
	cleaning.log = [
		entry
		for entry in cleaning.log
		if not entry.startswith("Duplicate rows were detected but retained")
		and not entry.endswith("duplicate rows were present before cleaning.")
		and not entry.startswith("Cleaning complete:")
	]
	if remove_duplicates:
		cleaning.log.append(
			f"Removed {removed_duplicates:,} exact duplicate source row(s) before imputation."
		)
	else:
		cleaning.log.append(
			f"Retained {before_report['duplicate_rows']:,} exact duplicate source row(s) by configuration."
		)
	cleaning.log.append(
		f"Cleaning complete: {before_report['rows']:,} input rows, "
		f"{cleaning.after_report['rows']:,} output rows, "
		f"{before_report['missing_cells']:,} missing cells before, "
		f"{cleaning.after_report['missing_cells']:,} after."
	)

	normalized_categories = frame.copy(deep=True)
	for column in LOAN_ALLOWED_CATEGORIES:
		if column in frame.columns:
			normalized_categories[column] = frame[column].map(
				lambda value: (
					normalize_category_value(value, "lower")
					if pd.notna(value)
					else value
				)
			)
	cleaning.corrupted_categories = detect_corrupted_categories(
		normalized_categories,
		allowed_categories=LOAN_ALLOWED_CATEGORIES,
	)
	return LoanDataQualityResult(
		cleaning=cleaning,
		consistency_flags=_build_consistency_flags(frame),
	)


def summarize_loan_data(frame: pd.DataFrame) -> dict[str, Any]:
	"""Compute actual target, employment, credit-history, and correlation summaries."""
	target = _validate_target(frame)
	valid_target = target.notna()
	data = frame.loc[valid_target].copy()
	target = target.loc[valid_target].astype("int8")

	counts = target.value_counts().reindex([0, 1], fill_value=0)
	target_distribution = pd.DataFrame(
		{
			"loan_status": [0, 1],
			"meaning": [LOAN_TARGET_MEANINGS[0], LOAN_TARGET_MEANINGS[1]],
			"count": counts.to_numpy(),
			"percentage": (
				counts.to_numpy() / len(target) * 100
				if len(target)
				else np.zeros(len(counts), dtype=float)
			),
		}
	)
	numeric = data.select_dtypes(include="number").copy()
	numeric[LOAN_TARGET_COLUMN] = target
	numeric_correlations = numeric.corr(numeric_only=True)[LOAN_TARGET_COLUMN].drop(
		labels=[LOAN_TARGET_COLUMN], errors="ignore"
	)

	if {"person_income", "loan_amnt"}.issubset(data.columns):
		income_amount = data[["person_income", "loan_amnt"]].apply(
			pd.to_numeric, errors="coerce"
		).dropna()
		if len(income_amount) > 1:
			pearson = float(income_amount.corr(method="pearson").iloc[0, 1])
			spearman = float(income_amount.corr(method="spearman").iloc[0, 1])
		else:
			pearson = float("nan")
			spearman = float("nan")
	else:
		pearson = float("nan")
		spearman = float("nan")

	employment_default_rates = pd.DataFrame()
	if "person_emp_length" in data.columns:
		years = pd.to_numeric(data["person_emp_length"], errors="coerce")
		buckets = pd.cut(
			years,
			bins=[-0.01, 1, 3, 5, 10, 20, np.inf],
			labels=["0-1", "1-3", "3-5", "5-10", "10-20", "20+"],
			include_lowest=True,
		).astype("string").fillna("Missing")
		employment_default_rates = (
			pd.DataFrame({"employment_years": buckets, "loan_status": target})
			.groupby("employment_years", observed=True)["loan_status"]
			.agg(rows="size", default_rate="mean")
			.reset_index()
		)

	default_rates_by_category: dict[str, pd.DataFrame] = {}
	for column in (
		"person_home_ownership",
		"loan_intent",
		"loan_grade",
		"cb_person_default_on_file",
	):
		if column in data.columns:
			default_rates_by_category[column] = (
				pd.DataFrame(
					{
						column: data[column].astype("string").fillna("(missing)"),
						"loan_status": target,
					}
				)
				.groupby(column, observed=True)["loan_status"]
				.agg(rows="size", default_rate="mean")
				.reset_index()
			)

	return {
		"target_column": LOAN_TARGET_COLUMN,
		"target_meanings": LOAN_TARGET_MEANINGS.copy(),
		"row_count": int(len(data)),
		"missing_target_rows": int((~valid_target).sum()),
		"default_rate": float(target.mean()) if len(target) else float("nan"),
		"target_distribution": target_distribution,
		"numeric_summary": data.select_dtypes(include="number").describe().transpose(),
		"numeric_correlations_with_target": numeric_correlations.sort_values(
			key=lambda values: values.abs(), ascending=False
		),
		"income_loan_correlation": {
			"pearson": pearson,
			"spearman": spearman,
		},
		"median_financial_values_by_status": data.assign(
			**{LOAN_TARGET_COLUMN: target}
		)
		.groupby(LOAN_TARGET_COLUMN)[
			[
				column
				for column in ("person_income", "loan_amnt", "loan_percent_income")
				if column in data.columns
			]
		]
		.median(numeric_only=True),
		"employment_default_rates": employment_default_rates,
		"default_rates_by_category": default_rates_by_category,
	}