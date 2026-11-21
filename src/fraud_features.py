"""Leakage-safe feature engineering for the European credit card dataset.

The dataset's ``Time`` column is elapsed seconds from the first recorded
transaction, not a wall-clock timestamp. Its derived hour is therefore an
elapsed-hour-of-day bucket, not the cardholder's local hour.

Features retained from the input:
* Numeric transaction fields (``Time``, ``Amount``, and ``V1``-``V28``) keep
  their supplied signal; invalid values are coerced and imputed from training
  medians so prediction inputs have the same schema.

Derived features:
* ``amount_log1p`` compresses the usually skewed transaction amounts while
  preserving unusually large values.
* ``amount_deviation_from_training_mean`` and ``amount_z_score`` quantify how
  far an amount is from typical training transactions, in currency units and
  standard deviations respectively.
* ``amount_robust_z_score`` measures amount deviation using the training
  median absolute deviation (or IQR fallback) and is less sensitive to extremes.
* ``is_high_transaction_amount`` flags amounts above the training-set
  quantile configured on the engineer.
* ``transaction_elapsed_seconds``, ``transaction_elapsed_hours``, and
  ``transaction_hour`` capture transaction position in the observed period
  and its repeating 24-hour elapsed-time pattern.

No customer identifier is present, so customer velocity and customer-based
rolling features cannot be computed without inventing relationships. All
amount statistics and imputation values are learned only in ``fit``; callers
must fit on training rows and reuse that instance for held-out or prediction
rows. The target column ``Class`` is never returned as an input feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FraudFeatureEngineer:
	"""Fit reusable fraud features on training rows and transform new rows."""

	def __init__(self, high_amount_quantile: float = 0.99) -> None:
		if not 0 < high_amount_quantile <= 1:
			raise ValueError("high_amount_quantile must be in the interval (0, 1]")
		self.high_amount_quantile = high_amount_quantile
		self.is_fitted = False
		self.numeric_columns_: list[str] = []
		self.numeric_medians_: dict[str, float] = {}
		self.amount_column_: str | None = None
		self.time_column_: str | None = None
		self.amount_mean_: float = 0.0
		self.amount_std_: float = 0.0
		self.amount_median_: float = 0.0
		self.amount_mad_: float = 0.0
		self.amount_iqr_: float = 0.0
		self.high_amount_threshold_: float = 0.0

	@staticmethod
	def _validate_frame(frame: pd.DataFrame, *, fitting: bool) -> None:
		if not isinstance(frame, pd.DataFrame):
			raise TypeError("frame must be a pandas DataFrame")
		if frame.columns.has_duplicates:
			raise ValueError("DataFrame contains duplicate column names")
		if fitting and frame.empty:
			raise ValueError("Cannot fit fraud features on an empty DataFrame")

	def fit(self, training_frame: pd.DataFrame) -> FraudFeatureEngineer:
		"""Learn numeric imputations and amount baselines from training rows."""
		self._validate_frame(training_frame, fitting=True)
		self.numeric_columns_ = []
		self.numeric_medians_ = {}

		for column in training_frame.columns:
			if str(column).casefold() == "class":
				continue
			series = training_frame[column]
			converted = pd.to_numeric(series, errors="coerce").replace(
				[float("inf"), float("-inf")], np.nan
			)
			non_missing = int(series.notna().sum())
			is_numeric = pd.api.types.is_numeric_dtype(series.dtype)
			parse_ratio = float(converted.notna().sum() / non_missing) if non_missing else 0.0
			if is_numeric or parse_ratio >= 0.6:
				name = str(column)
				self.numeric_columns_.append(name)
				median = converted.median()
				self.numeric_medians_[name] = float(median) if pd.notna(median) else 0.0

		self.amount_column_ = self._find_column("amount")
		self.time_column_ = self._find_column("time")
		if self.amount_column_ is not None:
			amounts = self._numeric_training_values(
				training_frame[self.amount_column_], self.numeric_medians_[self.amount_column_]
			)
			self.amount_mean_ = float(amounts.mean())
			self.amount_std_ = float(amounts.std(ddof=0))
			self.amount_median_ = float(amounts.median())
			self.amount_mad_ = float((amounts - self.amount_median_).abs().median())
			q1, q3 = amounts.quantile([0.25, 0.75])
			self.amount_iqr_ = float(q3 - q1)
			self.high_amount_threshold_ = float(
				amounts.quantile(self.high_amount_quantile)
			)

		self.is_fitted = True
		return self

	def _find_column(self, expected: str) -> str | None:
		return next(
			(column for column in self.numeric_columns_ if column.casefold() == expected),
			None,
		)

	@staticmethod
	def _numeric_training_values(series: pd.Series, median: float) -> pd.Series:
		return pd.to_numeric(series, errors="coerce").replace(
			[float("inf"), float("-inf")], np.nan
		).fillna(median)

	def fit_transform(self, training_frame: pd.DataFrame) -> pd.DataFrame:
		"""Fit on training rows and return their engineered features."""
		return self.fit(training_frame).transform(training_frame)

	def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
		"""Return a feature-only copy using fitted training statistics."""
		if not self.is_fitted:
			raise RuntimeError("Call fit(training_frame) before transform(frame)")
		self._validate_frame(frame, fitting=False)

		features = pd.DataFrame(index=frame.index)
		for column in self.numeric_columns_:
			if column in frame.columns:
				values = pd.to_numeric(frame[column], errors="coerce")
			else:
				values = pd.Series(np.nan, index=frame.index)
			values = values.replace([float("inf"), float("-inf")], np.nan)
			features[column] = values.fillna(self.numeric_medians_[column])

		if self.amount_column_ is not None:
			amount = features[self.amount_column_].astype(float)
			features["amount_log1p"] = np.sign(amount) * np.log1p(np.abs(amount))
			features["amount_deviation_from_training_mean"] = (
				amount - self.amount_mean_
			)
			features["amount_z_score"] = (
				(amount - self.amount_mean_) / self.amount_std_
				if self.amount_std_ > 0
				else 0.0
			)
			features["amount_robust_z_score"] = (
				0.6745 * (amount - self.amount_median_) / self.amount_mad_
				if self.amount_mad_ > 0
				else 1.349 * (amount - self.amount_median_) / self.amount_iqr_
				if self.amount_iqr_ > 0
				else 0.0
			)
			features["is_high_transaction_amount"] = (
				amount.gt(self.high_amount_threshold_).astype("int8")
			)

		if self.time_column_ is not None:
			elapsed = features[self.time_column_].astype(float)
			features["transaction_elapsed_seconds"] = elapsed
			features["transaction_elapsed_hours"] = elapsed / 3600.0
			features["transaction_hour"] = (
				np.floor(elapsed / 3600.0).astype("int64") % 24
			)

		return features


def engineer_fraud_features(
	training_frame: pd.DataFrame,
	frame: pd.DataFrame,
	high_amount_quantile: float = 0.99,
) -> pd.DataFrame:
	"""Fit on ``training_frame`` and engineer features for ``frame``.

	Use the same training frame as ``frame`` only for exploratory analysis.
	For model workflows, pass training rows separately and reuse a fitted
	``FraudFeatureEngineer`` to transform validation, test, or prediction data.
	"""
	return FraudFeatureEngineer(high_amount_quantile).fit(training_frame).transform(frame)