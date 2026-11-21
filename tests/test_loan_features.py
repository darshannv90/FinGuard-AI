import unittest

import numpy as np
import pandas as pd

from src.loan_features import (
	LOAN_TARGET_MEANINGS,
	LoanFeatureEngineer,
	clean_loan_data,
	summarize_loan_data,
)


class LoanDataWorkflowTests(unittest.TestCase):
	@staticmethod
	def _sample_data() -> pd.DataFrame:
		return pd.DataFrame(
			{
				"person_age": [25, 25, 30, 144, 30],
				"person_income": [50000, 50000, 70000, 6000000, 70000],
				"person_home_ownership": [" RENT ", "RENT", "OWN", "RENT", "OWN"],
				"person_emp_length": [np.nan, 4.0, 4.0, 150.0, 4.0],
				"loan_intent": ["MEDICAL", "MEDICAL", "EDUCATION", "MEDICAL", "EDUCATION"],
				"loan_grade": ["B", "B", "A", "G", "A"],
				"loan_amnt": [10000, 10000, 5000, 35000, 5000],
				"loan_int_rate": [10.0, 10.0, np.nan, 23.0, np.nan],
				"loan_status": [0, 0, 1, 0, 1],
				"loan_percent_income": [0.2, 0.2, 0.08, 0.59, 0.08],
				"cb_person_default_on_file": ["N", "N", "Y", "N", "Y"],
				"cb_person_cred_hist_length": [5, 5, 2, 3, 2],
			}
		)

	def test_cleaning_preserves_target_flags_quality_issues_and_input(self) -> None:
		frame = self._sample_data()
		original = frame.copy(deep=True)

		result = clean_loan_data(frame)
		cleaned = result.cleaning.data

		self.assertEqual(result.target_column, "loan_status")
		self.assertEqual(result.cleaning.before_report["rows"], 5)
		self.assertEqual(result.cleaning.before_report["duplicate_rows"], 1)
		self.assertEqual(result.cleaning.after_report["rows"], 4)
		self.assertEqual(result.cleaning.after_report["missing_cells"], 0)
		self.assertEqual(cleaned["loan_status"].tolist(), [0, 0, 1, 0])
		self.assertEqual(cleaned["person_home_ownership"].tolist(), ["rent"] * 2 + ["own", "rent"])
		self.assertEqual(int(result.consistency_flags["age_over_100_review"].sum()), 1)
		self.assertEqual(int(result.consistency_flags["employment_length_exceeds_age"].sum()), 1)
		self.assertTrue(
			bool(result.consistency_flags.loc[3, "loan_percent_income_differs_by_over_2pp"])
		)
		self.assertIn(
			"Removed 1 exact duplicate source row(s) before imputation.",
			result.cleaning.log,
		)
		pd.testing.assert_frame_equal(frame, original)

	def test_distinct_missing_rows_are_not_removed_after_imputation(self) -> None:
		frame = self._sample_data().iloc[:4].copy()
		result = clean_loan_data(frame)

		self.assertEqual(result.cleaning.data["person_emp_length"].iloc[0], 4.0)
		self.assertEqual(result.cleaning.after_report["rows"], 4)
		self.assertEqual(result.cleaning.after_report["duplicate_rows"], 1)

	def test_invalid_numeric_strings_are_imputed_and_categories_are_checked(self) -> None:
		frame = self._sample_data()
		frame["person_emp_length"] = frame["person_emp_length"].astype("object")
		frame.loc[0, "person_emp_length"] = "invalid"
		frame.loc[2, "person_home_ownership"] = "unlisted"

		result = clean_loan_data(frame, remove_duplicates=False)

		self.assertTrue(pd.api.types.is_numeric_dtype(result.cleaning.data["person_emp_length"]))
		self.assertTrue(np.isfinite(result.cleaning.data["person_emp_length"]).all())
		self.assertIn("not in allowed categories", result.cleaning.corrupted_categories.iloc[0]["reason"])
		self.assertTrue(
			bool(result.consistency_flags.loc[0, "nonnumeric_value_person_emp_length"])
		)
		self.assertTrue(any("invalid values" in entry for entry in result.cleaning.log))

	def test_eda_summary_reports_observed_target_meaning_and_relationships(self) -> None:
		frame = self._sample_data()
		original = frame.copy(deep=True)

		summary = summarize_loan_data(frame)

		self.assertEqual(summary["target_meanings"], LOAN_TARGET_MEANINGS)
		self.assertEqual(summary["target_distribution"]["count"].tolist(), [3, 2])
		self.assertAlmostEqual(summary["default_rate"], 0.4)
		self.assertIn("pearson", summary["income_loan_correlation"])
		self.assertIn("spearman", summary["income_loan_correlation"])
		self.assertFalse(summary["employment_default_rates"].empty)
		self.assertIn("cb_person_default_on_file", summary["default_rates_by_category"])
		pd.testing.assert_frame_equal(frame, original)

	def test_target_encoding_is_validated_instead_of_assumed(self) -> None:
		frame = self._sample_data()
		frame.loc[0, "loan_status"] = 2
		with self.assertRaisesRegex(ValueError, "outside the observed 0/1 encoding"):
			clean_loan_data(frame)
		with self.assertRaisesRegex(ValueError, "outside the observed 0/1 encoding"):
			summarize_loan_data(frame)

	def test_optional_one_hot_encoding_is_applied_to_categories_only(self) -> None:
		result = clean_loan_data(self._sample_data(), encode_categorical=True)

		self.assertNotIn("person_home_ownership", result.cleaning.data.columns)
		self.assertIn("person_home_ownership__rent", result.cleaning.data.columns)
		self.assertIn("loan_status", result.cleaning.data.columns)

	def test_feature_formulas_zero_denominators_and_target_exclusion(self) -> None:
		training = self._sample_data()
		engineer = LoanFeatureEngineer().fit(training)
		prediction = training.iloc[[0, 1]].copy()
		invalid_index = prediction.index[1]
		prediction.loc[invalid_index, "person_income"] = 0
		prediction.loc[invalid_index, "loan_amnt"] = 0
		prediction.loc[invalid_index, "loan_status"] = 1

		result = engineer.transform(prediction)

		valid_index = prediction.index[0]

		self.assertNotIn("loan_status", result.columns)
		self.assertAlmostEqual(result.loc[valid_index, "loan_to_income_ratio"], 10000 / 50000)
		self.assertAlmostEqual(result.loc[valid_index, "income_to_loan_ratio"], 50000 / 10000)
		self.assertAlmostEqual(result.loc[valid_index, "interest_burden_proxy"], 0.02)
		self.assertEqual(result.loc[invalid_index, "loan_to_income_invalid_denominator"], 1)
		self.assertEqual(result.loc[invalid_index, "income_to_loan_invalid_denominator"], 1)
		self.assertTrue(np.isfinite(result.to_numpy(dtype=float)).all())
		self.assertNotIn("debt_to_income_ratio", result.columns)

	def test_feature_schema_is_stable_and_unknown_values_are_indicated(self) -> None:
		training = self._sample_data()
		engineer = LoanFeatureEngineer().fit(training)
		prediction = training.iloc[[0]].copy()
		prediction.loc[prediction.index[0], "person_home_ownership"] = "new_category"
		prediction.loc[prediction.index[0], "person_emp_length"] = np.nan

		training_features = engineer.transform(training)
		prediction_features = engineer.transform(prediction)

		self.assertEqual(training_features.columns.tolist(), prediction_features.columns.tolist())
		self.assertEqual(prediction_features.iloc[0]["person_home_ownership__unknown"], 1)
		self.assertEqual(prediction_features.iloc[0]["person_emp_length__missing"], 1)
		self.assertEqual(prediction_features.iloc[0]["age_group_25_34"], 1)
		self.assertEqual(prediction_features.iloc[0]["cb_person_default_on_file__n"], 1)

	def test_income_bands_and_thresholds_are_fit_only_on_training_rows(self) -> None:
		training = self._sample_data()
		engineer = LoanFeatureEngineer().fit(training)
		training_edges = engineer.income_quantile_edges_
		prediction = training.iloc[[0]].copy()
		prediction.loc[prediction.index[0], "person_income"] = 1_000_000

		result = engineer.transform(prediction)

		self.assertEqual(engineer.income_quantile_edges_, training_edges)
		self.assertEqual(result.iloc[0]["income_band_top_quartile"], 1)
		self.assertGreater(
			result.iloc[0]["loan_to_income_ratio"],
			0,
		)

	def test_labels_do_not_change_fitted_features(self) -> None:
		training = self._sample_data()
		relabeled = training.copy(deep=True)
		relabeled["loan_status"] = 1 - relabeled["loan_status"]

		first = LoanFeatureEngineer().fit_transform(training)
		second = LoanFeatureEngineer().fit_transform(relabeled)

		pd.testing.assert_frame_equal(first, second)

	def test_interest_features_require_interest_rate_column(self) -> None:
		training = self._sample_data().drop(columns="loan_int_rate")

		result = LoanFeatureEngineer().fit_transform(training)

		self.assertNotIn("interest_burden_proxy", result.columns)
		self.assertNotIn("debt_to_income_ratio", result.columns)

	def test_feature_engineer_requires_valid_fit_input(self) -> None:
		engineer = LoanFeatureEngineer()
		with self.assertRaisesRegex(RuntimeError, "fit"):
			engineer.transform(self._sample_data())
		with self.assertRaisesRegex(ValueError, "empty"):
			engineer.fit(pd.DataFrame())


if __name__ == "__main__":
	unittest.main()
