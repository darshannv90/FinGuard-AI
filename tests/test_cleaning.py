import unittest

import pandas as pd

from src.cleaning import (
	CleaningConfig,
	DataCleaningPipeline,
	clean_dataset,
	detect_corrupted_categories,
)


class DataCleaningPipelineTests(unittest.TestCase):
	def test_imputes_normalizes_deduplicates_and_preserves_input(self) -> None:
		frame = pd.DataFrame(
			{
				"amount": [1.0, None, 1.0],
				"intent": ["  HOME  ", None, "  HOME  "],
			}
		)
		original = frame.copy(deep=True)

		result = clean_dataset(frame)

		self.assertEqual(result.before_report["missing_cells"], 2)
		self.assertEqual(result.before_report["duplicate_rows"], 1)
		self.assertEqual(result.after_report["rows"], 1)
		self.assertEqual(result.data.loc[result.data.index[0], "amount"], 1.0)
		self.assertEqual(result.data.loc[result.data.index[0], "intent"], "home")
		pd.testing.assert_frame_equal(frame, original)
		self.assertTrue(any("Removed 2" in entry for entry in result.log))
		self.assertTrue(any("Imputed 1 missing numeric values" in entry for entry in result.log))
		self.assertTrue(any("Imputed 1 missing categorical values" in entry for entry in result.log))
		self.assertTrue(any("values changed" in entry for entry in result.log))

	def test_invalid_numeric_values_are_coerced_and_imputed(self) -> None:
		frame = pd.DataFrame({"amount": ["10", "bad", "20"]})
		result = clean_dataset(frame)

		self.assertEqual(result.data["amount"].tolist(), [10.0, 15.0, 20.0])
		self.assertEqual(result.after_report["missing_cells"], 0)
		self.assertTrue(any("invalid values" in entry for entry in result.log))

	def test_unknown_fallback_handles_all_missing_categories(self) -> None:
		frame = pd.DataFrame({"kind": [None, None]})
		result = clean_dataset(
			frame,
			CleaningConfig(categorical_imputation="unknown", remove_duplicates=False),
		)

		self.assertEqual(result.data["kind"].tolist(), ["unknown", "unknown"])

	def test_missing_target_labels_are_not_imputed_or_normalized(self) -> None:
		frame = pd.DataFrame(
			{"amount": [1.0, None], "target": [" YES ", None]}
		)
		result = clean_dataset(frame, CleaningConfig(target_columns=("target",)))

		self.assertEqual(result.data["amount"].tolist(), [1.0, 1.0])
		self.assertEqual(result.data["target"].iloc[0], " YES ")
		self.assertTrue(pd.isna(result.data["target"].iloc[1]))
		self.assertTrue(any("Preserved missing target labels" in entry for entry in result.log))

	def test_outlier_flagging_does_not_cap_by_default(self) -> None:
		frame = pd.DataFrame({"amount": [0, 1, 1, 2, 100], "label": [0, 0, 1, 1, 0]})
		result = clean_dataset(frame)

		self.assertTrue(bool(result.outlier_flags.loc[4, "amount"]))
		self.assertEqual(result.data.loc[4, "amount"], 100)
		self.assertNotIn("label", result.outlier_flags.columns)

	def test_outlier_capping_is_opt_in(self) -> None:
		frame = pd.DataFrame({"amount": [0, 1, 1, 2, 100]})
		result = clean_dataset(frame, CleaningConfig(outlier_action="cap"))

		self.assertTrue(bool(result.outlier_flags.loc[4, "amount"]))
		self.assertLess(result.data.loc[4, "amount"], 100)
		self.assertTrue(any("Capped flagged values" in entry for entry in result.log))

	def test_train_only_imputation_scaling_and_encoding(self) -> None:
		training = pd.DataFrame(
			{"income": [0.0, 10.0], "purpose": ["Home", "Auto"], "target": [0, 1]}
		)
		validation = pd.DataFrame(
			{"income": [100.0], "purpose": ["New category"], "target": [0]}
		)
		pipeline = DataCleaningPipeline(
			CleaningConfig(
				remove_duplicates=False,
				scale_numeric=True,
				encode_categorical=True,
				target_columns=("target",),
			)
		).fit(training)

		result = pipeline.transform(validation)

		self.assertEqual(result.data.loc[0, "income"], 19.0)
		self.assertEqual(result.data.loc[0, "purpose__auto"], 0)
		self.assertEqual(result.data.loc[0, "purpose__home"], 0)
		self.assertEqual(result.data.loc[0, "target"], 0)
		self.assertTrue(any("Fitted preprocessing values on 2 rows" in entry for entry in result.log))

	def test_missing_and_unexpected_columns_are_handled(self) -> None:
		training = pd.DataFrame(
			{"income": [1.0, 3.0, 5.0], "purpose": ["home", "home", "auto"]}
		)
		validation = pd.DataFrame({"unexpected": ["kept"]})
		result = DataCleaningPipeline().fit(training).transform(validation)

		self.assertEqual(result.data.loc[0, "income"], 3.0)
		self.assertEqual(result.data.loc[0, "purpose"], "home")
		self.assertEqual(result.data.loc[0, "unexpected"], "kept")
		self.assertTrue(any("Added missing training columns" in entry for entry in result.log))
		self.assertTrue(any("Preserved unexpected columns" in entry for entry in result.log))

	def test_corrupted_categories_support_allow_lists(self) -> None:
		frame = pd.DataFrame({"status": ["approved", "bad\u0001value", "unknown"]})
		findings = detect_corrupted_categories(
			frame,
			allowed_categories={"status": ["approved", "rejected"]},
		)

		self.assertEqual(len(findings), 2)
		self.assertIn("control character", findings.iloc[0]["reason"])
		self.assertIn("not in allowed categories", findings.iloc[1]["reason"])

	def test_categorical_dtype_accepts_unseen_transform_values(self) -> None:
		training = pd.DataFrame(
			{"purpose": pd.Series(["home", "auto"], dtype="category")}
		)
		validation = pd.DataFrame(
			{"purpose": pd.Series(["business"], dtype="category")}
		)
		pipeline = DataCleaningPipeline(
			CleaningConfig(remove_duplicates=False, encode_categorical=True)
		).fit(training)

		result = pipeline.transform(validation)

		self.assertEqual(result.data.loc[0, "purpose__auto"], 0)
		self.assertEqual(result.data.loc[0, "purpose__home"], 0)

	def test_transform_requires_fit(self) -> None:
		with self.assertRaises(RuntimeError):
			DataCleaningPipeline().transform(pd.DataFrame({"amount": [1]}))

	def test_duplicate_column_names_are_rejected(self) -> None:
		frame = pd.DataFrame([[1, 2]], columns=["amount", "amount"])
		with self.assertRaisesRegex(ValueError, "duplicate column"):
			DataCleaningPipeline().fit(frame)


if __name__ == "__main__":
	unittest.main()