import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import DatasetLoadError
from src.train_loan import train_loan_models


class LoanTrainingTests(unittest.TestCase):
	@staticmethod
	def _sample_data() -> pd.DataFrame:
		rng = np.random.default_rng(19)
		labels = np.array([0] * 150 + [1] * 50)
		rows = len(labels)
		frame = pd.DataFrame(
			{
				"person_age": rng.integers(20, 60, rows),
				"person_income": rng.normal(60000, 18000, rows) - labels * 12000,
				"person_home_ownership": np.where(labels == 1, "RENT", "MORTGAGE"),
				"person_emp_length": rng.integers(0, 20, rows).astype(float),
				"loan_intent": np.where(labels == 1, "MEDICAL", "EDUCATION"),
				"loan_grade": np.where(labels == 1, "D", "B"),
				"loan_amnt": rng.integers(1000, 25000, rows),
				"loan_int_rate": rng.normal(11, 3, rows) + labels * 2,
				"loan_status": labels,
				"loan_percent_income": rng.uniform(0.03, 0.6, rows),
				"cb_person_default_on_file": np.where(labels == 1, "Y", "N"),
				"cb_person_cred_hist_length": rng.integers(2, 20, rows),
			}
		)
		frame.loc[2, "person_emp_length"] = np.nan
		frame.loc[8, "loan_int_rate"] = np.nan
		return frame

	def test_models_are_compared_and_selected_pipeline_is_saved(self) -> None:
		frame = self._sample_data()
		original = frame.copy(deep=True)

		with tempfile.TemporaryDirectory() as output:
			result = train_loan_models(frame, models_dir=output)

			self.assertEqual(
				result.results["Model"].tolist(),
				["Logistic Regression", "Decision Tree", "Random Forest"],
			)
			self.assertEqual(int(result.results["Selected"].sum()), 1)
			selected = result.results.loc[result.results["Selected"]].iloc[0]
			self.assertEqual(selected["Model"], result.best_model_name)
			self.assertAlmostEqual(float(selected["PR-AUC"]), result.best_pr_auc)
			self.assertTrue(
				{
					"Precision",
					"Recall",
					"F1-score",
					"ROC-AUC",
					"PR-AUC",
					"Confusion Matrix",
					"TN",
					"FP",
					"FN",
					"TP",
				}.issubset(result.results.columns)
			)
			self.assertTrue(
				np.isfinite(
					result.results[
						["Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"]
					].to_numpy()
				).all()
			)
			for artifact in result.artifact_paths.values():
				self.assertTrue(Path(artifact).is_file())
			summary = json.loads(result.artifact_paths["summary"].read_text(encoding="utf-8"))
			self.assertEqual(summary["target_meanings"], {"0": "Non-default", "1": "Default"})
			saved_model = joblib.load(result.artifact_paths["model"])
			prediction_frame = frame.drop(columns="loan_status").head()
			self.assertEqual(saved_model.predict(prediction_frame).shape, (5,))
			self.assertTrue(
				np.isfinite(saved_model.predict_scores(prediction_frame)).all()
			)
			self.assertNotIn(
				"loan_status", saved_model.feature_engineer.feature_names_out_
			)
			self.assertEqual(result.training_rows + result.test_rows, len(frame))
			self.assertEqual(result.dropped_duplicate_rows, 0)

		pd.testing.assert_frame_equal(frame, original)

	def test_missing_target_labels_are_dropped_and_other_errors_are_clear(self) -> None:
		frame = self._sample_data()
		frame["loan_status"] = frame["loan_status"].astype(float)
		frame.loc[0, "loan_status"] = np.nan
		with tempfile.TemporaryDirectory() as output:
			result = train_loan_models(frame, models_dir=output)
		self.assertEqual(result.dropped_unlabeled_rows, 1)
		self.assertEqual(result.training_rows + result.test_rows, len(frame) - 1)

		frame = self._sample_data().drop(columns="loan_status")
		with tempfile.TemporaryDirectory() as output:
			with self.assertRaisesRegex(ValueError, "target column"):
				train_loan_models(frame, models_dir=output)

		frame = self._sample_data()
		frame.loc[0, "loan_status"] = 2
		with tempfile.TemporaryDirectory() as output:
			with self.assertRaisesRegex(ValueError, "verified 0/1 encoding"):
				train_loan_models(frame, models_dir=output)

	def test_exact_duplicates_are_removed_before_the_split(self) -> None:
		frame = self._sample_data()
		frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
		with tempfile.TemporaryDirectory() as output:
			result = train_loan_models(frame, models_dir=output)

		self.assertEqual(result.dropped_duplicate_rows, 1)
		self.assertEqual(result.training_rows + result.test_rows, len(frame) - 1)

	def test_missing_dataset_reports_loader_error(self) -> None:
		with tempfile.TemporaryDirectory() as missing_data:
			with self.assertRaisesRegex(DatasetLoadError, "loan_data.csv was not found"):
				train_loan_models(data_dir=missing_data, models_dir=missing_data)

	def test_training_is_repeatable_with_fixed_random_state(self) -> None:
		frame = self._sample_data()
		with tempfile.TemporaryDirectory() as output:
			first = train_loan_models(
				frame, models_dir=Path(output) / "first", random_state=7
			)
			second = train_loan_models(
				frame, models_dir=Path(output) / "second", random_state=7
			)

		pd.testing.assert_frame_equal(first.results, second.results)
		self.assertEqual(first.best_model_name, second.best_model_name)

	def test_invalid_split_and_insufficient_classes_are_rejected(self) -> None:
		with self.assertRaisesRegex(ValueError, "test_size"):
			train_loan_models(self._sample_data(), test_size=0)
		frame = self._sample_data()
		frame["loan_status"] = 0
		with tempfile.TemporaryDirectory() as output:
			with self.assertRaisesRegex(ValueError, "two labeled rows"):
				train_loan_models(frame, models_dir=output)


if __name__ == "__main__":
	unittest.main()
