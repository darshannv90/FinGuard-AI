import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.train_fraud import train_fraud_models


class FraudTrainingTests(unittest.TestCase):
	@staticmethod
	def _sample_data() -> pd.DataFrame:
		rng = np.random.default_rng(42)
		labels = np.array([0] * 100 + [1] * 20)
		return pd.DataFrame(
			{
				"Time": np.arange(len(labels), dtype=float) * 60,
				"Amount": rng.normal(25, 8, len(labels)) + labels * 35,
				"V1": rng.normal(size=len(labels)) + labels * 2,
				"Class": labels,
			}
		)

	def test_models_are_compared_and_selected_pipeline_is_saved(self) -> None:
		frame = self._sample_data()
		original = frame.copy(deep=True)
		progress: list[tuple[float, str]] = []

		with tempfile.TemporaryDirectory() as output:
			result = train_fraud_models(
				frame,
				models_dir=output,
				progress_callback=lambda value, message: progress.append((value, message)),
			)

			self.assertEqual(
				result.results["Model"].tolist(),
				["Logistic Regression", "Random Forest", "Isolation Forest"],
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
			for artifact in result.artifact_paths.values():
				self.assertTrue(Path(artifact).is_file())
			saved_model = joblib.load(result.artifact_paths["model"])
			predictions = saved_model.predict(frame.drop(columns="Class").head())
			self.assertEqual(predictions.shape, (5,))
			self.assertNotIn("Class", saved_model.feature_engineer.numeric_columns_)
			self.assertEqual(progress[-1][0], 1.0)
			self.assertEqual(result.training_rows + result.test_rows, len(frame))
			self.assertEqual(result.dropped_duplicate_rows, 0)

		pd.testing.assert_frame_equal(frame, original)

	def test_unlabeled_rows_are_dropped_and_invalid_labels_rejected(self) -> None:
		frame = self._sample_data()
		frame.loc[0, "Class"] = np.nan
		with tempfile.TemporaryDirectory() as output:
			result = train_fraud_models(frame, models_dir=output)
		self.assertEqual(result.dropped_unlabeled_rows, 1)
		self.assertEqual(result.training_rows + result.test_rows, len(frame) - 1)

		frame.loc[1, "Class"] = 2
		with tempfile.TemporaryDirectory() as output:
			with self.assertRaisesRegex(ValueError, "binary 0/1"):
				train_fraud_models(frame, models_dir=output)

	def test_exact_duplicates_are_removed_before_the_split(self) -> None:
		frame = self._sample_data()
		frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
		with tempfile.TemporaryDirectory() as output:
			result = train_fraud_models(frame, models_dir=output)

		self.assertEqual(result.dropped_duplicate_rows, 1)
		self.assertEqual(result.training_rows + result.test_rows, len(frame) - 1)

	def test_fixed_random_state_repeats_model_metrics(self) -> None:
		frame = self._sample_data()
		with tempfile.TemporaryDirectory() as output:
			first = train_fraud_models(frame, models_dir=Path(output) / "first")
			second = train_fraud_models(frame, models_dir=Path(output) / "second")

		pd.testing.assert_frame_equal(first.results, second.results)
		self.assertEqual(first.best_model_name, second.best_model_name)

	def test_invalid_split_and_missing_target_are_rejected(self) -> None:
		with self.assertRaisesRegex(ValueError, "test_size"):
			train_fraud_models(self._sample_data(), test_size=1)
		with tempfile.TemporaryDirectory() as output:
			with self.assertRaisesRegex(ValueError, "Class"):
				train_fraud_models(
					self._sample_data().drop(columns="Class"), models_dir=output
				)


if __name__ == "__main__":
	unittest.main()
