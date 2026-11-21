import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.evaluation import (
	calculate_confusion_metrics,
	feature_importance_from_bundle,
	load_evaluation_results,
)


class EvaluationTests(unittest.TestCase):
	def test_confusion_metrics_and_error_rates(self) -> None:
		metrics = calculate_confusion_metrics(tn=90, fp=10, fn=20, tp=80)

		self.assertAlmostEqual(metrics["Precision"], 80 / 90)
		self.assertEqual(metrics["Recall"], 0.8)
		self.assertAlmostEqual(metrics["F1-score"], 2 * (80 / 90) * 0.8 / ((80 / 90) + 0.8))
		self.assertEqual(metrics["False positive rate"], 0.1)
		self.assertEqual(metrics["False negative rate"], 0.2)

	def test_confusion_metrics_handle_empty_prediction_denominators(self) -> None:
		metrics = calculate_confusion_metrics(tn=0, fp=0, fn=5, tp=0)

		self.assertEqual(metrics["Precision"], 0)
		self.assertEqual(metrics["Recall"], 0)
		self.assertEqual(metrics["F1-score"], 0)
		self.assertEqual(metrics["False positive rate"], 0)
		self.assertEqual(metrics["False negative rate"], 1)

	def test_confusion_metrics_reject_invalid_counts(self) -> None:
		with self.assertRaisesRegex(ValueError, "nonnegative integer"):
			calculate_confusion_metrics(tn=1, fp=-1, fn=0, tp=1)
		with self.assertRaisesRegex(ValueError, "nonnegative integer"):
			calculate_confusion_metrics(tn=1, fp=0.5, fn=0, tp=1)

	def test_saved_evaluation_is_loaded_and_validated_against_counts(self) -> None:
		with tempfile.TemporaryDirectory() as output:
			pd.DataFrame(
				[
					{
						"Model": "Random Forest",
						"Precision": 0.8,
						"Recall": 0.5,
						"F1-score": 2 * 0.8 * 0.5 / (0.8 + 0.5),
						"ROC-AUC": 0.9,
						"PR-AUC": 0.7,
						"TN": 95,
						"FP": 5,
						"FN": 20,
						"TP": 20,
						"Selected": True,
					}
				]
			).to_csv(Path(output) / "fraud_evaluation_results.csv", index=False)

			results = load_evaluation_results("fraud", models_dir=output)

		self.assertEqual(results.loc[0, "Dataset"], "Fraud")
		self.assertTrue(results.loc[0, "Selected"])
		self.assertEqual(results.loc[0, "FP"], 5)
		self.assertEqual(results.loc[0, "FN"], 20)
		self.assertEqual(results.loc[0, "Confusion Matrix"], "[[95, 5], [20, 20]]")

	def test_inconsistent_metrics_and_missing_results_fail_explicitly(self) -> None:
		with tempfile.TemporaryDirectory() as output:
			pd.DataFrame(
				[
					{
						"Model": "Broken",
						"Precision": 0.1,
						"Recall": 0.5,
						"F1-score": 0.2,
						"TN": 90,
						"FP": 10,
						"FN": 20,
						"TP": 20,
					}
				]
			).to_csv(Path(output) / "loan_model_evaluation_results.csv", index=False)
			with self.assertRaisesRegex(ValueError, "does not match"):
				load_evaluation_results("loan", models_dir=output)
			with self.assertRaisesRegex(FileNotFoundError, "was not found"):
				load_evaluation_results("fraud", models_dir=output)

	def test_feature_importance_uses_supported_model_values(self) -> None:
		bundle = SimpleNamespace(
			estimator=SimpleNamespace(feature_importances_=np.array([0.2, 0.8])),
			feature_engineer=SimpleNamespace(
				feature_names_out_=["income", "loan_amount"]
			),
		)
		importance, method = feature_importance_from_bundle(bundle)

		self.assertEqual(method, "Tree impurity-based feature importance")
		self.assertEqual(importance["Feature"].tolist(), ["loan_amount", "income"])
		self.assertEqual(importance["Importance"].tolist(), [0.8, 0.2])

	def test_feature_importance_supports_linear_models_and_reports_unsupported(self) -> None:
		linear_bundle = SimpleNamespace(
			estimator=SimpleNamespace(coef_=np.array([[2.0, -1.0]])),
			feature_engineer=SimpleNamespace(feature_names_out_=["income", "grade"]),
		)
		importance, method = feature_importance_from_bundle(linear_bundle)
		self.assertEqual(method, "Absolute model coefficients")
		self.assertEqual(importance["Importance"].tolist(), [2.0, 1.0])

		unsupported_bundle = SimpleNamespace(
			estimator=SimpleNamespace(),
			feature_engineer=SimpleNamespace(feature_names_out_=["income"]),
		)
		unsupported, method = feature_importance_from_bundle(unsupported_bundle)
		self.assertTrue(unsupported.empty)
		self.assertIsNone(method)

	def test_feature_importance_rejects_mismatched_names(self) -> None:
		bundle = SimpleNamespace(
			estimator=SimpleNamespace(feature_importances_=np.array([1.0, 0.0])),
			feature_engineer=SimpleNamespace(feature_names_out_=["income"]),
		)
		with self.assertRaisesRegex(ValueError, "does not match"):
			feature_importance_from_bundle(bundle)


if __name__ == "__main__":
	unittest.main()
