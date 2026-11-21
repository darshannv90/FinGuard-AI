import unittest

import numpy as np
import pandas as pd

from src.fraud_features import FraudFeatureEngineer, engineer_fraud_features


class FraudFeatureEngineerTests(unittest.TestCase):
	def test_features_exclude_class_and_use_training_statistics(self) -> None:
		training = pd.DataFrame(
			{
				"Time": [0, 3600, 7200],
				"Amount": [1.0, 3.0, 5.0],
				"V1": [10.0, 20.0, 30.0],
				"Class": [0, 1, 0],
			}
		)
		engineer = FraudFeatureEngineer(high_amount_quantile=0.75).fit(training)
		prediction = pd.DataFrame(
			{"Time": [86400], "Amount": [100.0], "V1": [40.0], "Class": [1]}
		)

		result = engineer.transform(prediction)

		self.assertNotIn("Class", result.columns)
		self.assertEqual(result.loc[0, "amount_deviation_from_training_mean"], 97.0)
		self.assertAlmostEqual(
			result.loc[0, "amount_z_score"], 97.0 / np.std([1.0, 3.0, 5.0])
		)
		self.assertEqual(result.loc[0, "is_high_transaction_amount"], 1)
		self.assertEqual(result.loc[0, "transaction_hour"], 0)
		self.assertEqual(result.loc[0, "transaction_elapsed_hours"], 24.0)
		training_features = engineer.transform(training)
		self.assertNotIn("Class", training_features.columns)
		self.assertEqual(training_features.columns.tolist(), result.columns.tolist())

	def test_missing_invalid_and_infinite_values_are_imputed(self) -> None:
		training = pd.DataFrame(
			{
				"Time": [0, 3600, 7200],
				"Amount": [2.0, 4.0, 6.0],
				"V1": ["2", "invalid", "6"],
				"Class": [0, 0, 1],
			}
		)
		prediction = pd.DataFrame(
			{
				"Time": ["invalid", np.inf],
				"Amount": ["bad", np.inf],
				"V1": ["bad", "10"],
				"Class": [1, 0],
			}
		)
		engineer = FraudFeatureEngineer().fit(training)

		result = engineer.transform(prediction)

		self.assertTrue(np.isfinite(result.to_numpy(dtype=float)).all())
		self.assertEqual(result.loc[0, "Amount"], 4.0)
		self.assertEqual(result.loc[1, "Time"], 3600.0)
		self.assertEqual(result.loc[0, "V1"], 4.0)
		self.assertEqual(result["transaction_hour"].tolist(), [1, 1])

	def test_constant_amounts_produce_finite_anomaly_features(self) -> None:
		training = pd.DataFrame({"Amount": [5.0, 5.0, 5.0], "Class": [0, 1, 0]})

		result = FraudFeatureEngineer().fit_transform(training)

		self.assertTrue(
			np.isfinite(
				result[
					["amount_z_score", "amount_robust_z_score", "amount_log1p"]
				].to_numpy()
			).all()
		)
		self.assertEqual(result["amount_z_score"].tolist(), [0.0, 0.0, 0.0])
		self.assertEqual(result["amount_robust_z_score"].tolist(), [0.0, 0.0, 0.0])

	def test_convenience_function_is_deterministic_and_does_not_mutate_input(self) -> None:
		training = pd.DataFrame(
			{"Time": [0, 3600], "Amount": [10.0, 20.0], "Class": [0, 1]}
		)
		original = training.copy(deep=True)

		first = engineer_fraud_features(training, training)
		second = engineer_fraud_features(training, training)

		pd.testing.assert_frame_equal(first, second)
		pd.testing.assert_frame_equal(training, original)
		self.assertNotIn("Class", first.columns)
		self.assertFalse(any("velocity" in column or "rolling" in column for column in first))

	def test_fit_validation_and_duplicate_columns(self) -> None:
		with self.assertRaisesRegex(ValueError, "empty"):
			FraudFeatureEngineer().fit(pd.DataFrame())
		with self.assertRaisesRegex(ValueError, "duplicate column"):
			FraudFeatureEngineer().fit(
				pd.DataFrame([[1.0, 2.0]], columns=["Amount", "Amount"])
			)
		with self.assertRaisesRegex(ValueError, "quantile"):
			FraudFeatureEngineer(high_amount_quantile=0)


if __name__ == "__main__":
	unittest.main()
