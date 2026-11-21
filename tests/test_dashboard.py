from io import BytesIO
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.dashboard import (
	categorize_risk_scores,
	csv_download_bytes,
	inspect_input_quality,
	prepare_model_inputs,
	read_uploaded_csv,
)
from src.data_loader import DatasetLoadError


class DashboardHelperTests(unittest.TestCase):
	def test_csv_upload_accepts_valid_data_and_rejects_bad_headers_or_content(self) -> None:
		valid_upload = BytesIO(b"Amount,V1\n12.5,0.2\n")
		frame = read_uploaded_csv(valid_upload)
		self.assertEqual(frame.to_dict(orient="records"), [{"Amount": 12.5, "V1": 0.2}])

		with self.assertRaisesRegex(DatasetLoadError, "duplicate column"):
			read_uploaded_csv(BytesIO(b"Amount,Amount\n1,2\n"))
		with self.assertRaisesRegex(DatasetLoadError, "empty"):
			read_uploaded_csv(BytesIO(b""))
		with self.assertRaisesRegex(DatasetLoadError, "valid CSV"):
			read_uploaded_csv(BytesIO(b"Amount,V1\n\"unterminated,1\n"))

	def test_download_csv_bytes_round_trip_without_an_index(self) -> None:
		frame = pd.DataFrame(
			{"Applicant name": ["Example, Applicant"], "Risk score": [73.5]}
		)

		payload = csv_download_bytes(frame)
		restored = pd.read_csv(BytesIO(payload))

		pd.testing.assert_frame_equal(restored, frame)
		with self.assertRaises(TypeError):
			csv_download_bytes(["not", "a", "dataframe"])

	def test_model_inputs_drop_labels_and_report_ignored_fields(self) -> None:
		bundle = SimpleNamespace(
			feature_engineer=SimpleNamespace(
				numeric_columns_=["Amount", "V1"],
				categorical_columns_=[],
			)
		)
		frame = pd.DataFrame(
			{
				"Amount": [12.5],
				"V1": [0.2],
				"Class": [1],
				"note": ["ignored"],
			}
		)

		model_inputs, ignored = prepare_model_inputs(frame, bundle, "fraud")

		self.assertEqual(model_inputs.columns.tolist(), ["Amount", "V1", "note"])
		self.assertEqual(ignored, ["note"])
		self.assertNotIn("Class", model_inputs)

	def test_model_input_validation_reports_missing_fields(self) -> None:
		bundle = SimpleNamespace(
			feature_engineer=SimpleNamespace(
				numeric_columns_=["person_income"],
				categorical_columns_=["loan_intent"],
			)
		)
		frame = pd.DataFrame({"person_income": [50000]})

		with self.assertRaisesRegex(ValueError, "loan_intent"):
			prepare_model_inputs(frame, bundle, "loan")

	def test_input_quality_counts_invalid_missing_duplicate_and_outlier_rows(self) -> None:
		frame = pd.DataFrame(
			{
				"amount": [1.0, 1.0, 1.0, 1.0, 5.0, 10000.0, "bad", np.nan],
				"category": [
					"same",
					"same",
					"same",
					"same",
					"middle",
					"high",
					"invalid",
					"missing",
				],
			}
		)

		report = inspect_input_quality(frame, ["amount"])

		self.assertEqual(report["missing_cells"], 1)
		self.assertEqual(report["duplicate_rows"], 3)
		self.assertEqual(report["invalid_numeric_cells"], 1)
		self.assertEqual(report["invalid_rows"], 1)
		self.assertEqual(report["outliers_by_column"]["amount"], 1)
		self.assertEqual(report["outlier_rows"], 1)

	def test_risk_categories_use_configured_boundaries(self) -> None:
		categories = categorize_risk_scores(
			np.array([0.0, 0.3, 0.7, 1.0]),
			medium_cutoff=0.3,
			high_cutoff=0.7,
		)

		self.assertEqual(categories.tolist(), ["Low", "Medium", "High", "High"])
		with self.assertRaisesRegex(ValueError, "cutoffs"):
			categorize_risk_scores(np.array([0.5]), medium_cutoff=0.8, high_cutoff=0.7)
		with self.assertRaisesRegex(ValueError, "probabilities"):
			categorize_risk_scores(np.array([1.2]), medium_cutoff=0.3, high_cutoff=0.7)


if __name__ == "__main__":
	unittest.main()
