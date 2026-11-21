import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.data_loader import (
	DatasetLoadError,
	inspect_dataset,
	load_dataset,
	load_uploaded_dataset,
)


class DatasetLoaderTests(unittest.TestCase):
	def test_loads_expected_dataset_and_inspects_real_structure(self) -> None:
		with tempfile.TemporaryDirectory() as output:
			pd.DataFrame({"Amount": [1.0, 2.0], "Class": [0, 1]}).to_csv(
				Path(output) / "creditcard.csv", index=False
			)

			frame = load_dataset("fraud", data_dir=output)
			report = inspect_dataset(frame)

		self.assertEqual(frame.shape, (2, 2))
		self.assertEqual(report["row_count"], 2)
		self.assertEqual(report["column_names"], ["Amount", "Class"])
		self.assertEqual(report["duplicate_count"], 0)

	def test_missing_file_and_wrong_target_raise_clear_errors(self) -> None:
		with tempfile.TemporaryDirectory() as output:
			with self.assertRaisesRegex(DatasetLoadError, "was not found"):
				load_dataset("loan", data_dir=output)
			pd.DataFrame({"amount": [1]}).to_csv(
				Path(output) / "loan_data.csv", index=False
			)
			with self.assertRaisesRegex(DatasetLoadError, "loan_status"):
				load_dataset("loan", data_dir=output)

	def test_empty_and_duplicate_header_files_are_rejected(self) -> None:
		with tempfile.TemporaryDirectory() as output:
			empty_path = Path(output) / "creditcard.csv"
			empty_path.write_text("", encoding="utf-8")
			with self.assertRaisesRegex(DatasetLoadError, "empty"):
				load_dataset("fraud", data_dir=output)

			empty_path.write_text("Class,Class\n0,1\n", encoding="utf-8")
			with self.assertRaisesRegex(DatasetLoadError, "duplicate column"):
				load_dataset("fraud", data_dir=output)

	def test_uploaded_dataset_is_read_in_memory_and_validated(self) -> None:
		upload = BytesIO(b"person_income,loan_status\n50000,0\n")
		upload.name = "loan_data.csv"

		frame = load_uploaded_dataset("loan", upload)

		self.assertEqual(frame.to_dict(orient="records"), [{"person_income": 50000, "loan_status": 0}])
		with self.assertRaisesRegex(DatasetLoadError, "extension"):
			load_uploaded_dataset("loan", SimpleNamespace(name="loan_data.txt"))

	def test_duplicate_column_names_are_rejected(self) -> None:
		frame = pd.DataFrame([[1, 2]])
		frame.columns = ["Class", "Class"]
		with self.assertRaisesRegex(ValueError, "duplicate column"):
			inspect_dataset(frame)


if __name__ == "__main__":
	unittest.main()
