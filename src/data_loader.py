"""Read-only loading and inspection helpers for FinGuard AI datasets."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass(frozen=True)
class DatasetSpec:
	filename: str
	target_column: str
	label: str


DATASETS = {
	"fraud": DatasetSpec(
		filename="creditcard.csv",
		target_column="Class",
		label="Credit card fraud",
	),
	"loan": DatasetSpec(
		filename="loan_data.csv",
		target_column="loan_status",
		label="Loan risk",
	),
}


class DatasetLoadError(ValueError):
	"""A dataset could not be loaded or does not match its expected format."""


def validate_csv_headers(source: Any, source_name: str) -> None:
	"""Reject empty CSVs and duplicate raw headers before pandas can rename them."""
	try:
		if isinstance(source, (str, Path)):
			with Path(source).open("r", encoding="utf-8-sig", newline="") as input_file:
				headers = next(csv.reader(input_file))
		else:
			source.seek(0)
			text_stream = io.TextIOWrapper(source, encoding="utf-8-sig", newline="")
			try:
				headers = next(csv.reader(text_stream))
			finally:
				text_stream.detach()
				source.seek(0)
	except StopIteration as error:
		raise DatasetLoadError(f"{source_name} is empty and has no CSV header.") from error
	except (csv.Error, UnicodeDecodeError, OSError, ValueError) as error:
		raise DatasetLoadError(
			f"Could not read the CSV header from {source_name}. Check its encoding and format."
		) from error
	if len(headers) != len(set(headers)):
		raise DatasetLoadError(f"{source_name} contains duplicate column names.")


def _get_spec(dataset_name: str) -> DatasetSpec:
	spec = DATASETS.get(dataset_name.strip().lower())
	if spec is None:
		choices = ", ".join(DATASETS)
		raise DatasetLoadError(
			f"Unknown dataset '{dataset_name}'. Choose one of: {choices}."
		)
	return spec


def _read_csv(source: Any, spec: DatasetSpec, source_name: str) -> pd.DataFrame:
	try:
		validate_csv_headers(source, source_name)
		frame = pd.read_csv(source, low_memory=False)
	except DatasetLoadError:
		raise
	except pd.errors.EmptyDataError as error:
		raise DatasetLoadError(f"{source_name} is empty and has no CSV data.") from error
	except (pd.errors.ParserError, UnicodeDecodeError, OSError, ValueError) as error:
		raise DatasetLoadError(
			f"Could not read {source_name} as a valid CSV file. Check its encoding and format."
		) from error

	if frame.empty:
		raise DatasetLoadError(f"{source_name} contains column headers but no data rows.")
	if spec.target_column not in frame.columns:
		raise DatasetLoadError(
			f"{source_name} does not look like the {spec.label} dataset. "
			f"Expected a '{spec.target_column}' column."
		)
	return frame


def load_dataset(
	dataset_name: str,
	data_dir: str | Path | None = None,
) -> pd.DataFrame:
	"""Load a named dataset from the raw-data directory without modifying it."""
	spec = _get_spec(dataset_name)
	directory = Path(data_dir) if data_dir is not None else RAW_DATA_DIR
	path = directory / spec.filename

	if path.suffix.lower() != ".csv":
		raise DatasetLoadError(f"Expected a CSV file for {spec.label}.")
	if not path.is_file():
		raise DatasetLoadError(
			f"{path.name} was not found in {directory}. "
			"Download the dataset or upload a CSV file to inspect it."
		)

	return _read_csv(path, spec, path.name)


def load_fraud_dataset(data_dir: str | Path | None = None) -> pd.DataFrame:
	"""Load ``creditcard.csv`` from the raw-data directory."""
	return load_dataset("fraud", data_dir=data_dir)


def load_loan_dataset(data_dir: str | Path | None = None) -> pd.DataFrame:
	"""Load ``loan_data.csv`` from the raw-data directory."""
	return load_dataset("loan", data_dir=data_dir)


def load_uploaded_dataset(dataset_name: str, uploaded_file: Any) -> pd.DataFrame:
	"""Read an uploaded CSV in memory; this function never saves the upload."""
	spec = _get_spec(dataset_name)
	filename = str(getattr(uploaded_file, "name", ""))
	if Path(filename).suffix.lower() != ".csv":
		raise DatasetLoadError("Please select a file with the .csv extension.")

	try:
		if hasattr(uploaded_file, "seek"):
			uploaded_file.seek(0)
	except (OSError, ValueError) as error:
		raise DatasetLoadError("The uploaded file could not be read.") from error

	return _read_csv(uploaded_file, spec, Path(filename).name)


def inspect_dataset(frame: pd.DataFrame) -> dict[str, Any]:
	"""Return structural, missingness, duplicate, category, and summary details."""
	if not isinstance(frame, pd.DataFrame):
		raise TypeError("frame must be a pandas DataFrame")
	if frame.columns.has_duplicates:
		raise ValueError("DataFrame contains duplicate column names")

	missing_counts = frame.isna().sum().astype(int)
	missing_percentages = (frame.isna().mean() * 100).round(2)
	numeric_columns = frame.select_dtypes(include="number").columns.tolist()
	categorical_columns = frame.select_dtypes(
		include=["object", "category", "string", "bool"]
	).columns.tolist()

	return {
		"row_count": int(frame.shape[0]),
		"column_count": int(frame.shape[1]),
		"column_names": frame.columns.tolist(),
		"data_types": frame.dtypes.astype(str).to_dict(),
		"missing_counts": missing_counts.to_dict(),
		"missing_percentages": missing_percentages.to_dict(),
		"duplicate_count": int(frame.duplicated().sum()),
		"numeric_columns": numeric_columns,
		"categorical_columns": categorical_columns,
		"categorical_unique_values": {
			column: frame[column].dropna().unique().tolist()
			for column in categorical_columns
		},
		"descriptive_statistics": frame.describe(include="all").transpose(),
	}
