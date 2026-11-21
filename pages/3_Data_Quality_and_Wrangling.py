from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.cleaning import CleaningConfig, DataCleaningPipeline
from src.dashboard import (
	csv_download_bytes,
	inspect_input_quality,
	load_dashboard_dataset,
	read_uploaded_csv,
)
from src.data_loader import RAW_DATA_DIR, DatasetLoadError
from src.loan_features import LOAN_TARGET_COLUMN, clean_loan_data


st.title("Data Quality & Wrangling")
st.caption(
	"Inspect and clean working copies of actual datasets. Raw CSVs are never overwritten. "
	"Full-data cleaning here is for review/EDA only; model preprocessing is fitted separately."
)

dataset_name = st.selectbox(
	"Dataset",
	options=["fraud", "loan"],
	format_func=lambda value: (
		"Credit-card fraud" if value == "fraud" else "Loan default risk"
	),
)
uploaded_file = st.file_uploader(
	"Optional CSV upload",
	type=["csv"],
	help="An uploaded CSV is analyzed in memory and is not saved.",
)
if uploaded_file is not None:
	try:
		frame = read_uploaded_csv(uploaded_file)
	except DatasetLoadError as error:
		st.error(str(error))
		st.stop()
else:
	try:
		with st.spinner(f"Loading {dataset_name} data for quality review..."):
			frame = load_dashboard_dataset(dataset_name)
	except DatasetLoadError as error:
		filename = "creditcard.csv" if dataset_name == "fraud" else "loan_data.csv"
		st.warning(
			f"{error} Add `{filename}` under `{RAW_DATA_DIR}` or upload a CSV above."
		)
		st.stop()

with st.sidebar:
	st.subheader("Cleaning options")
	remove_duplicates = st.toggle(
		"Remove exact duplicate rows",
		value=True,
		help="No applicant or cardholder identifiers are inferred; only exact row matches are removed.",
	)
	cap_outliers = st.toggle(
		"Cap IQR outliers in working copy",
		value=False,
		help="Off by default. Outliers are review flags, not automatically invalid records.",
	)

target_column = "Class" if dataset_name == "fraud" else LOAN_TARGET_COLUMN
try:
	if dataset_name == "loan" and target_column in frame.columns:
		quality_result = clean_loan_data(
			frame,
			remove_duplicates=remove_duplicates,
			cap_outliers=cap_outliers,
		)
		cleaning = quality_result.cleaning
		consistency_flags = quality_result.consistency_flags
	else:
		config = CleaningConfig(
			remove_duplicates=remove_duplicates,
			outlier_action="cap" if cap_outliers else "flag",
			target_columns=(target_column,) if target_column in frame.columns else (),
		)
		cleaning = DataCleaningPipeline(config).fit_transform(frame)
		consistency_flags = pd.DataFrame(index=frame.index)
except (TypeError, ValueError) as error:
	st.error(f"The dataset could not be cleaned: {error}")
	st.stop()

numeric_columns = frame.select_dtypes(include="number").columns.tolist()
numeric_columns = [column for column in numeric_columns if column != target_column]
input_quality = inspect_input_quality(frame, numeric_columns)
consistency_row_count = (
	int(consistency_flags.any(axis=1).sum()) if len(consistency_flags.columns) else 0
)

metrics = st.columns(5)
metrics[0].metric("Source rows", f"{input_quality['rows']:,}")
metrics[1].metric("Missing cells", f"{input_quality['missing_cells']:,}")
metrics[2].metric("Exact duplicates", f"{input_quality['duplicate_rows']:,}")
metrics[3].metric("Rows with invalid numeric values", f"{input_quality['invalid_rows']:,}")
metrics[4].metric("Domain consistency flags", f"{consistency_row_count:,}")

before = cleaning.before_report
after = cleaning.after_report
comparison = pd.DataFrame(
	[
		{
			"Stage": "Before cleaning",
			"Rows": before["rows"],
			"Missing cells": before["missing_cells"],
			"Duplicate rows": before["duplicate_rows"],
		},
		{
			"Stage": "Cleaned working copy",
			"Rows": after["rows"],
			"Missing cells": after["missing_cells"],
			"Duplicate rows": after["duplicate_rows"],
		},
	]
)
st.subheader("Before / after comparison")
st.dataframe(comparison, width="stretch", hide_index=True)

outlier_counts = cleaning.outlier_flags.sum().astype(int)
outlier_counts = outlier_counts.loc[outlier_counts.gt(0)].sort_values(ascending=False)
left_column, right_column = st.columns(2)
with left_column:
	st.subheader("Outlier review")
	if outlier_counts.empty:
		st.info("No eligible numeric IQR outliers were flagged.")
	else:
		st.plotly_chart(
			px.bar(
				outlier_counts.rename("Flagged rows").rename_axis("Feature").reset_index(),
				x="Feature",
				y="Flagged rows",
				title="IQR outliers by feature",
			),
			width="stretch",
		)
		st.caption("IQR flags identify values for review; they do not prove an error.")
with right_column:
	st.subheader("Invalid and inconsistent records")
	if input_quality["invalid_numeric_cells"]:
		invalid_table = pd.DataFrame(
			[
				{"Feature": field, "Invalid numeric cells": count}
				for field, count in input_quality["invalid_by_column"].items()
				if count
			]
		)
		st.dataframe(invalid_table, width="stretch", hide_index=True)
	else:
		st.success("No invalid numeric cells were found in the inspected numeric fields.")
	if consistency_row_count:
		flag_counts = consistency_flags.sum().astype(int)
		st.dataframe(
			flag_counts.loc[flag_counts.gt(0)].rename("Rows flagged").to_frame(),
			width="stretch",
		)
	corrupted = cleaning.corrupted_categories
	if not corrupted.empty:
		st.dataframe(corrupted, width="stretch", hide_index=True)
	elif not consistency_row_count and not input_quality["invalid_numeric_cells"]:
		st.success("No configured category or domain-consistency issues were found.")

st.subheader("Cleaned working copy")
st.caption(
	f"{after['rows']:,} rows · {after['missing_cells']:,} missing cells · "
	f"{after['duplicate_rows']:,} duplicate rows. Target labels are preserved rather than imputed."
)
st.dataframe(cleaning.data.head(100), width="stretch", hide_index=True)
st.download_button(
	"Download cleaned working copy",
	data=csv_download_bytes(cleaning.data),
	file_name=f"{dataset_name}_cleaned.csv",
	mime="text/csv",
)

with st.expander("Data cleaning log"):
	for entry in cleaning.log:
		st.markdown(f"- {entry}")
