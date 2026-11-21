from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard import clear_dashboard_model_cache, load_dashboard_dataset
from src.data_loader import RAW_DATA_DIR, DatasetLoadError
from src.evaluation import DATASET_ARTIFACTS, MODELS_DIR, load_evaluation_results
from src.loan_features import LOAN_TARGET_COLUMN, summarize_loan_data


def _load_local_dataset(dataset_name: str) -> pd.DataFrame | None:
	try:
		with st.spinner(f"Loading the local {dataset_name} dataset..."):
			return load_dashboard_dataset(dataset_name)
	except DatasetLoadError as error:
		st.warning(
			f"{dataset_name.title()} dataset unavailable: {error} "
			f"Add the expected CSV to `{RAW_DATA_DIR}` or use the upload on its analysis page."
		)
		return None


def _selected_model(dataset_name: str) -> tuple[str, float | None] | None:
	model_path = MODELS_DIR / DATASET_ARTIFACTS[dataset_name][1]
	if not model_path.is_file():
		return None
	try:
		results = load_evaluation_results(dataset_name)
	except FileNotFoundError:
		return None
	except (OSError, ValueError) as error:
		st.error(f"The saved {dataset_name} model evaluation could not be loaded: {error}")
		return None
	selected = results.loc[results["Selected"]]
	if len(selected) != 1:
		return None
	row = selected.iloc[0]
	score = row["PR-AUC"]
	return str(row["Model"]), float(score) if pd.notna(score) else None


st.title("Overview")
st.caption(
	"FinGuard AI financial risk analytics for credit-card fraud and loan default. "
	"Predictions use saved model artifacts; opening the dashboard never starts training."
)

fraud_frame = _load_local_dataset("fraud")
loan_frame = _load_local_dataset("loan")

fraud_rows: int | None = None
fraud_cases: int | None = None
if fraud_frame is not None:
	fraud_rows = len(fraud_frame)
	if "Class" in fraud_frame:
		fraud_labels = pd.to_numeric(fraud_frame["Class"], errors="coerce")
		fraud_cases = int(fraud_labels.eq(1).sum())

loan_rows: int | None = None
loan_defaults: int | None = None
loan_default_rate: float | None = None
if loan_frame is not None:
	loan_rows = len(loan_frame)
	try:
		loan_summary = summarize_loan_data(loan_frame)
	except ValueError as error:
		st.error(f"Loan target statistics are unavailable: {error}")
	else:
		distribution = loan_summary["target_distribution"].set_index(LOAN_TARGET_COLUMN)
		loan_defaults = int(distribution.loc[1, "count"])
		loan_default_rate = float(loan_summary["default_rate"])

fraud_model = _selected_model("fraud")
loan_model = _selected_model("loan")

kpi_columns = st.columns(5)
kpi_columns[0].metric(
	"Fraud transactions",
	"N/A" if fraud_rows is None else f"{fraud_rows:,}",
	help="Actual rows in the local credit-card dataset.",
)
kpi_columns[1].metric(
	"Observed fraud cases",
	"N/A" if fraud_cases is None else f"{fraud_cases:,}",
	help="Rows labeled Class=1 in the local dataset.",
)
kpi_columns[2].metric(
	"Loan applicants",
	"N/A" if loan_rows is None else f"{loan_rows:,}",
	help="Actual rows in the local loan dataset.",
)
kpi_columns[3].metric(
	"Observed defaults",
	"N/A" if loan_defaults is None else f"{loan_defaults:,}",
	help="Rows validated as loan_status=1.",
)
kpi_columns[4].metric(
	"Observed default rate",
	"N/A" if loan_default_rate is None else f"{loan_default_rate:.2%}",
	help="Calculated from the dataset's validated loan_status labels.",
)

st.subheader("Model status")
status_columns = st.columns(2)
for column, dataset_name, model_info in (
	(status_columns[0], "fraud", fraud_model),
	(status_columns[1], "loan", loan_model),
):
	label = "Credit-card fraud" if dataset_name == "fraud" else "Loan default"
	if model_info is None:
		column.warning(
			f"{label}: no selected trained model/evaluation is available. "
			"Use the training controls below after placing the dataset."
		)
	else:
		model_name, pr_auc = model_info
		column.success(f"{label}: {model_name} is available.")
		column.metric(
			"Saved holdout PR-AUC",
			"N/A" if pr_auc is None else f"{pr_auc:.3f}",
		)

chart_columns = st.columns(2)
if fraud_frame is not None and "Class" in fraud_frame:
	fraud_distribution = (
		pd.to_numeric(fraud_frame["Class"], errors="coerce")
		.value_counts(dropna=False)
		.rename_axis("Class")
		.reset_index(name="Transactions")
	)
	fraud_distribution["Class label"] = fraud_distribution["Class"].map(
		{0: "Legitimate", 1: "Fraud"}
	).fillna("Unknown / missing")
	fraud_distribution = fraud_distribution.groupby(
		"Class label", as_index=False
	)["Transactions"].sum()
	chart_columns[0].plotly_chart(
		px.bar(
			fraud_distribution,
			x="Class label",
			y="Transactions",
			color="Class label",
			title="Observed credit-card class distribution",
		),
		width="stretch",
	)
if loan_frame is not None and loan_default_rate is not None:
	loan_distribution = summarize_loan_data(loan_frame)["target_distribution"].copy()
	loan_distribution["Outcome"] = loan_distribution["loan_status"].map(
		{0: "Non-default", 1: "Default"}
	)
	chart_columns[1].plotly_chart(
		px.bar(
			loan_distribution,
			x="Outcome",
			y="count",
			color="Outcome",
			title="Observed loan outcomes",
		),
		width="stretch",
	)

st.divider()
with st.expander("Dedicated model training controls", expanded=False):
	st.caption(
		"Training runs only after an explicit button click. It is not triggered by page loads. "
		"Each run evaluates the real local dataset and replaces the corresponding saved artifacts."
	)
	training_columns = st.columns(2)
	if training_columns[0].button("Train / retrain fraud models", type="primary"):
		from src.train_fraud import train_fraud_models

		status = st.status("Preparing fraud training", expanded=True)
		progress_bar = st.progress(0)

		def report_progress(progress: float, message: str) -> None:
			status.update(label=message)
			progress_bar.progress(int(progress * 100))

		try:
			result = train_fraud_models(progress_callback=report_progress)
		except (DatasetLoadError, OSError, ValueError) as error:
			status.update(label="Fraud model training failed", state="error")
			st.error(str(error))
		else:
			status.update(label="Fraud model training complete", state="complete")
			clear_dashboard_model_cache()
			st.success(
				f"Selected {result.best_model_name}; measured PR-AUC "
				f"{result.best_pr_auc:.4f}."
			)
			st.dataframe(result.results, width="stretch", hide_index=True)

	if training_columns[1].button("Train / retrain loan models", type="primary"):
		from src.train_loan import train_loan_models

		status = st.status("Training loan models", expanded=True)
		try:
			result = train_loan_models()
		except (DatasetLoadError, OSError, ValueError) as error:
			status.update(label="Loan model training failed", state="error")
			st.error(str(error))
		else:
			status.update(label="Loan model training complete", state="complete")
			clear_dashboard_model_cache()
			st.success(
				f"Selected {result.best_model_name}; measured PR-AUC "
				f"{result.best_pr_auc:.4f}."
			)
			st.dataframe(result.results, width="stretch", hide_index=True)

st.caption(
	"Data is read from local CSV files and uploads are processed in memory. "
	"Model scores support review and are not guarantees of future outcomes."
)
