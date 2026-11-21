from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard import (
	csv_download_bytes,
	inspect_input_quality,
	load_dashboard_dataset,
	load_dashboard_model,
	model_artifact_revision,
	prepare_model_inputs,
	read_uploaded_csv,
)
from src.data_loader import RAW_DATA_DIR, DatasetLoadError
from src.evaluation import load_evaluation_results


st.title("Credit Card Fraud Detection")
st.caption(
	"Score transactions with the saved fraud model. Uploads are read in memory and "
	"never saved to the raw-data folder."
)

uploaded_file = st.file_uploader(
	"Upload transactions as CSV",
	type=["csv"],
	help="Include the trained model's feature columns. Class is optional and is never sent as a model input.",
)
if uploaded_file is not None:
	try:
		transaction_frame = read_uploaded_csv(uploaded_file)
	except DatasetLoadError as error:
		st.error(str(error))
		st.stop()
else:
	try:
		with st.spinner("Loading credit-card transaction data..."):
			transaction_frame = load_dashboard_dataset("fraud")
	except DatasetLoadError as error:
		st.warning(
			f"{error} Add `creditcard.csv` under `{RAW_DATA_DIR}` or upload a CSV above."
		)
		st.stop()

try:
	with st.spinner("Loading saved fraud model..."):
		bundle = load_dashboard_model("fraud")
except FileNotFoundError as error:
	bundle = None
	st.warning(
		f"No trained fraud model is available: {error}. Train from the Overview page "
		"or run the dedicated fraud training function before scoring."
	)
except (OSError, ValueError, TypeError) as error:
	bundle = None
	st.error(f"The saved fraud model could not be loaded: {error}")

numeric_columns = (
	list(bundle.feature_engineer.numeric_columns_)
	if bundle is not None
	else transaction_frame.select_dtypes(include="number").columns.tolist()
)
quality = inspect_input_quality(transaction_frame, numeric_columns)
quality_columns = st.columns(4)
quality_columns[0].metric("Transactions", f"{quality['rows']:,}")
quality_columns[1].metric("Missing cells", f"{quality['missing_cells']:,}")
quality_columns[2].metric("Exact duplicate rows", f"{quality['duplicate_rows']:,}")
quality_columns[3].metric(
	"Invalid numeric cells", f"{quality['invalid_numeric_cells']:,}"
)
st.info(
	"Rows are preserved during scoring. The saved training-fitted feature pipeline handles "
	"missing and invalid numeric feature values; this page does not alter or save the uploaded CSV."
)

st.subheader("Dataset preview")
st.dataframe(transaction_frame.head(20), width="stretch", hide_index=True)
if bundle is not None:
	try:
		model_inputs, ignored_columns = prepare_model_inputs(
			transaction_frame, bundle, "fraud"
		)
	except (TypeError, ValueError) as error:
		st.error(f"This dataset cannot be scored: {error}")
		st.stop()
	if ignored_columns:
		st.caption(
			"Additional columns are ignored by this model: "
			+ ", ".join(map(str, ignored_columns))
		)

	uses_probability = (
		bundle.model_name != "Isolation Forest"
		and hasattr(bundle.estimator, "predict_proba")
	)
	if uses_probability:
		risk_bands = st.sidebar.slider(
			"Fraud risk category cutoffs (%)",
			min_value=0,
			max_value=100,
			value=(25, 50),
			help=(
				"Operational review buckets only. They do not calibrate the model's "
				"probability or change its predicted class."
			),
		)
		st.caption(
			"Risk categories are configurable triage bands. Model probabilities may not be "
			"calibrated; the model's predicted class remains separate."
		)
	else:
		risk_bands = None
		st.info(
			"This saved model returns an anomaly score rather than a calibrated probability. "
			"The page labels it as a score and uses the model's flagged/not-flagged output."
		)

	current_signature = (
		int(pd.util.hash_pandas_object(transaction_frame, index=True).sum()),
		risk_bands,
		model_artifact_revision("fraud"),
	)
	if st.button("Run fraud predictions", type="primary"):
		status = st.status("Scoring transactions with the saved model", expanded=True)
		try:
			predictions = bundle.predict(model_inputs)
			scores = np.asarray(bundle.predict_scores(model_inputs), dtype=float)
			if len(predictions) != len(transaction_frame) or not np.isfinite(scores).all():
				raise ValueError("The saved model returned invalid prediction output.")
			result_frame = transaction_frame.copy(deep=True)
			result_frame["Predicted fraud"] = predictions.astype("int8")
			if uses_probability:
				if ((scores < 0) | (scores > 1)).any():
					raise ValueError("The saved classifier returned scores outside [0, 1].")
				result_frame["Fraud probability"] = scores
				result_frame["Risk category"] = np.where(
					scores >= risk_bands[1] / 100,
					"High",
					np.where(scores >= risk_bands[0] / 100, "Medium", "Low"),
				)
			else:
				result_frame["Anomaly score"] = scores
				result_frame["Risk category"] = np.where(
					predictions == 1, "Flagged", "Not flagged"
				)
		except (OSError, TypeError, ValueError) as error:
			status.update(label="Fraud scoring failed", state="error")
			st.error(str(error))
		else:
			status.update(label="Fraud scoring complete", state="complete")
			st.session_state["fraud_dashboard_predictions"] = {
				"signature": current_signature,
				"data": result_frame,
				"uses_probability": uses_probability,
			}

	prediction_state = st.session_state.get("fraud_dashboard_predictions")
	if prediction_state is not None and prediction_state["signature"] == current_signature:
		prediction_frame = prediction_state["data"]
		flagged = prediction_frame.loc[prediction_frame["Predicted fraud"].eq(1)]
		result_columns = st.columns(3)
		result_columns[0].metric("Flagged transactions", f"{len(flagged):,}")
		result_columns[1].metric(
			"Unflagged transactions",
			f"{len(prediction_frame) - len(flagged):,}",
		)
		result_columns[2].metric("Scored rows", f"{len(prediction_frame):,}")

		st.subheader("Flagged transaction review")
		risk_options = prediction_frame["Risk category"].dropna().unique().tolist()
		selected_categories = st.multiselect(
			"Filter by risk category",
			options=risk_options,
			default=risk_options,
		)
		show_only_flagged = st.checkbox("Show only model-flagged transactions", value=True)
		filtered = prediction_frame.loc[
			prediction_frame["Risk category"].isin(selected_categories)
		]
		if show_only_flagged:
			filtered = filtered.loc[filtered["Predicted fraud"].eq(1)]
		st.dataframe(filtered.head(2000), width="stretch", hide_index=True)
		if len(filtered) > 2000:
			st.caption("Showing the first 2,000 matching rows; the download contains all filtered rows.")
		st.download_button(
			"Download filtered predictions",
			data=csv_download_bytes(filtered),
			file_name="fraud_predictions.csv",
			mime="text/csv",
		)
	elif prediction_state is not None:
		st.info("The input changed after the previous run. Run predictions again to refresh results.")
else:
	st.info("Load a trained fraud model to validate feature columns and score transactions.")

st.divider()
st.subheader("Saved model evaluation")
try:
	evaluation = load_evaluation_results("fraud")
except FileNotFoundError:
	st.info("No saved fraud comparison is available yet. Train the fraud models from Overview.")
except (OSError, ValueError) as error:
	st.error(f"Saved fraud evaluation could not be loaded: {error}")
else:
	metric_columns = [
		column
		for column in (
			"Model",
			"Precision",
			"Recall",
			"F1-score",
			"ROC-AUC",
			"PR-AUC",
			"FP",
			"FN",
			"Selected",
		)
		if column in evaluation
	]
	st.dataframe(
		evaluation[metric_columns].style.format(
			{column: "{:.3f}" for column in ("Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC")},
			na_rep="N/A",
		),
		width="stretch",
		hide_index=True,
	)
	st.warning(
		"Prioritize false negatives (missed fraud) as well as false positives (legitimate "
		"transactions raised as alarms). A lower false-alarm count alone does not mean fraud is fully detected."
	)
