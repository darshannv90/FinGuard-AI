from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard import (
	categorize_risk_scores,
	csv_download_bytes,
	inspect_input_quality,
	load_dashboard_dataset,
	load_dashboard_model,
	model_artifact_revision,
	prepare_model_inputs,
	read_uploaded_csv,
)
from src.data_loader import RAW_DATA_DIR, DatasetLoadError


st.title("Loan Default Risk Analysis")
st.caption(
	"Assess a single applicant or score a batch with the saved loan model. "
	"Predictions are decision support, not a credit decision or guarantee."
)

risk_bands = st.sidebar.slider(
	"Risk score category cutoffs (%)",
	min_value=0,
	max_value=100,
	value=(30, 70),
	help="Low is below the first cutoff; medium is between cutoffs; high is at or above the second.",
)
st.caption(
	f"Risk categories use configurable score bands: Low < {risk_bands[0]}%, "
	f"Medium < {risk_bands[1]}%, High ≥ {risk_bands[1]}%. These are operational "
	"triage bands, not empirically validated lending thresholds."
)

uploaded_file = st.file_uploader(
	"Upload applicants for batch scoring (CSV)",
	type=["csv"],
	help="Include every required feature column listed by the saved model; loan_status is optional and is never used as an input.",
)
if uploaded_file is not None:
	try:
		applicant_frame = read_uploaded_csv(uploaded_file)
	except DatasetLoadError as error:
		st.error(str(error))
		st.stop()
else:
	try:
		with st.spinner("Loading loan applicant data..."):
			applicant_frame = load_dashboard_dataset("loan")
		st.caption(
			"Showing the local loan dataset for quality review. For a meaningful independent "
			"batch assessment, upload new applicants; scoring the training dataset is not a holdout evaluation."
		)
	except DatasetLoadError as error:
		applicant_frame = None
		st.warning(
			f"{error} Add `loan_data.csv` under `{RAW_DATA_DIR}` or upload an applicant CSV above."
		)

try:
	with st.spinner("Loading saved loan model..."):
		bundle = load_dashboard_model("loan")
except FileNotFoundError as error:
	bundle = None
	st.warning(
		f"No trained loan model is available: {error}. Train from the Overview page "
		"before assessing applicants."
	)
except (OSError, ValueError, TypeError) as error:
	bundle = None
	st.error(f"The saved loan model could not be loaded: {error}")

if applicant_frame is not None:
	numeric_columns = (
		list(bundle.feature_engineer.numeric_columns_)
		if bundle is not None
		else applicant_frame.select_dtypes(include="number").columns.tolist()
	)
	quality = inspect_input_quality(applicant_frame, numeric_columns)
	quality_columns = st.columns(4)
	quality_columns[0].metric("Applicant rows", f"{quality['rows']:,}")
	quality_columns[1].metric("Missing cells", f"{quality['missing_cells']:,}")
	quality_columns[2].metric("Exact duplicate rows", f"{quality['duplicate_rows']:,}")
	quality_columns[3].metric(
		"Invalid numeric cells", f"{quality['invalid_numeric_cells']:,}"
	)
	st.info(
		"Rows are preserved during scoring. Fitted model preprocessing handles missing values "
		"and unseen categories; uploaded data is not saved."
	)
	st.subheader("Applicant data preview")
	st.dataframe(applicant_frame.head(20), width="stretch", hide_index=True)

	if bundle is not None:
		ignored_columns: list[str] = []
		try:
			batch_inputs, ignored_columns = prepare_model_inputs(
				applicant_frame, bundle, "loan"
			)
		except (TypeError, ValueError) as error:
			batch_inputs = None
			st.error(f"This batch cannot be scored: {error}")
		if ignored_columns:
			st.caption(
				"Additional columns are ignored by this model: "
				+ ", ".join(map(str, ignored_columns))
			)

		if batch_inputs is not None and st.button(
			"Run batch predictions", type="primary"
		):
			status = st.status("Scoring applicants with the saved loan model", expanded=True)
			try:
				predictions = bundle.predict(batch_inputs)
				probabilities = np.asarray(bundle.predict_scores(batch_inputs), dtype=float)
				if (
					len(predictions) != len(batch_inputs)
					or not np.isfinite(probabilities).all()
					or ((probabilities < 0) | (probabilities > 1)).any()
				):
					raise ValueError("The saved model returned invalid default probabilities.")
				result_frame = applicant_frame.copy(deep=True)
				result_frame["Predicted default"] = predictions.astype("int8")
				result_frame["Default probability"] = probabilities
				result_frame["Risk score (0-100)"] = probabilities * 100
				result_frame["Risk category"] = categorize_risk_scores(
					probabilities, risk_bands[0] / 100, risk_bands[1] / 100
				)
			except (OSError, TypeError, ValueError) as error:
				status.update(label="Loan batch scoring failed", state="error")
				st.error(str(error))
			else:
				status.update(label="Loan batch scoring complete", state="complete")
				signature = (
					int(pd.util.hash_pandas_object(applicant_frame, index=True).sum()),
					risk_bands,
					model_artifact_revision("loan"),
				)
				st.session_state["loan_dashboard_batch_predictions"] = {
					"signature": signature,
					"data": result_frame,
				}

		if bundle is not None:
			signature = (
				int(pd.util.hash_pandas_object(applicant_frame, index=True).sum()),
				risk_bands,
				model_artifact_revision("loan"),
			)
			batch_state = st.session_state.get("loan_dashboard_batch_predictions")
			if batch_state is not None and batch_state["signature"] == signature:
				batch_result = batch_state["data"]
				st.subheader("Batch prediction results")
				count_columns = st.columns(3)
				count_columns[0].metric(
					"Predicted defaults",
					f"{int(batch_result['Predicted default'].sum()):,}",
				)
				count_columns[1].metric(
					"Scored applicants", f"{len(batch_result):,}"
				)
				count_columns[2].metric(
					"High risk",
					f"{int(batch_result['Risk category'].eq('High').sum()):,}",
				)
				risk_distribution = (
					batch_result["Risk category"]
					.value_counts()
					.rename_axis("Risk category")
					.reset_index(name="Applicants")
				)
				st.plotly_chart(
					px.bar(
						risk_distribution,
						x="Risk category",
						y="Applicants",
						color="Risk category",
						title="Scored applicant risk categories",
					),
					width="stretch",
				)
				st.dataframe(batch_result.head(2000), width="stretch", hide_index=True)
				if len(batch_result) > 2000:
					st.caption("Showing the first 2,000 rows; the download contains the full scored batch.")
				st.download_button(
					"Download batch risk report",
					data=csv_download_bytes(batch_result),
					file_name="loan_default_risk_predictions.csv",
					mime="text/csv",
				)
			elif batch_state is not None:
				st.info("Applicant data or risk cutoffs changed. Run batch predictions again.")

		st.divider()
		st.subheader("Single-applicant assessment")
		engineer = bundle.feature_engineer
		numeric_fields = list(engineer.numeric_columns_)
		categorical_fields = list(engineer.categorical_columns_)
		with st.form("loan_applicant_assessment"):
			st.caption("Enter the applicant's actual values. No sample applicant is prefilled.")
			numeric_inputs: dict[str, float | None] = {}
			categorical_inputs: dict[str, str] = {}
			numeric_grid = st.columns(3)
			for index, field in enumerate(numeric_fields):
				with numeric_grid[index % len(numeric_grid)]:
					numeric_inputs[field] = st.number_input(
						field.replace("_", " ").title(),
						value=None,
						step=0.1,
						format="%.2f",
						placeholder="Required",
					)
			categorical_grid = st.columns(3)
			for index, field in enumerate(categorical_fields):
				values = list(engineer.categorical_values_.get(field, []))
				options = [""] + values
				with categorical_grid[index % len(categorical_grid)]:
					categorical_inputs[field] = st.selectbox(
						field.replace("_", " ").title(),
						options=options,
						index=0,
						format_func=lambda value: "Choose..." if value == "" else value,
					)
			assess = st.form_submit_button("Assess applicant risk", type="primary")

		if assess:
			missing_values = [
				field for field, value in numeric_inputs.items() if value is None
			] + [
				field for field, value in categorical_inputs.items() if not value
			]
			if missing_values:
				st.error(
					"Complete all required fields before assessment: "
					+ ", ".join(missing_values)
				)
			else:
				applicant = pd.DataFrame(
					[{**numeric_inputs, **categorical_inputs}]
				)
				try:
					probability = float(bundle.predict_scores(applicant)[0])
					prediction = int(bundle.predict(applicant)[0])
					if not np.isfinite(probability) or not 0 <= probability <= 1:
						raise ValueError("The saved model returned an invalid default probability.")
					category = categorize_risk_scores(
						np.array([probability]),
						risk_bands[0] / 100,
						risk_bands[1] / 100,
					)[0]
				except (OSError, TypeError, ValueError) as error:
					st.error(f"Applicant assessment failed: {error}")
				else:
					result_columns = st.columns(3)
					result_columns[0].metric("Default probability", f"{probability:.1%}")
					result_columns[1].metric("Risk score", f"{probability * 100:.1f} / 100")
					result_columns[2].metric("Risk category", category)
					st.write(
						"Model class: **Default**"
						if prediction == 1
						else "Model class: **Non-default**"
					)
					st.caption(
						"Risk category is based on the configurable score bands. The score is a "
						"model output, not a guaranteed default probability or lending decision."
					)
