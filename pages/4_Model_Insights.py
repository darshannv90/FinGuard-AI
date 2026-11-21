import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard import load_dashboard_model
from src.evaluation import (
	OPTIONAL_METRICS,
	calculate_confusion_metrics,
	feature_importance_from_bundle,
	load_evaluation_results,
)


st.title("Model Insights")
st.caption(
	"Central comparison of saved holdout evaluations. This page reads the saved results "
	"and model bundles; it does not retrain models or calculate scores from invented data."
)

st.markdown(
	"""
	- **Precision:** Of the cases flagged as fraud/default, how many were actually positive?
	- **Recall:** Of all actual fraud/default cases, how many did the model find?
	- **F1-score:** A single score balancing precision and recall.
	- **ROC-AUC:** How well the model ranks positives above negatives across thresholds.
	- **PR-AUC:** Precision/recall ranking quality; especially informative when positives are rare.
	- **False positive (FP):** A negative case incorrectly flagged as positive.
	- **False negative (FN):** A positive case the model missed.
	"""
)

evaluation_tables: dict[str, pd.DataFrame] = {}
for dataset_name, display_name in (("fraud", "Fraud Detection"), ("loan", "Loan Default Risk")):
	try:
		evaluation_tables[dataset_name] = load_evaluation_results(dataset_name)
	except FileNotFoundError:
		st.warning(f"No saved {display_name.lower()} evaluation was found. Train and save its models first.")
	except (OSError, ValueError) as error:
		st.error(f"The saved {display_name.lower()} evaluation could not be loaded: {error}")

if not evaluation_tables:
	st.info("No saved model evaluations are available yet.")
	st.stop()

combined_results = pd.concat(evaluation_tables.values(), ignore_index=True)
st.subheader("Model comparison")
display_columns = [
	"Dataset",
	"Model",
	"Precision",
	"Recall",
	"F1-score",
	"ROC-AUC",
	"PR-AUC",
	"FP",
	"FN",
	"Confusion Matrix",
	"Selected",
]
st.dataframe(
	combined_results[display_columns].style.format(
		{
			"Precision": "{:.3f}",
			"Recall": "{:.3f}",
			"F1-score": "{:.3f}",
			"ROC-AUC": "{:.3f}",
			"PR-AUC": "{:.3f}",
		},
		na_rep="N/A",
	),
	width="stretch",
	hide_index=True,
)

chart_data = combined_results.melt(
	id_vars=["Dataset", "Model"],
	value_vars=["Precision", "Recall", "F1-score", *OPTIONAL_METRICS],
	var_name="Metric",
	value_name="Score",
).dropna(subset=["Score"])
if not chart_data.empty:
	st.plotly_chart(
		px.bar(
			chart_data,
			x="Model",
			y="Score",
			color="Metric",
			barmode="group",
			facet_col="Dataset",
			range_y=[0, 1],
			title="Saved holdout metrics by model",
		),
		width="stretch",
	)

for dataset_name, result_table in evaluation_tables.items():
	is_fraud = dataset_name == "fraud"
	display_name = "Fraud detection" if is_fraud else "Loan default risk"
	positive_class = "fraud transaction" if is_fraud else "default"
	negative_class = "legitimate transaction" if is_fraud else "non-default"
	st.divider()
	st.header(display_name)

	if is_fraud:
		st.warning(
			"Fraud errors have two distinct costs: a false negative is a fraud transaction "
			"that passes undetected, while a false positive is a legitimate transaction "
			"raised as an alarm. Review both counts; high overall scores do not eliminate missed fraud."
		)
	else:
		st.info(
			"A false negative here is an actual default predicted as non-default, which can "
			"lead to an unrecognized credit loss. A false positive is a non-default applicant "
			"flagged as default, potentially causing an unnecessary decline or review. These "
			"are model error counts, not estimates of monetary impact."
		)

	for _, row in result_table.iterrows():
		model_name = str(row["Model"])
		with st.expander(f"{model_name} evaluation", expanded=False):
			metric_columns = st.columns(5)
			for column, metric in zip(
				metric_columns,
				("Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"),
			):
				value = row[metric]
				column.metric(metric, "N/A" if pd.isna(value) else f"{value:.3f}")

			counts = calculate_confusion_metrics(
				int(row["TN"]),
				int(row["FP"]),
				int(row["FN"]),
				int(row["TP"]),
			)
			error_columns = st.columns(4)
			error_columns[0].metric(
				f"False alarms (FP)",
				f"{counts['FP']:,}",
				help=f"{negative_class.title()} cases incorrectly predicted as {positive_class}.",
			)
			error_columns[1].metric(
				f"Missed {positive_class}s (FN)",
				f"{counts['FN']:,}",
				help=f"Actual {positive_class}s incorrectly predicted as {negative_class}.",
			)
			error_columns[2].metric("False alarm rate", f"{counts['False positive rate']:.2%}")
			error_columns[3].metric("Miss rate", f"{counts['False negative rate']:.2%}")

			matrix = np.array(
				[[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]],
				dtype=int,
			)
			figure = go.Figure(
				go.Heatmap(
					z=matrix,
					x=[f"Predicted {negative_class}", f"Predicted {positive_class}"],
					y=[f"Actual {negative_class}", f"Actual {positive_class}"],
					colorscale="Blues",
					showscale=False,
					hovertemplate="%{y}<br>%{x}<br>Count: %{z:,}<extra></extra>",
				)
			)
			for matrix_row, row_label in enumerate(
				(f"Actual {negative_class}", f"Actual {positive_class}")
			):
				for matrix_column, column_label in enumerate(
					(f"Predicted {negative_class}", f"Predicted {positive_class}")
				):
					figure.add_annotation(
						x=column_label,
						y=row_label,
						text=f"{matrix[matrix_row, matrix_column]:,}",
						showarrow=False,
					)
			figure.update_layout(
				title="Confusion matrix (rows: actual, columns: predicted)",
				height=330,
				margin={"l": 20, "r": 20, "t": 55, "b": 70},
			)
			st.plotly_chart(figure, width="stretch")

	selected_models = result_table.loc[result_table["Selected"]]
	if len(selected_models) == 1:
		selected_name = str(selected_models.iloc[0]["Model"])
		st.subheader(f"Feature importance — selected {selected_name}")
		try:
			bundle = load_dashboard_model(dataset_name)
			importance, method = feature_importance_from_bundle(bundle)
		except (OSError, ValueError, TypeError) as error:
			st.warning(f"Feature importance is unavailable: {error}")
		else:
			if importance.empty or method is None:
				st.info(f"{selected_name} does not expose supported feature importance values.")
			else:
				st.caption(
					f"{method}. These values describe this fitted model; they are not causal effects."
				)
				top_features = importance.head(15).sort_values(
					"Importance", ascending=True, kind="stable"
				)
				st.plotly_chart(
					px.bar(
						top_features,
						x="Importance",
						y="Feature",
						orientation="h",
						title=f"Top {len(top_features)} model features",
					),
					width="stretch",
				)
	elif len(selected_models) > 1:
		st.warning("Feature importance is skipped because the saved evaluation selects multiple models.")

with st.expander("Model limitations and interpretation"):
	st.markdown(
		"""
		- Metrics describe the saved holdout split, not guaranteed performance on future data.
		- Fraud labels are rare; false negatives and false alarms should be reviewed alongside ranking metrics.
		- Loan error counts do not account for loan size, recovery, approval policy, or the financial cost of each mistake.
		- A model score may not be calibrated as a real-world probability. Risk thresholds need validation against operating costs.
		- Tree importance and coefficient magnitudes describe fitted model behavior; they do not establish causal effects.
		- The source datasets omit some real-world signals (for example, customer identifiers in the card dataset); the dashboard does not invent them.
		"""
	)
