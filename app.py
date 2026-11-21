import streamlit as st


st.set_page_config(
	page_title="FinGuard AI",
	page_icon="🛡️",
	layout="wide",
	initial_sidebar_state="expanded",
)

navigation = st.navigation(
	[
		st.Page(
			"pages/0_Overview.py",
			title="Overview",
			icon=":material/dashboard:",
			default=True,
		),
		st.Page(
			"pages/1_Credit_Card_Fraud_Detection.py",
			title="Credit Card Fraud Detection",
			icon="💳",
		),
		st.Page(
			"pages/2_Loan_Default_Risk_Analysis.py",
			title="Loan Default Risk Analysis",
			icon="🏦",
		),
		st.Page(
			"pages/3_Data_Quality_and_Wrangling.py",
			title="Data Quality & Wrangling",
			icon=":material/cleaning_services:",
		),
		st.Page(
			"pages/4_Model_Insights.py",
			title="Model Insights",
			icon="📊",
		),
	]
)
navigation.run()
