"""UK Motor Insurance: Retail Pricing, Customer Behaviour and Optimisation. Simulated data only.

Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

import views  # noqa: E402

st.set_page_config(page_title="UK Motor Retail Pricing (Simulated data)", layout="wide")

PAGES = {
    "Executive overview": views.executive_overview,
    "Competitor and PCW position": views.competitor_position,
    "Conversion model performance": views.conversion_models,
    "Elasticity and scenario testing": views.elasticity_scenarios,
    "Optimisation results": views.optimisation_results,
    "Retention and CLV": views.retention_clv,
    "Governance and monitoring": views.governance,
    "Manual quote input": views.manual_input,
}

st.sidebar.title("UK Motor Retail Pricing")
st.sidebar.caption("Simulated data")
choice = st.sidebar.radio("Page", list(PAGES))
PAGES[choice]()
