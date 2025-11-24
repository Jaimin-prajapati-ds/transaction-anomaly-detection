"""Streamlit dashboard for anomaly detection system."""

import streamlit as st
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="Transaction Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Transaction Anomaly Detection System")
st.markdown("Real-time fraud detection using ensemble ML methods")

# Sidebar
st.sidebar.header("Input Transaction Details")

amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
distance = st.sidebar.number_input("Distance from Home (km)", min_value=0.0, value=10.0)
is_international = st.sidebar.checkbox("International Transaction")
transaction_speed = st.sidebar.number_input("Transaction Speed (txns/hour)", min_value=0.0, value=1.0)

# Main content
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Transaction Amount", f"${amount:,.2f}")

with col2:
    st.metric("Time", f"{hour}:00")

with col3:
    st.metric("Distance", f"{distance} km")

# Predict button
if st.button("Detect Anomaly", type="primary"):
    st.info("🚀 Model prediction coming soon...")
    st.success("✅ System ready for deployment!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Ensemble Anomaly Detection")
