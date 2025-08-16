# app.py
import streamlit as st

st.set_page_config(page_title="Distracted Driver Detection System", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Live Monitoring", "Dashboard"])

if page == "Live Monitoring":
    from live_monitoring import run_live_monitoring
    run_live_monitoring()
elif page == "Dashboard":
    from dashboard import show_dashboard
    show_dashboard()
