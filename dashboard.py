import streamlit as st
import pandas as pd
import altair as alt
import os

def show_dashboard():
    st.title("Driver Monitoring Dashboard")

    file_path = "ddd1.csv"

    # Check if CSV exists
    if not os.path.exists(file_path):
        st.warning("No log file found. Please run the live monitoring system first.")
        st.stop()

    # Load data
    df = pd.read_csv(file_path, parse_dates=["timestamp"])

    # Sidebar filters
    st.sidebar.header("Filter")
    status_filter = st.sidebar.multiselect("Select status", options=df['status'].unique(), default=df['status'].unique())

    # Filter data
    filtered_df = df[df['status'].isin(status_filter)]

    # Show recent logs
    st.subheader("Recent Activity")
    st.dataframe(filtered_df.sort_values(by="timestamp", ascending=False).head(10), use_container_width=True)

    # Status frequency bar chart
    st.subheader("Status Frequency")
    freq_chart = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X("status:N", title="Driver Status"),
        y=alt.Y("count():Q", title="Count"),
        color="status:N",
        tooltip=["status:N", "count():Q"]
    ).properties(width=600, height=400)

    st.altair_chart(freq_chart)

    # Time series line chart
    st.subheader("Timeline of Statuses")
    time_chart = alt.Chart(filtered_df).mark_line(point=True).encode(
        x=alt.X("timestamp:T", title="Time"),
        y=alt.Y("status:N", title="Status"),
        color="status:N",
        tooltip=["timestamp:T", "status:N"]
    ).properties(width=700, height=400)

    st.altair_chart(time_chart)
