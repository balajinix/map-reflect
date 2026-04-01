"""Streaming DQ Pipeline Control — Streamlit App.

Provides a web UI for:
  Tab 1: Upload CSV → Kafka
  Tab 2: Run Flink transformation → Iceberg
  Tab 3: Query Iceberg tables
"""

import time

import pandas as pd
import streamlit as st

from pipeline import (
    cancel_job,
    create_kafka_topic,
    generate_query_sql,
    generate_transform_sql,
    get_running_jobs,
    parse_tableau_output,
    produce_to_kafka,
    run_flink_sql,
    sanitize_name,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Streaming Data Pipeline", page_icon="🔄", layout="wide")
st.title("Streaming Data Pipeline")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "topic" not in st.session_state:
    st.session_state.topic = None
if "columns" not in st.session_state:
    st.session_state.columns = []
if "output_table" not in st.session_state:
    st.session_state.output_table = None
if "known_tables" not in st.session_state:
    st.session_state.known_tables = []

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "1. Upload & Ingest to Kafka",
    "2. Transform with Flink",
    "3. Query Iceberg",
])

# ===== TAB 1: Upload & Ingest =============================================
with tab1:
    st.header("Upload CSV & Send to Kafka")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        topic_name = f"upload_{sanitize_name(uploaded_file.name)}"

        st.subheader("Preview")
        st.dataframe(df, use_container_width=True)
        st.info(f"**Topic:** `{topic_name}` — **Rows:** {len(df)} — **Columns:** {list(df.columns)}")

        if st.button("Send to Kafka", type="primary"):
            with st.spinner("Creating topic and producing messages..."):
                try:
                    create_kafka_topic(topic_name)
                    records = df.astype(str).to_dict(orient="records")
                    count = produce_to_kafka(topic_name, records)

                    st.session_state.topic = topic_name
                    st.session_state.columns = list(df.columns)

                    st.success(f"Delivered **{count}** messages to topic `{topic_name}`")
                except Exception as e:
                    st.error(f"Failed: {e}")

# ===== TAB 2: Transform with Flink ========================================
with tab2:
    st.header("Run Flink Transformation")

    if st.session_state.topic is None:
        st.warning("Upload a CSV in Tab 1 first.")
    else:
        topic = st.session_state.topic
        columns = st.session_state.columns

        st.info(f"**Source topic:** `{topic}` — **Columns:** {columns}")

        transform = st.selectbox(
            "Transformation",
            ["UPPERCASE", "lowercase", "TRIM whitespace"],
        )
        output_table = f"{sanitize_name(topic)}_{sanitize_name(transform)}"
        st.text(f"Output Iceberg table: streamlit_db.{output_table}")

        if st.button("Run Transformation", type="primary"):
            with st.spinner("Generating SQL and submitting Flink job..."):
                try:
                    sql = generate_transform_sql(topic, columns, transform, output_table)

                    with st.expander("Generated SQL", expanded=False):
                        st.code(sql, language="sql")

                    result = run_flink_sql(sql, wait=False)

                    if result.returncode != 0 and result.stderr:
                        st.error(f"Flink SQL error:\n```\n{result.stderr}\n```")
                    else:
                        st.session_state.output_table = output_table
                        if output_table not in st.session_state.known_tables:
                            st.session_state.known_tables.append(output_table)
                        st.success(
                            "Job submitted! The streaming INSERT job is running. "
                            "Wait ~60s for a checkpoint cycle, then cancel the job below."
                        )
                except Exception as e:
                    st.error(f"Failed: {e}")

        # -- Job status & cancel --
        st.subheader("Flink Jobs")
        if st.button("Refresh Job List"):
            pass  # button click triggers rerun

        jobs = get_running_jobs()
        if jobs:
            for job in jobs:
                jid = job.get("jid", "")
                name = job.get("name", "unknown")
                state = job.get("state", "UNKNOWN")
                col1, col2 = st.columns([3, 1])
                col1.text(f"{jid[:12]}…  {name}  [{state}]")
                if state == "RUNNING":
                    if col2.button("Cancel", key=f"cancel_{jid}"):
                        if cancel_job(jid):
                            st.success(f"Cancelled job {jid[:12]}")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Cancel request failed")
        else:
            st.caption("No jobs found (Flink REST API may be unreachable).")

# ===== TAB 3: Query Iceberg ===============================================
with tab3:
    st.header("Query Iceberg Table")

    # Table name input
    default_table = st.session_state.output_table or ""
    known = st.session_state.known_tables

    if known:
        table_name = st.selectbox("Select a table", known, index=known.index(default_table) if default_table in known else 0)
    else:
        st.warning("No tables available yet. Run a transformation in Tab 2 first.")
        table_name = None

    if st.button("Query", type="primary") and table_name:
        with st.spinner(f"Querying `streamlit_db.{table_name}` ..."):
            try:
                sql = generate_query_sql(table_name)

                with st.expander("Generated SQL", expanded=False):
                    st.code(sql, language="sql")

                result = run_flink_sql(sql, wait=True)

                if result.returncode != 0:
                    st.error(f"Query failed:\n```\n{result.stderr}\n```")
                else:
                    df = parse_tableau_output(result.stdout)
                    if df.empty:
                        st.warning("Query returned no rows (or output could not be parsed).")
                        with st.expander("Raw output"):
                            st.code(result.stdout)
                    else:
                        st.success(f"Returned **{len(df)}** rows")
                        st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Failed: {e}")
