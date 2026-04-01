"""Backend module for the Streaming DQ Streamlit app.

Handles Kafka ingestion, Flink SQL generation/execution, and Iceberg querying.
"""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FLINK_DIR = "/data/streaming_dq/flink"
KAFKA_DIR = "/data/streaming_dq/kafka"
BOOTSTRAP_SERVERS = "localhost:9092"
FLINK_REST_URL = "http://localhost:8081"
SQL_TEMP_DIR = "/tmp/streaming_dq_sql"
ICEBERG_CATALOG_DDL = """CREATE CATALOG iceberg_catalog WITH (
  'type' = 'iceberg',
  'catalog-type' = 'hive',
  'uri' = 'thrift://localhost:9083',
  'warehouse' = 'file:///data/streaming_dq/iceberg/warehouse'
)"""
ICEBERG_DB = "streamlit_db"

# ---------------------------------------------------------------------------
# Kafka helpers
# ---------------------------------------------------------------------------

def sanitize_name(name: str) -> str:
    """Convert a string to a safe lowercase identifier (alphanumeric + underscore)."""
    name = os.path.splitext(name)[0]  # strip extension
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name.lower()


def create_kafka_topic(topic: str) -> None:
    """Create a Kafka topic idempotently via kafka-topics.sh."""
    cmd = [
        f"{KAFKA_DIR}/bin/kafka-topics.sh",
        "--create",
        "--if-not-exists",
        "--topic", topic,
        "--bootstrap-server", BOOTSTRAP_SERVERS,
        "--partitions", "1",
        "--replication-factor", "1",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def produce_to_kafka(topic: str, records: list[dict]) -> int:
    """Produce records (list of dicts) as JSON messages to a Kafka topic.

    Returns the number of successfully delivered messages.
    """
    from confluent_kafka import Producer

    conf = {"bootstrap.servers": BOOTSTRAP_SERVERS}
    producer = Producer(conf)

    delivered = 0

    def delivery_cb(err, _msg):
        nonlocal delivered
        if err is None:
            delivered += 1

    for record in records:
        value = json.dumps({k: str(v) for k, v in record.items()})
        producer.produce(topic, value=value.encode("utf-8"), callback=delivery_cb)

    producer.flush(timeout=30)
    return delivered


# ---------------------------------------------------------------------------
# Flink SQL generation
# ---------------------------------------------------------------------------

def generate_transform_sql(
    topic: str,
    columns: list[str],
    transform: str,
    output_table: str,
) -> str:
    """Generate a Flink SQL script that reads from Kafka, transforms, and writes to Iceberg."""
    col_defs = ", ".join(f"`{c}` STRING" for c in columns)

    if transform == "UPPERCASE":
        select_exprs = ", ".join(f"UPPER(`{c}`) AS `{c}`" for c in columns)
    elif transform == "lowercase":
        select_exprs = ", ".join(f"LOWER(`{c}`) AS `{c}`" for c in columns)
    elif transform == "TRIM whitespace":
        select_exprs = ", ".join(f"TRIM(`{c}`) AS `{c}`" for c in columns)
    else:
        select_exprs = ", ".join(f"`{c}`" for c in columns)

    group_id = f"streamlit-{topic}-{sanitize_name(transform)}"

    sql = f"""{ICEBERG_CATALOG_DDL};

CREATE TABLE default_catalog.default_database.kafka_source (
  {col_defs}
) WITH (
  'connector' = 'kafka',
  'topic' = '{topic}',
  'properties.bootstrap.servers' = '{BOOTSTRAP_SERVERS}',
  'properties.group.id' = '{group_id}',
  'scan.startup.mode' = 'earliest-offset',
  'format' = 'json'
);

USE CATALOG iceberg_catalog;
CREATE DATABASE IF NOT EXISTS {ICEBERG_DB};
USE {ICEBERG_DB};

CREATE TABLE IF NOT EXISTS `{output_table}` (
  {col_defs}
);

INSERT INTO `{output_table}`
SELECT {select_exprs}
FROM default_catalog.default_database.kafka_source;
"""
    return sql


def generate_query_sql(table_name: str) -> str:
    """Generate a batch-mode Flink SQL script to SELECT * from an Iceberg table."""
    sql = f"""{ICEBERG_CATALOG_DDL};

USE CATALOG iceberg_catalog;
USE {ICEBERG_DB};

SET 'execution.runtime-mode' = 'batch';
SET 'sql-client.execution.result-mode' = 'tableau';

SELECT * FROM `{table_name}`;
"""
    return sql


# ---------------------------------------------------------------------------
# Flink execution
# ---------------------------------------------------------------------------

def _ensure_sql_dir() -> str:
    os.makedirs(SQL_TEMP_DIR, exist_ok=True)
    return SQL_TEMP_DIR


def run_flink_sql(sql_content: str, wait: bool = False) -> subprocess.CompletedProcess:
    """Write SQL to a temp file and execute via sql-client.sh -f.

    If wait=False (default for streaming jobs), the process is started and
    returns immediately via Popen. If wait=True (batch queries), we wait for
    completion and capture output.
    """
    _ensure_sql_dir()
    fd, path = tempfile.mkstemp(dir=SQL_TEMP_DIR, suffix=".sql", prefix="flink_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(sql_content)

        cmd = [f"{FLINK_DIR}/bin/sql-client.sh", "-f", path]

        if wait:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120,
            )
            return result
        else:
            # For streaming jobs: run in background, don't wait
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            # Wait a few seconds for initial errors
            try:
                proc.wait(timeout=5)
                # If it exited quickly, capture output (likely an error)
                stdout = proc.stdout.read() if proc.stdout else ""
                stderr = proc.stderr.read() if proc.stderr else ""
                return subprocess.CompletedProcess(
                    cmd, proc.returncode, stdout, stderr,
                )
            except subprocess.TimeoutExpired:
                # Still running — that's expected for streaming jobs
                return subprocess.CompletedProcess(cmd, 0, "Job submitted (running in background)", "")
    finally:
        # Clean up temp file after a delay for batch; keep for streaming
        if wait:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Flink REST API helpers
# ---------------------------------------------------------------------------

def get_running_jobs() -> list[dict]:
    """Get all jobs from Flink REST API /jobs/overview."""
    try:
        resp = requests.get(f"{FLINK_REST_URL}/jobs/overview", timeout=5)
        resp.raise_for_status()
        return resp.json().get("jobs", [])
    except requests.RequestException:
        return []


def get_job_status(job_id: str) -> dict:
    """Get status of a specific Flink job."""
    try:
        resp = requests.get(f"{FLINK_REST_URL}/jobs/{job_id}", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return {}


def cancel_job(job_id: str) -> bool:
    """Cancel a Flink job via PATCH /jobs/<id>?mode=cancel."""
    try:
        resp = requests.patch(
            f"{FLINK_REST_URL}/jobs/{job_id}", params={"mode": "cancel"}, timeout=5,
        )
        return resp.status_code in (200, 202)
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# Tableau output parser
# ---------------------------------------------------------------------------

def parse_tableau_output(output: str) -> pd.DataFrame:
    """Parse Flink SQL client tableau-format output into a DataFrame.

    Tableau output looks like:
    +------+-------+
    | col1 |  col2 |
    +------+-------+
    | val1 |  val2 |
    | ...  |  ...  |
    +------+-------+
    """
    lines = output.strip().splitlines()

    # Find lines that start with '|' (header and data rows)
    data_lines = [l for l in lines if l.strip().startswith("|")]

    if not data_lines:
        return pd.DataFrame()

    # First data_line is the header
    header_line = data_lines[0]
    columns = [c.strip() for c in header_line.split("|") if c.strip()]

    # Remaining lines are data
    rows = []
    for line in data_lines[1:]:
        values = [v.strip() for v in line.split("|") if v.strip()]
        if len(values) == len(columns):
            rows.append(values)

    return pd.DataFrame(rows, columns=columns)
