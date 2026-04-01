#!/usr/bin/env bash
set -euo pipefail

# ==========================================================================
# install_streamlit.sh — Install Streamlit and dependencies into the venv
# ==========================================================================

VENV="/data/streaming_dq/venv"

echo "===> Activating virtualenv at ${VENV}"
source "${VENV}/bin/activate"

echo "===> Installing Streamlit and dependencies"
pip install --quiet streamlit confluent-kafka requests

echo "===> Installed versions:"
pip show streamlit confluent-kafka requests 2>/dev/null | grep -E "^(Name|Version):"

echo ""
echo "===> Done. Run the app with:"
echo "  source ${VENV}/bin/activate && cd /home/bganesa1/code/streaming_dq && streamlit run app/app.py --server.address 0.0.0.0"
