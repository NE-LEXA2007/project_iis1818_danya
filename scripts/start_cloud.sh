#!/usr/bin/env bash
set -euo pipefail
cd /workspace/project_iis1818_danya
mkdir -p /workspace/.logs
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_OFFLINE=1
export OLLAMA_MODELS=/workspace/.cache/ollama/models
export OLLAMA_HOST=127.0.0.1:11434
if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  nohup /workspace/tools/ollama/bin/ollama serve > /workspace/.logs/ollama.log 2>&1 &
  for attempt in $(seq 1 30); do
    if curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then break; fi
    sleep 1
  done
  curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null
fi
if [ "${1:-}" = '--llm-only' ]; then exit 0; fi
if ! curl -fsS --max-time 2 http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1; then
  nohup .venv/bin/streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false > /workspace/.logs/streamlit.log 2>&1 &
  for attempt in $(seq 1 30); do
    if curl -fsS --max-time 2 http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1; then break; fi
    sleep 1
  done
fi
curl -fsS --max-time 2 http://127.0.0.1:8501/_stcore/health
