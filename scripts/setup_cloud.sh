#!/usr/bin/env bash
# Только Linux cloud. Не меняет исходники, lock/constraints и чужой .env.
set -euo pipefail
cd /workspace/project_iis1818_danya
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/.python
uv python install --no-bin 3.11.17
if [ ! -x .venv/bin/python ]; then
  uv venv --python /workspace/.python/cpython-3.11-linux-x86_64-gnu/bin/python3.11 .venv
fi
uv pip install --python .venv/bin/python 'torch==2.5.1+cpu' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt -c constraints.txt
uv pip check --python .venv/bin/python
if [ ! -e .env ]; then
  cat > .env <<'ENV'
LLM_BACKEND=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
GIGACHAT_VERIFY_SSL=true
HF_HOME=/workspace/.cache/huggingface
HF_HUB_OFFLINE=1
ENV
fi
export HF_HOME=/workspace/.cache/huggingface
# Модель уже в snapshot; при обновлении среды разрешён безопасный догруз.
HF_HUB_OFFLINE=0 .venv/bin/python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('intfloat/multilingual-e5-small', cache_folder='/workspace/.cache/huggingface/hub')
PY
.venv/bin/python scripts/download_corpus.py
.venv/bin/python scripts/check_corpus.py
export HF_HUB_OFFLINE=1
.venv/bin/python ingest.py
# Официальная Ollama 0.40.11, проверяется опубликованная контрольная сумма.
ollama_dir=/workspace/tools/ollama
mkdir -p "$ollama_dir"
if [ ! -x "$ollama_dir/bin/ollama" ]; then
  curl -fL --max-time 300 https://github.com/ollama/ollama/releases/download/v0.40.11/ollama-linux-amd64.tar.zst -o "$ollama_dir/ollama-linux-amd64.tar.zst"
  echo '7bedab90dd122661be78798e9f0edf37fccf0cccf2699d2c58ea1a3afdc04b6d  /workspace/tools/ollama/ollama-linux-amd64.tar.zst' | sha256sum -c -
  tar --zstd -xf "$ollama_dir/ollama-linux-amd64.tar.zst" -C "$ollama_dir"
fi
bash scripts/start_cloud.sh --llm-only
.venv/bin/python - <<'PY'
import requests
r = requests.post('http://127.0.0.1:11434/api/pull', json={'model': 'qwen2.5:3b', 'stream': False}, timeout=300)
r.raise_for_status()
assert r.json().get('status') == 'success'
PY
.venv/bin/python -m unittest discover -s tests -v
