FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
 PYTHONUNBUFFERED=1 \
 PIP_NO_CACHE_DIR=1 \
 PORT=8080 \
 AGENT_ROOT=/workspace \
 MODEL_SERVER=http://127.0.0.1:8000/v1 \
 MODEL_NAME=Qwen/QwQ-32B \
 API_KEY=EMPTY \
 TEMPERATURE=0.6 \
 TOP_P=0.95 \
 TOP_K=20 \
 MAX_TOKENS=4096 \
 THOUGHT_IN_CONTENT=false

WORKDIR /app

# 假设当前仓库就是 Qwen-Agent 源码仓库（或其 fork/mirror）
COPY . /src/qwen-agent

RUN python -m pip install -U pip setuptools wheel \
 && python -m pip install /src/qwen-agent \
 && python -m pip install fastapi "uvicorn[standard]" json5

COPY agent_qwq_min.py /app/agent_qwq_min.py

RUN mkdir -p /workspace/inbox /workspace/output

EXPOSE 8080

CMD ["python", "/app/agent_qwq_min.py"]
