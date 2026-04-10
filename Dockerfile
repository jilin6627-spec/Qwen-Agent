FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
 PYTHONUNBUFFERED=1 \
 PIP_NO_CACHE_DIR=1 \
 START_GUI=true \
 START_API=true \
 GUI_HOST=0.0.0.0 \
 GUI_PORT=7860 \
 API_HOST=0.0.0.0 \
 API_PORT=8080 \
 AGENT_ROOT=/workspace \
 MODEL_SERVER=http://127.0.0.1:8000/v1 \
 MODEL_NAME=Qwen/QwQ-32B \
 API_KEY=EMPTY \
 TEMPERATURE=0.6 \
 TOP_P=0.95 \
 TOP_K=20 \
 MAX_TOKENS=4096 \
 THOUGHT_IN_CONTENT=false \
 USE_RAW_API=true \
 ENABLE_THINKING=false

WORKDIR /app

# 当前仓库就是 Qwen-Agent 源码仓库（或其 fork/mirror）
COPY . /src/qwen-agent

# 从源码安装 Qwen-Agent，并启用 GUI / MCP / Code Interpreter 相关依赖
RUN python -m pip install -U pip setuptools wheel \
 && python -m pip install -e '/src/qwen-agent[gui,mcp,code_interpreter]' \
 && python -m pip install uvicorn fastapi json5

COPY agent_qwq_gui.py /app/agent_qwq_gui.py

RUN mkdir -p /workspace/inbox /workspace/output

EXPOSE 7860 8080

CMD ["python", "/app/agent_qwq_gui.py"]
