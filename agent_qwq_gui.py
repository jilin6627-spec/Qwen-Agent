import json
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path
from typing import Any, List, Optional

import json5
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool


def env_bool(name: str, default: bool = False) -> bool:
 v = os.getenv(name)
 if v is None:
 return default
 return v.strip().lower() in {"1", "true", "yes", "on"}


AGENT_ROOT = Path(os.getenv("AGENT_ROOT", "/workspace")).resolve()

MODEL_SERVER = os.getenv("MODEL_SERVER", "http://127.0.0.1:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/QwQ-32B")
API_KEY = os.getenv("API_KEY", "EMPTY")

GUI_HOST = os.getenv("GUI_HOST", "0.0.0.0")
GUI_PORT = int(os.getenv("GUI_PORT", "7860"))

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8080"))

START_GUI = env_bool("START_GUI", True)
START_API = env_bool("START_API", True)

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
TOP_K = int(os.getenv("TOP_K", "20"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

# 官方文档说明，这个参数会影响 tool-call 解析逻辑：
# True: content 里包含 <think>...</think> 和 answer
# False: 服务返回 reasoning_content + content 两段
THOUGHT_IN_CONTENT = env_bool("THOUGHT_IN_CONTENT", False)

# 对自托管 vLLM / SGLang，Qwen-Agent 官方示例使用 use_raw_api=True。
USE_RAW_API = env_bool("USE_RAW_API", True)

# 对 QwQ-32B，通常先不要强行塞 enable_thinking。
# 若后端需要 thinking 参数，再自行打开。
ENABLE_THINKING = env_bool("ENABLE_THINKING", False)


def _safe_path(user_path: str) -> Path:
 p = Path(user_path)
 if not p.is_absolute():
 p = AGENT_ROOT / p
 p = p.resolve()

 if AGENT_ROOT not in p.parents and p != AGENT_ROOT:
 raise ValueError(f"path out of allowed root: {p}")
 return p


@register_tool("list_dir")
class ListDirTool(BaseTool):
 description = "列出目录中的文件和子目录。只能访问允许的工作目录。"
 parameters = [
 {
 "name": "path",
 "type": "string",
 "description": "要列出的目录路径，可用绝对路径或相对工作根目录路径。",
 "required": True,
 }
 ]

 def call(self, params: str, **kwargs) -> str:
 args = json5.loads(params)
 path = _safe_path(args["path"])

 if not path.exists():
 return json.dumps({"ok": False, "error": f"path not found: {str(path)}"}, ensure_ascii=False)
 if not path.is_dir():
 return json.dumps({"ok": False, "error": f"not a directory: {str(path)}"}, ensure_ascii=False)

 items = []
 for x in sorted(path.iterdir(), key=lambda z: (not z.is_dir(), z.name.lower())):
 items.append(
 {
 "name": x.name,
 "path": str(x),
 "type": "dir" if x.is_dir() else "file",
 "size": None if x.is_dir() else x.stat().st_size,
 }
 )

 return json.dumps(
 {
 "ok": True,
 "root": str(AGENT_ROOT),
 "path": str(path),
 "items": items,
 },
 ensure_ascii=False,
 )


@register_tool("read_text_file")
class ReadTextFileTool(BaseTool):
 description = "读取 UTF-8 文本文件内容。只能访问允许的工作目录。"
 parameters = [
 {
 "name": "path",
 "type": "string",
 "description": "要读取的文本文件路径。",
 "required": True,
 }
 ]

 def call(self, params: str, **kwargs) -> str:
 args = json5.loads(params)
 path = _safe_path(args["path"])

 if not path.exists():
 return json.dumps({"ok": False, "error": f"file not found: {str(path)}"}, ensure_ascii=False)
 if not path.is_file():
 return json.dumps({"ok": False, "error": f"not a file: {str(path)}"}, ensure_ascii=False)

 try:
 content = path.read_text(encoding="utf-8")
 except Exception as e:
 return json.dumps({"ok": False, "error": f"read failed: {str(e)}"}, ensure_ascii=False)

 return json.dumps(
 {
 "ok": True,
 "path": str(path),
 "content": content,
 },
 ensure_ascii=False,
 )


@register_tool("write_text_file")
class WriteTextFileTool(BaseTool):
 description = "写入 UTF-8 文本文件。只能写入允许的工作目录。"
 parameters = [
 {
 "name": "path",
 "type": "string",
 "description": "目标文件路径。",
 "required": True,
 },
 {
 "name": "content",
 "type": "string",
 "description": "要写入的文本内容。",
 "required": True,
 },
 ]

 def call(self, params: str, **kwargs) -> str:
 args = json5.loads(params)
 path = _safe_path(args["path"])
 content = args["content"]

 try:
 path.parent.mkdir(parents=True, exist_ok=True)
 path.write_text(content, encoding="utf-8")
 except Exception as e:
 return json.dumps({"ok": False, "error": f"write failed: {str(e)}"}, ensure_ascii=False)

 return json.dumps(
 {
 "ok": True,
 "path": str(path),
 "bytes": len(content.encode("utf-8")),
 },
 ensure_ascii=False,
 )


def build_llm_cfg() -> dict:
 generate_cfg = {
 "temperature": TEMPERATURE,
 "top_p": TOP_P,
 "max_tokens": MAX_TOKENS,
 "extra_body": {
 "top_k": TOP_K
 },
 }

 if USE_RAW_API:
 generate_cfg["use_raw_api"] = True

 if THOUGHT_IN_CONTENT:
 generate_cfg["thought_in_content"] = True

 # 官方示例里，对自托管 vLLM/SGLang 的 thinking 参数是经 extra_body.chat_template_kwargs 传的。
 if ENABLE_THINKING:
 generate_cfg.setdefault("extra_body", {})
 generate_cfg["extra_body"]["chat_template_kwargs"] = {"enable_thinking": True}

 return {
 "model": MODEL_NAME,
 "model_type": "qwenvl_oai",
 "model_server": MODEL_SERVER,
 "api_key": API_KEY,
 "generate_cfg": generate_cfg,
 }


def build_agent() -> Assistant:
 system_message = f"""
你不是普通聊天助手，而是一个本地离线任务代理。

工作要求：
1. 优先完成任务，不要空谈。
2. 涉及本地文件、目录、已有结果时，优先调用工具，不要猜测。
3. 多步骤任务先规划，再执行，再总结。
4. 最终输出尽量包含：已执行步骤、关键依据、产物路径。
5. 只允许访问以下工作根目录及其子目录：
{AGENT_ROOT}
"""

 tools = [
 "list_dir",
 "read_text_file",
 "write_text_file",
 ]

 return Assistant(
 llm=build_llm_cfg(),
 function_list=tools,
 name="QwQ Local Agent",
 description="Qwen-Agent GUI+API adapter on top of existing vLLM for QwQ-32B",
 system_message=system_message,
 )


BOT = build_agent()
APP = FastAPI(title="Qwen-Agent GUI+API Adapter", version="0.2.0")


class ChatMessage(BaseModel):
 role: str
 content: Any
 name: Optional[str] = None


class RunRequest(BaseModel):
 query: Optional[str] = None
 history: List[ChatMessage] = Field(default_factory=list)


class OpenAIChatRequest(BaseModel):
 model: Optional[str] = None
 messages: List[ChatMessage]
 stream: bool = False
 tools: Optional[Any] = None
 tool_choice: Optional[Any] = None
 temperature: Optional[float] = None
 top_p: Optional[float] = None
 max_tokens: Optional[int] = None


def normalize_msg(msg: Any) -> dict:
 if isinstance(msg, dict):
 return msg

 data = {}
 for key in ("role", "name", "content"):
 val = getattr(msg, key, None)
 if val is not None:
 data[key] = val

 extra = getattr(msg, "extra", None)
 if extra:
 data["extra"] = extra
 return data


def content_to_text(content: Any) -> str:
 if content is None:
 return ""
 if isinstance(content, str):
 return content
 if isinstance(content, list):
 parts = []
 for item in content:
 if isinstance(item, str):
 parts.append(item)
 elif isinstance(item, dict):
 if "text" in item:
 parts.append(str(item["text"]))
 elif "content" in item:
 parts.append(str(item["content"]))
 else:
 parts.append(json.dumps(item, ensure_ascii=False))
 else:
 parts.append(str(item))
 return "\n".join(parts)
 return str(content)


def run_agent_once(messages: List[dict]) -> dict:
 last_batch = []
 for batch in BOT.run(messages=messages):
 last_batch = batch

 normalized = [normalize_msg(m) for m in last_batch]

 assistant_texts = []
 tool_messages = []
 for m in normalized:
 role = m.get("role")
 if role == "assistant":
 text = content_to_text(m.get("content"))
 if text:
 assistant_texts.append(text)
 elif role in {"function", "tool"}:
 tool_messages.append(m)

 assistant_text = "\n".join(assistant_texts).strip()

 return {
 "assistant_text": assistant_text,
 "response": normalized,
 "tool_messages": tool_messages,
 }


@APP.get("/healthz")
def healthz():
 return {
 "ok": True,
 "model_server": MODEL_SERVER,
 "model_name": MODEL_NAME,
 "agent_root": str(AGENT_ROOT),
 "gui": {
 "enabled": START_GUI,
 "host": GUI_HOST,
 "port": GUI_PORT,
 },
 "api": {
 "enabled": START_API,
 "host": API_HOST,
 "port": API_PORT,
 },
 "generate_cfg": {
 "temperature": TEMPERATURE,
 "top_p": TOP_P,
 "top_k": TOP_K,
 "max_tokens": MAX_TOKENS,
 "thought_in_content": THOUGHT_IN_CONTENT,
 "use_raw_api": USE_RAW_API,
 "enable_thinking": ENABLE_THINKING,
 },
 }


@APP.get("/v1/models")
def list_models():
 return {
 "object": "list",
 "data": [
 {
 "id": MODEL_NAME,
 "object": "model",
 "owned_by": "local",
 }
 ],
 }


@APP.post("/run")
def run_agent(req: RunRequest):
 messages = [m.model_dump(exclude_none=True) for m in req.history]

 if req.query:
 messages.append({"role": "user", "content": req.query})

 if not messages:
 raise HTTPException(status_code=400, detail="empty request: provide query or history")

 try:
 result = run_agent_once(messages)
 except Exception as e:
 raise HTTPException(status_code=500, detail=f"agent execution failed: {str(e)}")

 return {"ok": True, **result}


@APP.post("/v1/chat/completions")
def openai_chat(req: OpenAIChatRequest):
 if req.stream:
 raise HTTPException(status_code=400, detail="stream=true is not supported in this minimal adapter")

 messages = [m.model_dump(exclude_none=True) for m in req.messages]
 if not messages:
 raise HTTPException(status_code=400, detail="messages is required")

 # 这里显式接收 tools / tool_choice，但不向 vLLM 透传，
 # 目的是阻断上游把 tool_choice:auto 直接打到 vLLM.
 try:
 result = run_agent_once(messages)
 except Exception as e:
 raise HTTPException(status_code=500, detail=f"agent execution failed: {str(e)}")

 content = result["assistant_text"]
 created = int(time.time())
