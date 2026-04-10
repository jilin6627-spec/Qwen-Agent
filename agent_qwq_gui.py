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
