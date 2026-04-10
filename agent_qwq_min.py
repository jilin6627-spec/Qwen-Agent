import json
import os
from pathlib import Path
from typing import Any, List, Optional

import json5
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from qwen_agent.agents import Assistant
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
PORT = int(os.getenv("PORT", "8080"))

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
TOP_K = int(os.getenv("TOP_K", "20"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))

# 只有在你的 vLLM 返回把思考内容包在 content 里时，再切成 true。
THOUGHT_IN_CONTENT = env_bool("THOUGHT_IN_CONTENT", False)


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
 "description": "要列出的目录路径，可以是绝对路径，也可以是相对工作根目录的路径。",
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
