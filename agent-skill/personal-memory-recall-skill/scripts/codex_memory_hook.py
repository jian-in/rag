#!/usr/bin/env python3
"""Codex native hook bridge for private memory recall and review candidates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from memory_client import ClientError, DEFAULT_ENDPOINT, MemoryClient  # noqa: E402


SECRET_PATTERNS = (
    re.compile(r"(?i)(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
PROFILE_TERMS = (
    "偏好",
    "喜欢",
    "习惯",
    "称呼",
    "我叫",
    "我的名字",
    "工作方式",
    "以后都",
)
EPISODE_TERMS = (
    "继续",
    "上次",
    "之前",
    "以前",
    "当前",
    "进展",
    "项目",
    "决定",
    "计划",
    "我们做",
    "刚才",
)
VAULT_TERMS = (
    "obsidian",
    "secondbrain",
    "笔记",
    "知识库",
    "文档",
    "资料",
    "文献",
)
REMEMBER_TERMS = ("记住", "记一下", "以后请", "请长期", "我的偏好是", "我习惯")
TRIVIAL_PROMPTS = {
    "好",
    "好的",
    "可以",
    "行",
    "嗯",
    "谢谢",
    "？",
    "?",
    "在吗",
    "你好",
    "收到",
}
SELF_CONTAINED_PREFIXES = ("翻译", "把这句翻", "现在几点", "今天几号")
COMPLETION_RE = re.compile(r"已完成|完成了|已经实现|已实现|已修复|已接通|已落地|测试通过|验证通过|已提交")


def _contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def _classify_prompt(prompt: str) -> tuple[list[str], bool, str | None]:
    normalized = prompt.strip()
    lowered = normalized.casefold()
    explicit_scope: str | None = None
    is_explicit = any(term in lowered for term in REMEMBER_TERMS)

    scopes: list[str] = []
    if any(term in lowered for term in PROFILE_TERMS):
        scopes.append("profile")
        if is_explicit:
            explicit_scope = "profile"
    if any(term in lowered for term in EPISODE_TERMS):
        scopes.append("episodes")
    if any(term in lowered for term in VAULT_TERMS):
        scopes.append("vault")
    if is_explicit and explicit_scope is None:
        explicit_scope = "episodes"
        scopes.append("episodes")

    if scopes:
        return list(dict.fromkeys(scopes)), True, explicit_scope
    if normalized in TRIVIAL_PROMPTS:
        return [], False, None
    if any(lowered.startswith(prefix) for prefix in SELF_CONTAINED_PREFIXES):
        return [], False, None
    if len(normalized) < 18:
        return [], False, None
    return ["episodes"], True, None


def _session_key(payload: dict[str, Any]) -> str | None:
    raw = payload.get("session_id") or payload.get("conversation_id") or payload.get("thread_id")
    if not raw:
        return None
    return hashlib.sha256(str(raw).encode("utf-8")).hexdigest()[:32]


def _spool_path(spool_root: Path, payload: dict[str, Any]) -> Path | None:
    key = _session_key(payload)
    return spool_root / f"{key}.json" if key else None


def _save_spool(spool_root: Path, payload: dict[str, Any], prompt: str) -> None:
    path = _spool_path(spool_root, payload)
    if path is None:
        return
    spool_root.mkdir(parents=True, exist_ok=True)
    record = {
        "prompt": prompt[:1200],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=spool_root,
            prefix=f"{path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(record, handle, ensure_ascii=False)
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _take_spool(spool_root: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    path = _spool_path(spool_root, payload)
    if path is None or not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _compact(text: str, limit: int) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _handle_user_prompt(
    payload: dict[str, Any], client: MemoryClient, spool_root: Path
) -> dict[str, Any]:
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt or _contains_secret(prompt):
        return {}
    scopes, is_complex, explicit_scope = _classify_prompt(prompt)
    if not scopes:
        return {}
    session_id = str(payload.get("session_id")) if payload.get("session_id") else None

    if explicit_scope is not None:
        try:
            client.propose(
                explicit_scope,
                prompt,
                "用户明确要求记住，等待本机人工确认。",
                "codex-user-prompt-hook",
                session_id,
            )
        except (ClientError, OSError, ValueError):
            pass
    elif is_complex:
        try:
            _save_spool(spool_root, payload, prompt)
        except OSError:
            pass

    try:
        bundle = client.context(
            _compact(prompt, 600), scopes, 8, 3000, retrieval_mode="local"
        )
    except (ClientError, OSError, ValueError):
        return {}
    context = str(bundle.get("context", "")).strip()
    if not context:
        return {}
    additional = (
        "以下内容来自阿飞本机的私有记忆检索，只作为可核验的背景；"
        "若与当前消息冲突，以当前消息为准。不要把整段内容复述给用户。\n\n"
        + context[:3200]
    )
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": additional,
        }
    }


def _handle_stop(
    payload: dict[str, Any], client: MemoryClient, spool_root: Path
) -> dict[str, Any]:
    record = _take_spool(spool_root, payload)
    if not record:
        return {}
    prompt = str(record.get("prompt", "")).strip()
    assistant = str(payload.get("last_assistant_message", "")).strip()
    if (
        not prompt
        or len(assistant) < 40
        or not COMPLETION_RE.search(assistant)
        or _contains_secret(prompt)
        or _contains_secret(assistant)
    ):
        return {}
    content = (
        "已完成任务候选\n"
        f"任务：{_compact(prompt, 500)}\n"
        f"结果：{_compact(assistant, 2200)}"
    )
    session_id = str(payload.get("session_id")) if payload.get("session_id") else None
    try:
        client.propose(
            "episodes",
            content,
            "复杂任务出现明确完成信号，等待本机人工确认后再沉淀。",
            "codex-stop-hook",
            session_id,
        )
    except (ClientError, OSError, ValueError):
        pass
    return {}


def handle_event(
    payload: dict[str, Any], client: MemoryClient, spool_root: Path
) -> dict[str, Any]:
    """Handle one native Codex hook payload and return a valid hook response."""
    event = str(payload.get("hook_event_name") or payload.get("hookEventName") or "")
    if event == "UserPromptSubmit":
        return _handle_user_prompt(payload, client, spool_root)
    if event == "Stop":
        return _handle_stop(payload, client, spool_root)
    return {}


def _default_spool_root() -> Path:
    local = os.getenv("LOCALAPPDATA")
    if local:
        return Path(local) / "FeiliMemoryHub" / "hook-spool"
    return Path(tempfile.gettempdir()) / "FeiliMemoryHub" / "hook-spool"


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="strict")


def main() -> int:
    _configure_utf8_stdio()
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if not isinstance(payload, dict):
            payload = {}
        endpoint = os.getenv("FEILI_MEMORY_HUB_URL", DEFAULT_ENDPOINT)
        result = handle_event(payload, MemoryClient(endpoint), _default_spool_root())
    except Exception:
        result = {}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
