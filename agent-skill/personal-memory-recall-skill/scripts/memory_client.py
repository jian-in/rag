#!/usr/bin/env python3
"""Loopback-only client for Feili Personal Memory Hub.

The client gives Agents a portable fallback when MCP tools are unavailable. It uses only
Python's standard library, validates every user-facing input, and emits JSON on success
or structured JSON errors on stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


SKILL_NAME = "personal-memory-recall-skill"
SKILL_VERSION = "1.0.0"
SCOPES = ("vault", "profile", "episodes")
DEFAULT_ENDPOINT = "http://127.0.0.1:8771"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "assets" / "config.json"


class ClientError(RuntimeError):
    """A safe, user-facing client failure.

    Attributes:
        error_type: Stable category used by Agent callers.
        hint: Concrete recovery guidance without secrets.
    """

    def __init__(self, message: str, error_type: str, hint: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.hint = hint


def _load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the bundled client configuration.

    Args:
        path: JSON configuration path.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ClientError: If the file is absent, malformed, or not a JSON object.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ClientError(
            f"Configuration file not found: {path}",
            "runtime",
            "Reinstall the complete skill directory, including assets/config.json.",
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ClientError(
            f"Configuration file is unreadable: {path}",
            "runtime",
            "Validate assets/config.json and retry.",
        ) from exc
    if not isinstance(payload, dict):
        raise ClientError(
            "Configuration root must be a JSON object.",
            "validation",
            "Restore the bundled assets/config.json file.",
        )
    return payload


def _validate_endpoint(endpoint: str) -> str:
    """Validate and normalize a loopback HTTP endpoint.

    Args:
        endpoint: Candidate base URL.

    Returns:
        Endpoint without a trailing slash.

    Raises:
        ClientError: If scheme, host, credentials, path, query, or fragment is unsafe.
    """
    parsed = urlparse(endpoint)
    if parsed.scheme != "http":
        raise ClientError("Endpoint must use http.", "validation", "Use the local HTTP endpoint.")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ClientError(
            "Endpoint must resolve explicitly to a loopback host.",
            "validation",
            "Use http://127.0.0.1:8771.",
        )
    if parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ClientError(
            "Endpoint must be a credential-free base URL without path, query, or fragment.",
            "validation",
            "Use http://127.0.0.1:8771.",
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ClientError("Endpoint port is invalid.", "validation", "Use port 8771.") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ClientError("Endpoint port is invalid.", "validation", "Use port 8771.")
    return endpoint.rstrip("/")


def _validate_query(query: str) -> str:
    """Return a safe retrieval query or raise a structured validation error."""
    normalized = query.strip()
    if not normalized:
        raise ClientError("Query cannot be empty.", "validation", "Provide a short topic query.")
    if len(normalized) > 4000:
        raise ClientError(
            "Query exceeds 4000 characters.",
            "validation",
            "Reduce it to the project, preference, or decision being recalled.",
        )
    return normalized


def _validate_scopes(scopes: Sequence[str] | None) -> list[str] | None:
    """Validate isolated memory scope names while preserving order."""
    if scopes is None:
        return None
    selected = list(dict.fromkeys(scopes))
    unknown = sorted(set(selected) - set(SCOPES))
    if unknown:
        raise ClientError(
            f"Unknown memory scope: {', '.join(unknown)}",
            "validation",
            f"Choose from: {', '.join(SCOPES)}.",
        )
    return selected


class MemoryClient:
    """Client for the local Memory Hub HTTP API.

    Example:
        >>> client = MemoryClient("http://127.0.0.1:8771")
        >>> client.status()["ready"] in (True, False)
        True
    """

    def __init__(self, endpoint: str, timeout: float = 20.0) -> None:
        self.endpoint = _validate_endpoint(endpoint)
        if not 0.1 <= timeout <= 300:
            raise ClientError("Timeout is out of range.", "validation", "Use 0.1-300 seconds.")
        self.timeout = timeout

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send one local request and validate its JSON object response."""
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.endpoint}{path}",
            data=body,
            headers={"Content-Type": "application/json"} if body is not None else {},
            method="POST" if body is not None else "GET",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise ClientError(
                f"Memory Hub returned HTTP {exc.code}.",
                "runtime",
                "Check the command arguments and the local service log.",
            ) from exc
        except URLError as exc:
            raise ClientError(
                "Cannot connect to Feili Personal Memory Hub.",
                "network",
                "Start the local Memory Hub, then retry check-prereqs.",
            ) from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ClientError(
                "Memory Hub returned an unreadable response.",
                "runtime",
                "Restart the local Memory Hub and retry.",
            ) from exc
        if not isinstance(result, dict):
            raise ClientError(
                "Memory Hub response must be a JSON object.",
                "runtime",
                "Restart or update the local Memory Hub.",
            )
        return result

    def health(self) -> dict[str, Any]:
        """Return the service identity and health payload."""
        result = self._request("/health")
        if result.get("ok") is not True or result.get("server") != "feili-personal-memory":
            raise ClientError(
                "The loopback endpoint is not Feili Personal Memory Hub.",
                "runtime",
                "Check whether another process is using port 8771.",
            )
        return result

    def status(self) -> dict[str, Any]:
        """Inspect isolated index counts and optional provider readiness."""
        return self._request("/api/status")

    def search(self, query: str, scopes: Sequence[str] | None, top_k: int) -> dict[str, Any]:
        """Return ranked memory hits with source locators."""
        if not 1 <= top_k <= 50:
            raise ClientError("top-k must be 1-50.", "validation", "Use the default value 8.")
        payload: dict[str, Any] = {"query": _validate_query(query), "top_k": top_k}
        selected = _validate_scopes(scopes)
        if selected is not None:
            payload["scopes"] = selected
        result = self._request("/api/search", payload)
        if result.get("result_count") == 0:
            result["_warnings"] = ["No relevant memory hit was found in the selected scopes."]
        return result

    def context(
        self,
        query: str,
        scopes: Sequence[str] | None,
        top_k: int,
        char_budget: int,
    ) -> dict[str, Any]:
        """Build a compact cited context bundle for an Agent prompt."""
        if not 1 <= top_k <= 50:
            raise ClientError("top-k must be 1-50.", "validation", "Use the default value 8.")
        if not 200 <= char_budget <= 30000:
            raise ClientError(
                "char-budget must be 200-30000.", "validation", "Use the default value 6000."
            )
        payload: dict[str, Any] = {
            "query": _validate_query(query),
            "top_k": top_k,
            "char_budget": char_budget,
        }
        selected = _validate_scopes(scopes)
        if selected is not None:
            payload["scopes"] = selected
        result = self._request("/api/context", payload)
        if not result.get("hits"):
            result["_warnings"] = ["No relevant memory hit was found in the selected scopes."]
        return result

    def sync(self, scopes: Sequence[str] | None) -> dict[str, Any]:
        """Incrementally refresh selected physically isolated indexes."""
        selected = _validate_scopes(scopes)
        return self._request("/api/sync", {"scopes": selected} if selected is not None else {})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve Afei's local personal memories.")
    parser.add_argument(
        "--endpoint",
        default=os.getenv("FEILI_MEMORY_HUB_URL"),
        help="Loopback Memory Hub base URL; defaults to assets/config.json.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show isolated index status.")
    commands.add_parser("check-prereqs", help="Check runtime and local service readiness.")
    commands.add_parser("diagnostics", help="Show non-secret skill capabilities.")

    for name in ("search", "context"):
        command = commands.add_parser(name, help=f"Run a memory {name} request.")
        command.add_argument("query")
        command.add_argument("--scopes", nargs="+", choices=SCOPES)
        command.add_argument("--top-k", type=int, default=8)
        if name == "context":
            command.add_argument("--char-budget", type=int, default=6000)

    sync = commands.add_parser("sync", help="Refresh isolated local indexes.")
    sync.add_argument("--scopes", nargs="+", choices=SCOPES)
    return parser


def _prerequisites(client: MemoryClient) -> dict[str, Any]:
    checks: list[dict[str, Any]] = [
        {
            "check": "python",
            "required": ">=3.10",
            "found": platform.python_version(),
            "ok": sys.version_info >= (3, 10),
        },
        {
            "check": "config",
            "required": str(CONFIG_PATH),
            "found": "present" if CONFIG_PATH.is_file() else "missing",
            "ok": CONFIG_PATH.is_file(),
        },
        {
            "check": "endpoint",
            "required": "loopback HTTP",
            "found": client.endpoint,
            "ok": True,
        },
    ]
    try:
        health = client.health()
        checks.append(
            {
                "check": "service",
                "required": "feili-personal-memory",
                "found": health.get("server"),
                "ok": True,
            }
        )
    except ClientError as exc:
        checks.append(
            {
                "check": "service",
                "required": "feili-personal-memory",
                "found": str(exc),
                "ok": False,
            }
        )
    return {"ready": all(item["ok"] for item in checks), "checks": checks}


def _diagnostics(client: MemoryClient) -> dict[str, Any]:
    return {
        "skill": SKILL_NAME,
        "version": SKILL_VERSION,
        "harness_level": "loopback-client",
        "endpoint": client.endpoint,
        "commands": ["status", "search", "context", "sync", "check-prereqs", "diagnostics"],
        "scopes": list(SCOPES),
        "harness_features": {
            "input_validation": True,
            "output_sanity": True,
            "structured_errors": True,
            "loopback_enforcement": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, execute one operation, and emit JSON."""
    try:
        args = _parser().parse_args(argv)
        config = _load_config()
        endpoint = args.endpoint or str(config.get("endpoint", DEFAULT_ENDPOINT))
        client = MemoryClient(endpoint, timeout=180.0 if args.command == "sync" else 20.0)

        if args.command == "status":
            result = client.status()
        elif args.command == "search":
            result = client.search(args.query, args.scopes, args.top_k)
        elif args.command == "context":
            result = client.context(args.query, args.scopes, args.top_k, args.char_budget)
        elif args.command == "sync":
            result = client.sync(args.scopes)
        elif args.command == "check-prereqs":
            result = _prerequisites(client)
        else:
            result = _diagnostics(client)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except ClientError as exc:
        error = {"error": str(exc), "error_type": exc.error_type, "hint": exc.hint}
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
