from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import uvicorn
from dotenv import load_dotenv

from .runtime import create_hub_from_environment
from .server import create_app


ROOT = Path(__file__).resolve().parents[1]
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
STATE_ROOT = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / "FeiliMemoryHub"
PID_PATH = STATE_ROOT / "server.pid"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Feili personal memory hub.")
    parser.add_argument("--host", default=os.getenv("MEMORY_HUB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MEMORY_HUB_PORT", "8771")))
    parser.add_argument("--no-auto-sync", action="store_true")
    parser.add_argument("--sync-only", action="store_true")
    return parser.parse_args()


def _write_status(message: str) -> None:
    """Write a status line when a console stream exists (pythonw has none)."""
    if sys.stderr is not None:
        print(message, file=sys.stderr, flush=True)


def _write_pid(path: Path, process_id: int) -> None:
    """Atomically record the managed server process identifier."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(str(process_id), encoding="ascii")
    os.replace(temporary, path)


def _remove_own_pid(path: Path, process_id: int) -> None:
    """Remove a PID file only when it still belongs to this process."""
    try:
        recorded = path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return
    if recorded == str(process_id):
        path.unlink(missing_ok=True)


def _service_is_running(host: str, port: int) -> bool:
    """Return whether the expected Memory Hub already answers on this endpoint."""
    display_host = f"[{host}]" if ":" in host else host
    try:
        with urlopen(f"http://{display_host}:{port}/health", timeout=0.75) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return False
    return payload.get("ok") is True and payload.get("server") == "feili-personal-memory"


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = _arguments()
    if args.host not in LOOPBACK_HOSTS:
        raise ValueError("personal memory hub may only listen on a loopback host")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if not args.sync_only and _service_is_running(args.host, args.port):
        return 0

    hub = create_hub_from_environment()
    if not args.no_auto_sync:
        try:
            report = hub.sync()
            counts = {
                scope: details["chunks"]
                for scope, details in report["scopes"].items()
            }
            _write_status(f"Memory indexes synchronized: {counts}")
        except Exception as exc:
            _write_status(f"Memory auto-sync warning: {exc}")
    if args.sync_only:
        return 0

    app = create_app(hub)
    process_id = os.getpid()
    _write_pid(PID_PATH, process_id)
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="warning",
            log_config=None,
            access_log=False,
        )
    finally:
        _remove_own_pid(PID_PATH, process_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
