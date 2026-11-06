"""Shared request guards for the local HTTP services.

The RAG web/API (7860), the RAG MCP HTTP server (8766) and the personal memory
hub (8771) all serve private data from this machine. Two browser-borne attacks
apply to every one of them, even when bound to loopback:

* DNS rebinding: an attacker's domain resolves to 127.0.0.1, so the browser
  talks to the local service while sending the attacker's ``Host`` header.
  Checking ``Host`` (not ``Origin``, which same-origin requests omit) stops it.
* CSRF via "simple" requests: ``text/plain``, form and multipart POSTs skip the
  CORS preflight, so any web page can fire them. Requiring
  ``application/json`` (or a custom header) forces a preflight, and none of
  these services grant CORS, so the browser never sends the real request.
"""

from __future__ import annotations

import hmac
from typing import Iterable, Optional
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.responses import JSONResponse


LOOPBACK_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1"})
BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def request_hostname(request: Request) -> str:
    """Return the lower-cased hostname of the ``Host`` header, without port."""
    # Host 形如 "127.0.0.1:8771" 或 "[::1]:8771"，补上 "//" 才能按 netloc 解析
    try:
        return (urlsplit("//" + request.headers.get("host", "")).hostname or "").lower()
    except ValueError:
        return ""


def is_json_request(request: Request) -> bool:
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    return media_type == "application/json"


def is_loopback_host(host: str) -> bool:
    return host.strip().strip("[]").lower() in LOOPBACK_HOSTNAMES


def parse_hostnames(raw: Optional[str]) -> frozenset[str]:
    """Parse a comma-separated hostname list such as ``"192.168.1.20,nas.lan"``."""
    if not raw:
        return frozenset()
    return frozenset(item.strip().strip("[]").lower() for item in raw.split(",") if item.strip())


def trusted_hostnames(extra: Iterable[str] = ()) -> frozenset[str]:
    return LOOPBACK_HOSTNAMES | frozenset(item.strip().strip("[]").lower() for item in extra)


def forbidden_host_response(request: Request, trusted: frozenset[str]) -> Optional[JSONResponse]:
    if request_hostname(request) in trusted:
        return None
    return JSONResponse(status_code=403, content={"detail": "forbidden: untrusted Host header"})


def json_required_response(request: Request) -> Optional[JSONResponse]:
    if request.method not in BODY_METHODS or is_json_request(request):
        return None
    return JSONResponse(
        status_code=415,
        content={"detail": "Content-Type must be application/json"},
    )


# Content types a browser may send cross-site without a CORS preflight. A body
# without any Content-Type (e.g. an untyped Blob) is also sent without one.
CORS_SAFELISTED_CONTENT_TYPES = frozenset(
    {"", "text/plain", "application/x-www-form-urlencoded", "multipart/form-data"}
)


def preflight_exempt_body_response(request: Request) -> Optional[JSONResponse]:
    """Reject writes whose Content-Type lets a browser skip the CORS preflight.

    More lenient than :func:`json_required_response` for protocol endpoints
    whose third-party clients may label JSON bodies differently.
    """
    if request.method not in BODY_METHODS:
        return None
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type not in CORS_SAFELISTED_CONTENT_TYPES:
        return None
    return JSONResponse(
        status_code=415,
        content={"detail": "Content-Type must be application/json"},
    )


def custom_header_required_response(request: Request, header: str) -> Optional[JSONResponse]:
    """Require a non-simple header on writes whose body cannot be JSON (multipart uploads).

    Any custom header forces a CORS preflight, which these services never grant.
    """
    if request.method not in BODY_METHODS or request.headers.get(header):
        return None
    return JSONResponse(status_code=403, content={"detail": f"missing required header: {header}"})


def bearer_token_response(request: Request, token: Optional[str]) -> Optional[JSONResponse]:
    """Reject the request unless it carries ``Authorization: Bearer <token>``.

    A falsy ``token`` disables the check (loopback-only default deployments).
    """
    if not token:
        return None
    header = request.headers.get("authorization", "")
    scheme, _, supplied = header.partition(" ")
    if scheme.lower() == "bearer" and hmac.compare_digest(
        supplied.strip().encode("utf-8"), token.encode("utf-8")
    ):
        return None
    return JSONResponse(
        status_code=401,
        content={"detail": "missing or invalid bearer token"},
        headers={"WWW-Authenticate": "Bearer"},
    )
