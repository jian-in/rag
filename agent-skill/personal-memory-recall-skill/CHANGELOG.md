# Changelog

## 1.2.0 - 2026-08-20

- Added explicit `local` and `hybrid` retrieval modes across Memory Hub interfaces.
- Switched native prompt hooks to local-only recall so they never call remote models.
- Kept MCP, dashboard, and client deep retrieval hybrid by default.

## 1.1.0 - 2026-08-20

- Added Codex native prompt and stop hooks for automatic local recall.
- Added review-only candidate proposals with secret filtering and ephemeral turn spooling.
- Documented the human approval, correction, rejection, and precise-forget boundary.

## 1.0.0 - 2026-08-20

- Added automatic and explicit activation guidance for personal continuity tasks.
- Added MCP-first retrieval workflow across three isolated memory scopes.
- Added loopback-only HTTP fallback client with structured errors and diagnostics.
- Added cross-platform installer support, tests, privacy rules, and troubleshooting.
