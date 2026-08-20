# Personal Memory Hub

## Goal

Build a local-first memory layer for Afei's agents. It keeps three memory domains physically isolated while exposing one small retrieval interface that any agent can use.

## Memory domains

1. `vault` — selected Markdown notes from the Obsidian vault. The indexer is read-only; approved Agent candidates may enter only the managed `00-Inbox/Agent Memory Inbox.md` note.
2. `profile` — curated user and assistant identity pages (`USER_PROFILE.md` and `FEILI_SOUL.md`).
3. `episodes` — current project context, memory log, and curated rollout summaries. Raw chat transcripts, secrets, backups, and dynamic dream material are excluded by default.

Each domain has its own SQLite index file. A federated search queries the requested domains independently and fuses results without merging their storage.

## Public interface

The `MemoryHub` module exposes retrieval plus a human-reviewed write path:

- `sync(scopes=None)` — incrementally synchronize selected domains.
- `search(query, scopes=None, top_k=8)` — return ranked, cited memory hits.
- `context(query, scopes=None, top_k=8, char_budget=6000)` — return an agent-ready context bundle.
- `status()` — report source paths, index counts, model configuration, and last sync state without exposing credentials.
- `propose_memory(...)` — create a deduplicated pending candidate without changing sources.
- `review_memory(...)` — let the local human interface update, approve, reject, or precisely forget a managed candidate.

Callers do not manage files, chunks, databases, embeddings, or reranking directly.

## Retrieval behavior

- Local lexical retrieval is always available and must work without an API key.
- When a SiliconFlow-compatible embedding key is configured, sync fills missing vectors incrementally.
- Vector and lexical candidates are fused with reciprocal-rank fusion.
- When a reranker key is configured, the fused candidate set is reranked before returning the final results.
- Every hit includes its domain, source path, heading, preview, stable chunk ID, and an Obsidian link when applicable.
- Context output groups evidence by domain and never exceeds the requested character budget.

## Privacy and safety

- Bind HTTP and MCP transports to loopback only.
- Never return or log API keys.
- Use explicit source allowlists; do not scan arbitrary home folders.
- Ignore `.env`, credentials, backups, raw Codex sessions, Obsidian internals, attachments, and archives by default.
- Indexes are derived local state and may be rebuilt from source files.
- Review state lives in a separate control-plane SQLite database and is never searched as a fourth memory domain.
- Agent-facing tools can propose but cannot approve, reject, edit, or forget.

## Agent entry

Expose the same hub through a local streamable HTTP MCP endpoint with these tools:

- `memory_status`
- `memory_sync`
- `memory_search`
- `memory_context`
- `memory_propose`

Also expose a local dashboard for humans and a well-known JSON manifest so other agent hosts can discover the endpoint and scope rules.

## Acceptance criteria

1. Scope-restricted search cannot return hits from another memory domain.
2. Repeated sync is idempotent; edits and deletions update only the affected domain.
3. Search and context work with no external API key.
4. Optional embedding and reranking adapters can be tested without network access.
5. Obsidian hits contain a working vault-relative locator and source file remains unchanged.
6. MCP tools and dashboard use the same `MemoryHub` interface.
7. The server starts on loopback, reports healthy state, and provides no secret values.
8. Candidate approval writes one identifiable managed block; forgetting removes only that block.
9. Codex hooks skip trivial and secret-like prompts and never directly mutate memory sources.

## Local operation

The default service endpoint is `http://127.0.0.1:8771`; its MCP transport is `/mcp`
and its discovery document is `/.well-known/agent-memory.json`.

```powershell
# Run in the foreground
.\scripts\launch-memory-hub.ps1

# Rebuild or incrementally refresh all three scopes, then exit
.\scripts\launch-memory-hub.ps1 -SyncOnly

# Install Windows sign-in startup and the independent desktop window shortcut
.\scripts\install-memory-hub.ps1

# Stop only the verified managed process
.\scripts\stop-memory-hub.ps1
```

The runtime records its managed process ID under
`%LOCALAPPDATA%\FeiliMemoryHub\server.pid`. The three derived databases live under
`%LOCALAPPDATA%\FeiliMemoryHub\indexes` unless `MEMORY_HUB_INDEX_ROOT` overrides the
location.

Windows installation creates:

- Startup shortcut `FeiliMemoryHub.lnk`, targeting the repository's `pythonw.exe`.
- Desktop shortcut `绯梨记忆观测台.lnk`, opening Edge in app-window mode.

## Agent installation

The self-contained package at `agent-skill/personal-memory-recall-skill` provides MCP-
first instructions and a loopback-only HTTP fallback client. Install it in the universal
Agent Skills directory, then register the MCP endpoint with Codex:

```powershell
Copy-Item .\agent-skill\personal-memory-recall-skill `
  "$HOME\.agents\skills\personal-memory-recall-skill" -Recurse
codex mcp add personal_memory --url http://127.0.0.1:8771/mcp
```

Restart the Agent host after changing MCP configuration. Agents should call
`memory_context` before asking Afei to repeat known preferences or active project
decisions.

Codex native hooks can call the installed skill's `codex_memory_hook.py` at
`UserPromptSubmit` and `Stop`. The first injects compact retrieved context; the second
may submit a completed-task candidate. Codex requires a one-time review of the exact
command hook after installation.

## Optional SiliconFlow providers

External semantic retrieval is disabled by default. A shared `SILICONFLOW_API_KEY` does
not authorize personal-memory upload. Enable a provider only with a dedicated memory key
or an explicit opt-in flag in the repository `.env`:

```dotenv
MEMORY_VECTOR_ENABLED=true
MEMORY_EMBEDDING_API_KEY=...
MEMORY_RERANK_ENABLED=true
MEMORY_RERANK_API_KEY=...
```

Never commit `.env` or paste keys into Agent prompts. Local lexical retrieval remains
fully usable when both providers are disabled.
