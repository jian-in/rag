# MCP Setup

## Summary

This project now provides two MCP entrypoints:

- `mcp_http_server.py`: recommended, more compatible with desktop clients
- `mcp_server.py`: stdio fallback

For current Codex desktop clients, use streamable HTTP at `/mcp`; older SSE-only
clients should use `/sse` instead.

## Recommended Setup

### 1. Start the HTTP MCP server

Run this in PowerShell and keep the window open:

```powershell
$env:MCP_HTTP_PORT="8766"
& "E:\Backup\rag-knowledge-base\RAG\Scripts\python.exe" -u "E:\Backup\rag-knowledge-base\mcp_http_server.py"
```

If `8766` is already occupied, change it to another free port such as `8767`.

### 2. Configure the MCP client

For current Codex desktop clients, use streamable HTTP:

```text
rag-kb=http://127.0.0.1:8766/mcp
```

For an older client that explicitly uses an SSE handshake, use:

```text
rag-kb=http://127.0.0.1:8766/sse
```

Do not point a streamable HTTP client at `/sse`: it will send POST requests
without `sessionId` and receive HTTP 422.

## Available Tools

- `get_status`: show index and model status
- `sync_knowledge_base`: mirror selected SecondBrain folders and incrementally synchronize files, vectors, and BM25
- `reload_knowledge_base`: reload Chroma and rebuild BM25 from `data/`
- `search_knowledge`: return relevant chunks
- `ask_knowledge`: answer with citations from the local knowledge base

## Add or Update SecondBrain Notes

The default source is `C:\Desktop\SecondBrain`. The sync tool watches these folders:

- `02-Areas`
- `03-Resources`

Put or edit supported files there (`.md`, `.txt`, `.pdf`, `.docx`), then call the MCP tool
`sync_knowledge_base`. It mirrors the selected folders into `data/`, calculates file
fingerprints, and only embeds added or changed files. Deleted source files have their
mirrored copy and vectors removed. A parse or copy failure keeps the previous vector
instead of deleting it.

`00-Inbox` is intentionally excluded by default so unfinished captures do not enter
the answerable knowledge base. To include it, set:

```powershell
$env:SECONDBRAIN_INCLUDE_DIRS="02-Areas,03-Resources,00-Inbox"
```

The sync manifests are `.secondbrain_sync_manifest.json` and
`.rag_vector_manifest.json`; they are local runtime state and should not be edited by hand.

## Stdio Fallback

If you still want to try stdio, use:

```text
rag-kb="E:\Backup\rag-knowledge-base\RAG\Scripts\python.exe" -u "E:\Backup\rag-knowledge-base\mcp_server.py"
```

But in practice, stdio was less stable with this desktop client.

## Troubleshooting

### `MCP request initialize timed out after 60000ms`

This usually means the client and the stdio transport did not complete the handshake.

Recommended fix:

1. Stop using stdio for this client.
2. Start `mcp_http_server.py` instead.
3. Configure the client with `http://127.0.0.1:8766/mcp`.

### `405 Method Not Allowed`

This usually means the client is doing an SSE handshake, but the URL was set to `/mcp`.

Fix:

- If the client shows `SSE handshake`, use `/sse`, not `/mcp`.

Correct example:

```text
rag-kb=http://127.0.0.1:8766/sse
```

### `error while attempting to bind ... 10048`

This means the port is already occupied by another process.

Fix options:

1. Kill the old process.
2. Or switch to another port.

Example with another port:

```powershell
$env:MCP_HTTP_PORT="8767"
& "E:\Backup\rag-knowledge-base\RAG\Scripts\python.exe" -u "E:\Backup\rag-knowledge-base\mcp_http_server.py"
```

Then update the client URL accordingly:

```text
rag-kb=http://127.0.0.1:8767/sse
```

### Manual Local Checks

Check whether the HTTP MCP server is alive:

```text
http://127.0.0.1:8766/health
```

A healthy response looks like:

```json
{"ok": true, "server": "rag-knowledge-base", "version": "1.0.0", "protocolVersion": "2024-11-05"}
```

## Lessons Learned

- MCP transport compatibility matters as much as tool logic.
- Current Codex desktop clients use streamable HTTP at `/mcp`.
- If a client explicitly says `SSE handshake`, use `/sse` for that client.
- Always separate these two questions first:
  - Did the server start?
  - Did the client and server agree on the transport format?
- Keep a health endpoint when exposing local MCP over HTTP. It makes debugging much easier.

## Notes

- Build the knowledge base before using MCP, so `chroma_db/` already exists.
- The server reload path rebuilds BM25 and vector retrieval from the final Chroma collection.
- The MCP servers reuse the existing RAG stack, including hybrid retrieval and reranking.
