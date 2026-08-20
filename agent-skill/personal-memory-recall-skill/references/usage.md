# Usage and troubleshooting

This skill is a discovery and retrieval layer for Feili Personal Memory Hub. The source
files remain in Afei's Obsidian vault and Codex memory directories; the skill does not
copy them. The service builds three independent SQLite indexes and exposes one stable
loopback endpoint.

## Choosing a scope

Use `profile` for stable facts: preferred interface style, name, recurring workflow,
assistant identity, and durable boundaries. Use `episodes` for state that changes over
time: an unfinished project, a decision from a previous run, a recent correction, or a
handoff. Use `vault` for authored knowledge in Obsidian: project notes, research,
reference material, and source locators.

Start narrow. For “What kind of interface do I like?”, query only `profile`. For
“Continue the memory system we were building”, start with `episodes`, then add `vault`
if project notes are needed. Narrow scope selection improves relevance and preserves the
three-domain isolation contract.

## Query construction

A good query names the subject and the fact class. Prefer `SecondBrain RAG current
decision and next step` over copying an entire conversation. Prefer `interface visual
preference dark blue` over `what do I like?`. Queries can be Chinese or English.

Never include credentials, access tokens, private keys, verification codes, payment
details, or unrelated personal text. The local lexical engine does not need them, and
an optional external embedding provider may be configured later by the service owner.

## MCP-first operation

Agents with MCP support should call `memory_context`. It returns a compact prompt-ready
string plus structured hits. `memory_search` is useful when the Agent needs to inspect
several locators before deciding what to include. `memory_status` checks readiness, and
`memory_sync` refreshes one or more indexes.

The server advertises its entry document at:

```text
http://127.0.0.1:8771/.well-known/agent-memory.json
```

The MCP endpoint is:

```text
http://127.0.0.1:8771/mcp
```

## HTTP fallback examples

Inspect readiness:

```bash
python scripts/memory_client.py check-prereqs
python scripts/memory_client.py status
```

Retrieve a preference:

```bash
python scripts/memory_client.py context "我喜欢什么样的界面" \
  --scopes profile --top-k 5 --char-budget 3000
```

Continue a project:

```bash
python scripts/memory_client.py context "SecondBrain RAG 当前进展与下一步" \
  --scopes episodes vault --top-k 8 --char-budget 6000
```

Refresh only changed source domains:

```bash
python scripts/memory_client.py sync --scopes profile episodes
```

## Interpreting results

Each hit includes `scope`, `source_path`, `title`, `heading`, `preview`, `score`, and
`locator`. Vault hits can also include an `obsidian_uri`. The locator is evidence for
the Agent; it does not need to be shown in every conversational answer. When the user
asks what is remembered or requests provenance, include compact source labels.

A memory can be stale. If the user's current message contradicts a hit, the current
message wins. Stable profile information should be updated at its source rather than
patched into the index; the next incremental sync will detect the change.

## Troubleshooting

### Cannot connect to the Memory Hub

Symptom: the client returns a `network` error. Confirm that the Windows startup shortcut
exists and that `http://127.0.0.1:8771/health` responds. Start the service from its
repository with `python -m memory_hub`, then rerun `check-prereqs`.

### Port 8771 belongs to another process

Symptom: health verification says the endpoint is not `feili-personal-memory`. Do not
send requests to that process. Stop the unrelated process or configure a new loopback
port consistently in the service, Codex MCP entry, desktop shortcut, and skill config.

### The result is empty

Confirm the correct scope first. Then run `status` and inspect its chunk counts. If a
source file was recently added or changed, run `sync` for that scope. Broadening from
one scope to all scopes should be a deliberate second step, not the default response to
every empty result.

### Results contain common-word noise

Use a concrete noun or project name in the query. The service includes Chinese phrase
tokenization and query noise filtering, but a query such as “what was that thing?” still
lacks a discriminating concept. Add the object: interface, memory hub, Obsidian, or the
project name.

### Optional vectors are disabled

This is the privacy-preserving default. Local lexical retrieval remains functional.
Dedicated SiliconFlow keys or explicit provider opt-in must be configured in the Memory
Hub environment; this skill never enables upload or reads key values.

### Obsidian link does not open

Confirm the vault is named `SecondBrain` and the note remains under an indexed directory.
The result still includes a filesystem locator, so the content is retrievable even when
the Obsidian URI handler is not registered.

## Privacy boundary

The client accepts only `http` endpoints whose hostname is exactly `127.0.0.1`,
`localhost`, or `::1`. It rejects credentials, arbitrary paths, query strings, and
fragments in the base URL. This prevents a copied configuration from silently sending
personal retrieval requests to an external host.
