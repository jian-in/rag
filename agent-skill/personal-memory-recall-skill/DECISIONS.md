# Architecture decisions

## Simple skill, not a suite

The package has one domain and two tightly coupled operations: retrieve context and
refresh indexes. A single `SKILL.md` plus one portable client is smaller and easier for
Agents to discover than a multi-skill suite.

## MCP first, HTTP fallback

MCP gives supporting Agents typed tools and server instructions. A standard-library
HTTP client covers tools that can execute scripts but cannot attach MCP servers. Both
paths call the same Memory Hub and therefore cannot diverge into separate memory stores.

## Three physical scopes

The skill preserves `profile`, `episodes`, and `vault` as explicit query dimensions. It
does not read SQLite files or combine them into a fourth index. Federation belongs to
the Memory Hub, which can retrieve across selected scopes while retaining provenance.

## Loopback enforcement

The fallback client accepts only literal loopback hosts over HTTP and rejects embedded
credentials or path overrides. Portability does not justify a remote default for private
memory.

## Standard library only

`urllib`, `json`, and `argparse` are sufficient. Avoiding third-party packages makes the
fallback usable in more Agent environments and removes package-install and supply-chain
friction.

## Native hooks with a human write gate

Codex `UserPromptSubmit` and `Stop` hooks provide automatic recall and task-outcome
capture without adding another Agent runtime. Hooks and MCP may create only pending
candidates. Dashboard decisions are kept outside the Agent tool surface, and approved
content is written as an individually removable managed block.
