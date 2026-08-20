# Personal Memory Recall Skill

A cross-platform Agent Skills package that gives assistants one safe entry to Afei's
three isolated local memory domains. MCP-capable Agents call the Memory Hub directly;
other Agents use the bundled standard-library client.

## Prerequisites

- Feili Personal Memory Hub running at `http://127.0.0.1:8771`.
- Python 3.10+ for the fallback client.
- No third-party Python package and no external network connection.

## Installation

### Universal path

Copy the whole directory to:

```text
~/.agents/skills/personal-memory-recall-skill/
```

This path is discovered by Codex CLI and other tools that support the universal Agent
Skills convention.

### Installer

On macOS, Linux, WSL, or Git Bash:

```bash
chmod +x install.sh
./install.sh --platform universal
./install.sh --all
./install.sh --dry-run
```

The installer also supports Claude Code, GitHub Copilot, Cursor, Windsurf, Cline,
Gemini CLI, Kiro, Trae, Goose, OpenCode, Roo Code, and Antigravity.

### Manual platform paths

| Platform | User or project path |
|---|---|
| Universal / Codex | `~/.agents/skills/personal-memory-recall-skill/` |
| Claude Code | `~/.claude/skills/personal-memory-recall-skill/` |
| GitHub Copilot | `.github/skills/personal-memory-recall-skill/` |
| Cursor | `.cursor/rules/personal-memory-recall-skill/` |
| Windsurf | `.windsurf/rules/personal-memory-recall-skill/` |
| Cline | `.clinerules/personal-memory-recall-skill/` |
| Gemini CLI | `~/.gemini/skills/personal-memory-recall-skill/` |

## Usage

Explicit invocation:

```text
/personal-memory-recall-skill 继续上次的记忆中枢工作
```

Direct client checks:

```bash
python scripts/memory_client.py check-prereqs
python scripts/memory_client.py context "我喜欢什么样的界面" --scopes profile
python scripts/memory_client.py sync --scopes episodes vault
python scripts/memory_client.py propose "阿飞偏好可审计的自动化" --scope profile
```

Agents should also activate the skill automatically when a request depends on previous
preferences, active project context, prior decisions, or Obsidian knowledge.

## MCP configuration

For Codex, register the streamable HTTP server:

```bash
codex mcp add personal_memory --url http://127.0.0.1:8771/mcp
```

Restart the client after changing MCP configuration. The server exposes
`memory_status`, `memory_sync`, `memory_search`, `memory_context`, and the review-only
`memory_propose`. Approval and forgetting are available only in the local dashboard.

## Codex native hooks

`scripts/codex_memory_hook.py` handles `UserPromptSubmit` and `Stop`. It retrieves
relevant memory before complex prompts and submits completed-task candidates after a
turn. The bridge uses only loopback HTTP and Python's standard library. Codex requires
the exact command hook to be reviewed once after installation.

## Verification

```bash
python -m unittest discover -s tests -v
python scripts/memory_client.py diagnostics
```

Expected readiness response includes `"ready": true` and service identity
`"feili-personal-memory"`.

## Troubleshooting

- Connection refused: start the local Memory Hub and rerun `check-prereqs`.
- Wrong service identity: another process owns port 8771; do not bypass the check.
- Empty result: choose the correct scope, inspect `status`, then refresh that scope.
- Candidate not written: expected until it is approved in the Memory Observatory.
- Vector provider disabled: expected by default; lexical retrieval remains local.
- Obsidian URI does not open: verify the local vault name and URI handler.

See [references/usage.md](references/usage.md) for detailed retrieval and privacy rules.

## License

MIT
