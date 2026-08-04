<!-- mcp-name: io.github.airblackbox/air-blackbox-mcp -->
# AIR Blackbox MCP Server

<p align="center">
  <img src="demo.gif" alt="AIR Blackbox MCP demo" width="900">
</p>

EU AI Act compliance scanning for **Claude Desktop**, **Claude Code**, **Cursor**, and any MCP-compatible client.

Unlike other compliance scanners that only report problems, AIR Blackbox also **remediates** - generating working code fixes, trust layer integrations, GDPR compliance checks, bias analysis, and full compliance reports. Under the hood, the scanning feeds into **air-trust**, a cryptographic audit chain (HMAC-SHA256) with Ed25519 signed handoffs that ensures compliance data integrity.

## 14 Tools (10 base + 4 SDK-powered)

| Tier | Tool | What it does | Requires SDK |
|------|------|-------------|---|
| Scanning | `scan_code` | Scan Python code string for all 6 EU AI Act articles | No |
| Scanning | `scan_file` | Read and scan a single Python file | No |
| Scanning | `scan_project` | Recursively scan all .py files in a directory | No |
| Analysis | `analyze_with_model` | Deep analysis via local fine-tuned model (Ollama) | No |
| Analysis | `check_injection` | Detect prompt injection attacks (15 patterns) | No |
| Analysis | `classify_risk` | Classify tools by EU AI Act risk level | No |
| Remediation | `add_trust_layer` | Generate trust layer integration code | No |
| Remediation | `suggest_fix` | Get article-specific fix recommendations | No |
| Documentation | `explain_article` | Technical explanation of EU AI Act articles | No |
| Documentation | `generate_compliance_report` | Full markdown compliance report | No |
| GDPR | `scan_gdpr` | GDPR-specific compliance scan | Yes |
| Bias | `scan_bias` | Bias and fairness analysis | Yes |
| Validation | `validate_action` | Validate agent actions before execution (Article 14) | Yes |
| History | `compliance_history` | View past scans, trends, and compliance scores | Yes |

## Supported Frameworks

LangChain, CrewAI, AutoGen, OpenAI, Haystack, LlamaIndex, Semantic Kernel, Google ADK, Claude Agent SDK, and generic RAG pipelines.

## Installation

### Basic (10 tools, no SDK features)

```bash
pip install air-blackbox-mcp
```

Works standalone with just the lightweight built-in scanner.

### Full (14 tools with GDPR, bias, validation, and history)

```bash
pip install air-blackbox-mcp[full]
```

Installs the full `air-blackbox` SDK (`>=1.13,<2`) for advanced compliance
features. The floor is the version this package is tested against, and the
major cap means a 2.x SDK cannot silently change your findings.

## MCP SDK compatibility (mcp 2.0)

**This package supports both MCP SDK generations — `mcp>=1.0`, no upper bound.**

mcp 2.0 removed `mcp.server.fastmcp` and replaced `FastMCP` with `MCPServer`.
Rather than pin away from it, the server detects which generation is installed
and binds to the right class, so it runs on 1.x and 2.x alike:

| installed | server class |
|---|---|
| `mcp` 1.x | `FastMCP` |
| `mcp` 2.x | `MCPServer` |

Both paths are covered by tests that launch `python -m air_blackbox_mcp` as a
real subprocess and drive it over stdio — the same way Claude Desktop and
Cursor do — and the full suite runs green on both.

**If you are on 0.2.3, upgrade.** That version declared an unpinned
`mcp>=1.0.0`, so once mcp 2.0 shipped, every fresh install produced a server
that died on import:

```
ModuleNotFoundError: No module named 'mcp.server.fastmcp'
```

```bash
pip install --upgrade air-blackbox-mcp
```

0.2.4 fixed it by capping at `mcp<2`; 0.3.0 removes the cap entirely, so this
server no longer conflicts with anything built for mcp 2.x sharing the same
environment.

## Claude Desktop Setup

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "air-blackbox": {
      "command": "python3",
      "args": ["-m", "air_blackbox_mcp"]
    }
  }
}
```

Restart Claude Desktop. The 14 tools will appear automatically.

## Claude Code / Cursor Setup

Add to `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "air-blackbox": {
      "command": "python3",
      "args": ["-m", "air_blackbox_mcp"]
    }
  }
}
```

Or add to `.claude/mcp.json` for Claude Code.

## Usage Examples

In Claude Desktop, Claude Code, or Cursor, just ask:

- "Scan this code for EU AI Act compliance"
- "Add a trust layer to this LangChain agent"
- "Check this text for prompt injection"
- "What does Article 12 require?"
- "Generate a compliance report for ~/myproject"
- "Classify the risk level of `send_email`"
- "Scan this code for GDPR issues" (requires full SDK)
- "Check for bias in this AI model code" (requires full SDK)
- "Can my agent call this shell function?" (requires full SDK)
- "Show me my compliance trends" (requires full SDK)

## SDK Features (Optional)

The full `air-blackbox` SDK unlocks 4 additional tools:

1. **GDPR Scanning** (`scan_gdpr`)
   - Personal data handling without consent
   - Data retention and erasure policies
   - Cross-border transfer safeguards
   - Data processing agreements

2. **Bias Analysis** (`scan_bias`)
   - Disparate impact risk detection
   - Protected attribute handling
   - Training data bias indicators
   - Fairness metric awareness

3. **Action Validation** (`validate_action`)
   - Pre-execution approval gates (Article 14)
   - ConsentGate policy enforcement
   - Risk-based action filtering
   - Audit trail generation

4. **Compliance History** (`compliance_history`)
   - Track past scan results
   - Analyze compliance trends
   - Export audit trails
   - Monitor improvement over time

## Optional: Deep Analysis with Ollama

For AI-powered analysis beyond regex patterns:

```bash
# Install Ollama
brew install ollama

# Pull the fine-tuned compliance model
ollama pull air-compliance-v2

# The analyze_with_model tool will automatically use it
```

## What Makes This Different

Other MCP compliance tools only scan. AIR Blackbox:

1. **Scans + Remediates** - finds issues across 6 EU AI Act articles AND generates working code fixes
2. **Analyzes deeply** - regex patterns + AI-powered model analysis + prompt injection detection (15 patterns)
3. **Validates before execution** - pre-approval gates and risk classification for agent actions (Article 14)
4. **Tracks compliance** - GDPR checks, bias analysis, full reports, and historical trend monitoring (SDK)

## Architecture

**Which engine runs is fixed per tool, not a runtime fallback.** Earlier versions
of this README described a "try the SDK first, fall back to built-in" pattern.
That was never what the code did, and it mattered: a reader could not tell
whether two reports came from the same rules. The actual behavior:

| Tools | Engine | If the SDK is missing |
|---|---|---|
| Tiers 1–4 (`scan_code`, `scan_file`, `scan_project`, `check_injection`, `classify_risk`, …) | Always the built-in rule-based scanner | No effect — these never use the SDK |
| Tier 5 (`scan_gdpr`, `scan_bias`, `validate_action`, `compliance_history`) | Always the full `air-blackbox` SDK | Explicit error telling you to install `[full]` |

So a given tool produces results from the same engine on every install, and
there is no silent switch between engines.

### Result provenance

Because a compliance finding is only comparable to another if you know what
produced it, **every machine-readable result carries a `provenance` block**:

```json
{
  "findings": [ ... ],
  "provenance": {
    "engine": "builtin-rules",
    "scanner_version": "0.2.4",
    "ruleset_id": "eu-ai-act-art9-15",
    "ruleset_version": "cee71577c486",
    "sdk_version": null
  }
}
```

- `engine` — `builtin-rules` or `air-blackbox-sdk`, whichever actually ran.
- `ruleset_version` — a content hash of the active rules, not a hand-maintained
  string. Change a regex and it changes by itself; a version someone must
  remember to bump is one that eventually misreports which rules ran.
- `sdk_version` — the SDK that *produced* this result, so it is `null` for
  built-in results even when the SDK is installed alongside. Reporting a
  version that contributed nothing would imply its rules ran.

Two reports with the same `engine` + `ruleset_version` were produced by
byte-identical rules and can be diffed directly. Different values mean the
rules moved, and the diff needs that context to be meaningful.

Errors carry provenance too — knowing which version produced an error is as
useful as knowing which version produced a finding.

Install `[full]` to unlock the Tier 5 SDK tools; the base install works
standalone.

## Part of AIR Blackbox

This MCP server is part of the **AIR Blackbox ecosystem**:

- **air-trust** on [PyPI](https://pypi.org/project/air-trust/) - the cryptographic audit chain that backs compliance scanning
- **air-blackbox** on [PyPI](https://pypi.org/project/air-blackbox/) - the full compliance SDK and CLI scanner
- **[airblackbox.ai](https://airblackbox.ai)** - the project homepage and docs

## Links

- [EU AI Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) - the regulation
- [GDPR](https://gdpr-info.eu/) - data protection regulation
