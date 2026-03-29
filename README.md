<!-- mcp-name: io.github.airblackbox/air-blackbox-mcp -->
# AIR Blackbox MCP Server

EU AI Act compliance scanning for **Claude Desktop**, **Claude Code**, **Cursor**, and any MCP-compatible client.

Unlike other compliance scanners that only report problems, AIR Blackbox also **remediates** — generating working code fixes, trust layer integrations, GDPR compliance checks, bias analysis, and full compliance reports.

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

Installs the full `air-blackbox` SDK (v1.6.3+) for advanced compliance features.

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

1. **Scans** — 18+ regex patterns across 6 EU AI Act articles
2. **Analyzes** — local fine-tuned model catches what regex misses
3. **Remediates** — generates working code to fix findings
4. **Verifies** — checks for prompt injection with 15 weighted patterns
5. **Classifies** — maps tools to EU AI Act risk levels
6. **Documents** — produces full compliance reports
7. **GDPR checks** — personal data and consent auditing (SDK)
8. **Bias detection** — fairness and disparate impact analysis (SDK)
9. **Validates actions** — pre-execution approval gates (SDK)
10. **Tracks trends** — compliance history and improvement metrics (SDK)

## Architecture

The server uses a smart fallback pattern:

1. **Try SDK first** — If `air-blackbox>=1.6.0` is installed, use the full compliance engine
2. **Fall back gracefully** — If SDK isn't installed, use the lightweight built-in scanner
3. **No breaking changes** — Works with just `pip install air-blackbox-mcp` (basic mode)
4. **Opt-in superpower** — Install `[full]` to unlock advanced features

This means the MCP server works standalone, but gets dramatically more powerful when the SDK is present.

## Links

- [AIR Blackbox Scanner](https://pypi.org/project/air-blackbox/) — the CLI scanner
- [airblackbox.ai](https://airblackbox.ai) — project homepage
- [EU AI Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — the regulation
- [GDPR](https://gdpr-info.eu/) — data protection regulation
