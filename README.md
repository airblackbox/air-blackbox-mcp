# AIR Blackbox MCP Server

EU AI Act compliance scanning for **Claude Desktop**, **Cursor**, and any MCP-compatible client.

Unlike other compliance scanners that only report problems, AIR Blackbox also **remediates** — generating working code fixes, trust layer integrations, and full compliance reports.

## 10 Tools

| Tier | Tool | What it does |
|------|------|-------------|
| Scanning | `scan_code` | Scan Python code string for all 6 EU AI Act articles |
| Scanning | `scan_file` | Read and scan a single Python file |
| Scanning | `scan_project` | Recursively scan all .py files in a directory |
| Analysis | `analyze_with_model` | Deep analysis via local fine-tuned model (Ollama) |
| Analysis | `check_injection` | Detect prompt injection attacks (15 patterns) |
| Analysis | `classify_risk` | Classify tools by EU AI Act risk level |
| Remediation | `add_trust_layer` | Generate trust layer integration code |
| Remediation | `suggest_fix` | Get article-specific fix recommendations |
| Documentation | `explain_article` | Technical explanation of EU AI Act articles |
| Documentation | `generate_compliance_report` | Full markdown compliance report |

## Supported Frameworks

LangChain, CrewAI, AutoGen, OpenAI, Haystack, LlamaIndex, Semantic Kernel, and generic RAG pipelines.

## Install

```bash
pip install air-blackbox-mcp
```

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

Restart Claude Desktop. The 10 tools will appear automatically.

## Cursor Setup

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

## Usage Examples

In Claude Desktop or Cursor, just ask:

- "Scan this code for EU AI Act compliance"
- "Add a trust layer to this LangChain agent"
- "Check this text for prompt injection"
- "What does Article 12 require?"
- "Generate a compliance report for my project at ~/myproject"
- "Classify the risk level of `send_email`"

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

## Links

- [AIR Blackbox Scanner](https://pypi.org/project/air-blackbox/) — the CLI scanner
- [airblackbox.ai](https://airblackbox.ai) — project homepage
- [EU AI Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — the regulation
