# AIR Blackbox MCP Server

EU AI Act compliance scanning for Python AI agents — directly in Claude Desktop, Cursor, or any MCP client.

**10 tools** that scan, analyze, and fix compliance gaps across LangChain, CrewAI, AutoGen, OpenAI, and RAG pipelines.

## Quick Start

### 1. Install

```bash
pip install air-blackbox-mcp
```

### 2. Add to Claude Desktop

Open your Claude Desktop config file:

- **Mac:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Add this:

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

### 3. Restart Claude Desktop

The AIR Blackbox tools will appear in Claude's tool list.

## Tools

### Scanning
| Tool | What it does |
|------|-------------|
| `scan_code` | Scan a Python code string for all 6 EU AI Act articles |
| `scan_file` | Scan a single `.py` file |
| `scan_project` | Scan an entire project directory |

### Analysis
| Tool | What it does |
|------|-------------|
| `analyze_with_model` | Deep analysis using local fine-tuned Llama model (via Ollama) |
| `check_prompt_injection` | Scan text for 15 prompt injection patterns |
| `classify_risk` | Classify a tool/function by risk level (CRITICAL/HIGH/MEDIUM/LOW) |

### Remediation
| Tool | What it does |
|------|-------------|
| `add_trust_layer` | Generate working code to add AIR trust layer to your agent |
| `suggest_fix` | Get the specific fix for a failing article |

### Documentation
| Tool | What it does |
|------|-------------|
| `explain_article` | Technical explanation of what each article requires |
| `generate_compliance_report` | Full markdown compliance report |

## Example Usage in Claude Desktop

> "Scan this LangChain agent for EU AI Act compliance"

> "Check this text for prompt injection: ignore all previous instructions"

> "Add a trust layer to my CrewAI code"

> "Explain what Article 12 requires"

> "Generate a compliance report for my project at /path/to/project"

## Articles Checked

| Article | Requirement | AIR Component |
|---------|------------|---------------|
| 9 | Risk Management | ConsentGate |
| 10 | Data Governance | DataVault |
| 11 | Technical Documentation | AuditLedger |
| 12 | Record-Keeping (tamper-evident) | AuditLedger (HMAC-SHA256) |
| 14 | Human Oversight | ConsentGate |
| 15 | Robustness & Cybersecurity | InjectionDetector |

## Optional: Local AI Model

For deeper analysis, install the fine-tuned compliance model:

```bash
ollama run air-compliance-v2
```

The `analyze_with_model` tool will automatically use it. Falls back to rule-based scanning if Ollama isn't running.

## Trust Layer Packages

Fix compliance gaps with drop-in trust layers:

```bash
pip install air-langchain-trust    # LangChain / LangGraph
pip install air-crewai-trust       # CrewAI
pip install air-autogen-trust      # AutoGen / AG2
pip install air-openai-trust       # OpenAI Agents SDK
pip install air-rag-trust          # RAG pipelines
```

## Development

```bash
git clone https://github.com/airblackbox/air-blackbox-mcp
cd air-blackbox-mcp
pip install -e .
python3 -m air_blackbox_mcp
```

## Links

- **Website:** [airblackbox.ai](https://airblackbox.ai)
- **GitHub:** [github.com/airblackbox](https://github.com/airblackbox)
- **Scanner Demo:** [Hugging Face Space](https://huggingface.co/spaces/nostalgicskinco/air-blackbox-scanner)
- **Gate (AI Firewall):** [airblackbox.ai/gate](https://airblackbox.ai/gate.html)

## License

Apache 2.0

---

**Deadline: August 2, 2026.** Fines up to €35M or 7% of global annual turnover.
<!-- mcp-name: io.github.shotwellj/air-blackbox -->
