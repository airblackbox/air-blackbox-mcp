"""
AIR Blackbox MCP Server — EU AI Act compliance scanning for AI agents.

14 tools across 4 tiers with full air-blackbox SDK integration:
  Tier 1 — Scanning:    scan_code, scan_file, scan_project
  Tier 2 — Analysis:    analyze_with_model, check_injection, classify_risk
  Tier 3 — Remediation: add_trust_layer, suggest_fix
  Tier 4 — Docs:        explain_article, generate_compliance_report
  Tier 5 — New SDK:     scan_gdpr, scan_bias, validate_action, compliance_history

Server tries to import from air_blackbox SDK first. Falls back to built-in
lightweight scanner if SDK is not installed.
"""
import json
import subprocess
import tempfile
import os
from mcp.server.fastmcp import FastMCP

from air_blackbox_mcp.scanner import (
    scan_code as _scan_code,
    scan_file as _scan_file,
    scan_project as _scan_project,
    check_injection as _check_injection,
    classify_risk as _classify_risk,
    detect_framework,
)

mcp = FastMCP(
    "air-blackbox",
    instructions="EU AI Act compliance scanner with GDPR and bias detection — scan, analyze, remediate, and protect AI agent code. 14 tools across scanning, analysis, remediation, and documentation.",
)


# ==============================================================================
# TIER 1: SCANNING
# ==============================================================================

@mcp.tool()
async def scan_code(code: str) -> str:
    """Scan Python AI agent code for EU AI Act compliance (Articles 9-15).

    Detects frameworks (LangChain, CrewAI, AutoGen, OpenAI, Haystack, LlamaIndex),
    checks for trust layer presence, and evaluates compliance across 6 articles:
    - Art 9: Risk Management (error handling, fallbacks)
    - Art 10: Data Governance (input validation, PII handling)
    - Art 11: Technical Documentation (docstrings, type hints)
    - Art 12: Record-Keeping (logging, tracing, audit trails)
    - Art 14: Human Oversight (HITL, rate limits, identity binding)
    - Art 15: Accuracy & Security (injection defense, output validation)

    Returns findings with severity, fix recommendations, and compliance score.
    """
    result = _scan_code(code)
    return json.dumps(result, indent=2)


@mcp.tool()
async def scan_file(file_path: str) -> str:
    """Scan a single Python file for EU AI Act compliance.

    Reads the file, detects framework and trust layer, then runs all
    compliance checks. Returns line count, findings, and score.
    """
    result = _scan_file(file_path)
    return json.dumps(result, indent=2)


@mcp.tool()
async def scan_project(directory: str) -> str:
    """Recursively scan all Python files in a project directory.

    Walks the directory tree (skipping node_modules, .git, __pycache__, etc.),
    reads all .py files, and runs a combined compliance analysis.
    Returns per-file info, aggregate findings, and project-level score.
    """
    result = _scan_project(directory)
    return json.dumps(result, indent=2)



# ==============================================================================
# TIER 2: ANALYSIS
# ==============================================================================

@mcp.tool()
async def analyze_with_model(code: str) -> str:
    """Analyze code using the local air-compliance-v2 fine-tuned model via Ollama.

    If Ollama is running with the air-compliance-v2 model, sends the code
    for deep analysis that catches issues regex patterns miss.

    Falls back gracefully to rule-based scanning if Ollama isn't available.
    """
    # Try Ollama first
    try:
        prompt = (
            "You are an EU AI Act compliance expert. Analyze this Python code for "
            "compliance with Articles 9 (risk management), 10 (data governance), "
            "11 (technical documentation), 12 (record-keeping), 14 (human oversight), "
            "and 15 (accuracy/robustness/cybersecurity). "
            "For each article, state whether the code is compliant, partially compliant, "
            "or non-compliant, and explain why. Suggest specific fixes.\n\n"
            f"Code:\n```python\n{code}\n```"
        )
        result = subprocess.run(
            ["ollama", "run", "air-compliance-v2", prompt],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.dumps({
                "source": "air-compliance-v2 (fine-tuned model)",
                "analysis": result.stdout.strip(),
                "rule_based_scan": _scan_code(code),
            }, indent=2)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # Fallback to rule-based
    result = _scan_code(code)
    result["source"] = "rule-based (Ollama not available — install: ollama pull air-compliance-v2)"
    return json.dumps(result, indent=2)


@mcp.tool()
async def check_injection(text: str) -> str:
    """Scan text for prompt injection attacks.

    Checks 15 weighted patterns including role override, jailbreak attempts,
    system prompt manipulation, safety bypass, privilege escalation, and more.

    Returns detected patterns, confidence score (0-1), and whether the text
    would be blocked by a guardrail.
    """
    result = _check_injection(text)
    return json.dumps(result, indent=2)


@mcp.tool()
async def classify_risk(tool_name: str) -> str:
    """Classify a tool or function by EU AI Act risk level.

    Maps tool names to risk categories:
    - CRITICAL: shell, exec, delete, spawn (requires human approval)
    - HIGH: database, email, deploy, git_push (requires approval or audit)
    - MEDIUM: http_request, api_call (should be logged)
    - LOW: file_read, search, query (standard logging)

    Uses the same risk map as AIR ConsentGate.
    """
    result = _classify_risk(tool_name)
    return json.dumps(result, indent=2)



# ==============================================================================
# TIER 3: REMEDIATION
# ==============================================================================

TRUST_LAYER_TEMPLATES = {
    "langchain": {
        "package": "air-blackbox[langchain]",
        "install": "pip install air-blackbox[langchain]",
        "code": '''from air_blackbox import AirTrust

# Auto-detecting trust layer
trust = AirTrust()

# Attach to your LangChain agent/chain
# Automatically adds HMAC-SHA256 audit trails, PII tokenization,
# and consent gates for high-risk actions
trusted_agent = trust.attach(your_agent)

result = trusted_agent.invoke(input_data)''',
    },
    "crewai": {
        "package": "air-blackbox[crewai]",
        "install": "pip install air-blackbox[crewai]",
        "code": '''from crewai import Crew, Agent, Task
from air_blackbox import AirTrust

# Auto-detecting trust layer
trust = AirTrust()

# Create your crew, then attach trust layer
crew = Crew(agents=[your_agent], tasks=[your_task])
trusted_crew = trust.attach(crew)

result = trusted_crew.kickoff()''',
    },
    "autogen": {
        "package": "air-blackbox[autogen]",
        "install": "pip install air-blackbox[autogen]",
        "code": '''from autogen import AssistantAgent, UserProxyAgent
from air_blackbox import AirTrust

# Auto-detecting trust layer
trust = AirTrust()

# Wrap your agents with trust layer
assistant = AssistantAgent("assistant", llm_config=llm_config)
trusted_assistant = trust.attach(assistant)

user_proxy = UserProxyAgent("user_proxy", code_execution_config=False)
user_proxy.initiate_chat(trusted_assistant, message="Your task here")''',
    },
    "openai": {
        "package": "air-blackbox[openai]",
        "install": "pip install air-blackbox[openai]",
        "code": '''from openai import OpenAI
from air_blackbox import AirTrust

# Auto-detecting trust layer
trust = AirTrust()

# Wrap the OpenAI client (auto-detects OpenAI SDK)
client = trust.attach(OpenAI())

# Use exactly like the normal OpenAI client
# Every call is now HMAC-logged with tamper-evident audit trails
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Your prompt here"}]
)''',
    },
    "haystack": {
        "package": "air-blackbox[haystack]",
        "install": "pip install air-blackbox[haystack]",
        "code": '''from air_blackbox import AirTrust

# Auto-detecting trust layer for Haystack RAG pipelines
trust = AirTrust()

# Wrap your Haystack pipeline
trusted_pipeline = trust.attach(your_pipeline)

result = trusted_pipeline.run({"query": "Your question here"})''',
    },
    "adk": {
        "package": "air-blackbox[adk]",
        "install": "pip install air-blackbox[adk]",
        "code": '''from air_blackbox import AirTrust

# Auto-detecting trust layer for Google Agent Development Kit
trust = AirTrust()

# Wrap your ADK agent
trusted_agent = trust.attach(your_adk_agent)''',
    },
    "claude": {
        "package": "air-blackbox[claude]",
        "install": "pip install air-blackbox[claude]",
        "code": '''from air_blackbox import AirTrust

# Auto-detecting trust layer for Claude Agent SDK
trust = AirTrust()

# Wrap your Claude agent
trusted_agent = trust.attach(your_claude_agent)''',
    },
}

ARTICLE_FIXES = {
    9: {
        "title": "Risk Management",
        "fixes": {
            "langchain": "Add try/except around .invoke() calls. Use with_fallbacks() for model failover. Set max_retries in LLM config.",
            "crewai": "Set max_rpm on agents. Use step_callback for error monitoring. Add fallback agents for critical tasks.",
            "autogen": "Set max_consecutive_auto_reply. Add error handling in code_execution_config. Use is_termination_msg for safety.",
            "openai": "Wrap completions.create() in try/except. Set timeout parameter. Add retry logic with exponential backoff.",
            "default": "Wrap all LLM calls in try/except. Add fallback responses. Set token budgets and timeouts.",
        },
    },
    10: {
        "title": "Data Governance",
        "fixes": {
            "langchain": "Use PydanticOutputParser for structured I/O. Add PromptTemplate input variables. Consider DataVault for PII.",
            "crewai": "Use output_pydantic on Tasks for structured output. Add input validation on Agent backstory templates.",
            "autogen": "Validate message content before processing. Add content filters in agent config.",
            "openai": "Use response_format for structured output. Validate inputs with Pydantic before sending to API.",
            "default": "Add Pydantic models for input/output validation. Implement PII detection before sending to LLM providers.",
        },
    },
    11: {
        "title": "Technical Documentation",
        "fixes": {
            "langchain": "Add docstrings to all chain/agent classes. Create MODEL_CARD.md. Add type hints to chain methods.",
            "crewai": "Document Agent roles, goals, backstories. Add docstrings to Task descriptions. Create system README.",
            "autogen": "Document agent configurations. Add type hints to message handlers. Create architecture README.",
            "openai": "Add docstrings to API wrapper functions. Document prompt templates. Create MODEL_CARD.md.",
            "default": "Add docstrings to all public functions. Add type hints. Create README.md and MODEL_CARD.md.",
        },
    },
    12: {
        "title": "Record-Keeping",
        "fixes": {
            "langchain": "Add LangSmith or LangFuse callbacks. Use AirTrustCallbackHandler for HMAC audit chain. Add structlog.",
            "crewai": "Use step_callback for execution logging. Add event bus listeners. Enable CONTENT_TRACING.",
            "autogen": "Log all agent messages via logging_session_id. Add custom message hooks for audit trail.",
            "openai": "Log all API calls with run_id, model, tokens. Add OpenTelemetry spans. Use AirOpenAITrust for audit chain.",
            "default": "Add import logging with structured format. Add OpenTelemetry tracing. Log all LLM calls and agent actions.",
        },
    },
    14: {
        "title": "Human Oversight",
        "fixes": {
            "langchain": "Add HumanApprovalCallbackHandler. Use interrupt_before in LangGraph. Set max_iterations on AgentExecutor.",
            "crewai": "Set allow_delegation=False for restricted agents. Add human_input=True on critical tasks. Set max_rpm.",
            "autogen": "Set human_input_mode='ALWAYS' for critical agents. Use max_consecutive_auto_reply. Add termination conditions.",
            "openai": "Add confirmation step before tool execution. Set max_tokens. Implement approval gates for actions.",
            "default": "Add human approval gates for high-risk actions. Set token/iteration limits. Track user identity per action.",
        },
    },
    15: {
        "title": "Accuracy, Robustness & Cybersecurity",
        "fixes": {
            "langchain": "Add NeMo Guardrails or Rebuff. Use PydanticOutputParser. Set retry logic with tenacity.",
            "crewai": "Use hallucination_guardrail and output_guardrail. Set output_pydantic for structured validation.",
            "autogen": "Add content filters in agent config. Validate code before execution. Set execution timeout.",
            "openai": "Add input sanitization. Use structured output (response_format). Set timeout and max_retries.",
            "default": "Add prompt injection detection. Use output parsers. Add retry with backoff. Sanitize user input.",
        },
    },
}


@mcp.tool()
async def add_trust_layer(code: str, framework: str = "") -> str:
    """Generate modified code WITH the AIR trust layer added.

    Given existing AI agent code and framework name, returns working,
    copy-paste-ready code with imports and setup for the AIR trust layer.

    Supports: langchain, crewai, autogen, openai, rag.
    If framework is empty, auto-detects from the code.
    """
    # Auto-detect framework if not provided
    if not framework:
        detected = detect_framework(code)
        framework = detected[0] if detected else ""

    fw = framework.lower().replace("-", "").replace("_", "").replace(" ", "")
    # Normalize common names
    fw_map = {"llama_index": "haystack", "llamaindex": "haystack", "rag": "haystack",
              "googleadk": "adk", "claudeagent": "claude", "anthropic": "claude"}
    fw = fw_map.get(fw, fw)

    if fw not in TRUST_LAYER_TEMPLATES:
        return json.dumps({
            "error": f"Framework '{framework}' not recognized.",
            "supported": list(TRUST_LAYER_TEMPLATES.keys()),
            "tip": "Try one of: langchain, crewai, autogen, openai, rag",
        }, indent=2)

    tmpl = TRUST_LAYER_TEMPLATES[fw]
    return json.dumps({
        "framework": fw,
        "package": tmpl["package"],
        "install_command": tmpl["install"],
        "trust_layer_code": tmpl["code"],
        "original_code": code,
        "instructions": (
            f"1. Install: {tmpl['install']}\n"
            f"2. Add the trust layer imports and setup shown above\n"
            f"3. The trust layer automatically handles:\n"
            f"   - HMAC-SHA256 tamper-evident audit logging (Article 12)\n"
            f"   - ConsentGate for high-risk actions (Article 14)\n"
            f"   - DataVault PII tokenization (Article 10)\n"
            f"4. Run 'air-blackbox comply --scan' to verify compliance improvement"
        ),
    }, indent=2)


@mcp.tool()
async def suggest_fix(article: int, framework: str = "") -> str:
    """Get specific code fix suggestions for a given EU AI Act article.

    Provide an article number (9, 10, 11, 12, 14, or 15) and optionally
    the framework name for framework-specific recommendations.
    """
    if article not in ARTICLE_FIXES:
        return json.dumps({
            "error": f"Article {article} not covered. Supported: 9, 10, 11, 12, 14, 15",
        }, indent=2)

    art = ARTICLE_FIXES[article]
    fw = framework.lower().strip() if framework else "default"

    # Try framework-specific fix, fall back to default
    fix = art["fixes"].get(fw, art["fixes"]["default"])

    # Also include trust layer info if applicable
    # Normalize framework names for template lookup
    fw_lookup = {"rag": "haystack"}.get(fw, fw)
    trust_info = None
    if fw_lookup in TRUST_LAYER_TEMPLATES:
        tmpl = TRUST_LAYER_TEMPLATES[fw_lookup]
        trust_info = {
            "package": tmpl["package"],
            "install": tmpl["install"],
        }

    return json.dumps({
        "article": article,
        "title": art["title"],
        "framework": fw,
        "fix_recommendations": fix,
        "trust_layer": trust_info,
    }, indent=2)



# ==============================================================================
# TIER 4: DOCUMENTATION
# ==============================================================================

ARTICLE_EXPLANATIONS = {
    9: {
        "title": "Risk Management (Article 9)",
        "summary": "Providers must establish and maintain a risk management system throughout the AI system's lifecycle.",
        "requirements": [
            "Identify and analyze known and foreseeable risks",
            "Implement risk mitigation measures (error handling, fallbacks)",
            "Test and validate risk controls",
            "Monitor residual risks during deployment",
        ],
        "trust_layer_components": [
            "Error handling around all LLM calls (try/except)",
            "Fallback models and default responses",
            "Rate limiting and token budgets",
            "Gateway guardrails for runtime risk control",
        ],
    },
    10: {
        "title": "Data Governance (Article 10)",
        "summary": "Training, validation, and testing data must meet quality criteria including relevance, representativeness, and accuracy.",
        "requirements": [
            "Document data sources, collection, and processing",
            "Ensure data quality and representativeness",
            "Detect and address potential biases",
            "Implement PII protection and consent management",
        ],
        "trust_layer_components": [
            "DataVault for PII tokenization before LLM calls",
            "Input validation with Pydantic schemas",
            "Data governance documentation (DATA_GOVERNANCE.md)",
            "Consent tracking via ConsentGate",
        ],
    },
    11: {
        "title": "Technical Documentation (Article 11)",
        "summary": "Technical documentation must be created and maintained, covering system architecture, purpose, and capabilities.",
        "requirements": [
            "General system description and intended purpose",
            "Design specifications and architecture",
            "Monitoring, functioning, and control descriptions",
            "Documentation of data requirements",
        ],
        "trust_layer_components": [
            "README.md with system purpose and architecture",
            "MODEL_CARD.md with model capabilities and limitations",
            "Code documentation (docstrings, type hints)",
            "Runtime AI-BOM (Bill of Materials) from gateway traffic data",
        ],
    },
    12: {
        "title": "Record-Keeping (Article 12)",
        "summary": "High-risk AI systems must enable automatic recording of events (logs) for traceability.",
        "requirements": [
            "Automatic logging of system events",
            "Traceability of AI decisions and actions",
            "Tamper-evident record storage",
            "Retention of logs for compliance audits",
        ],
        "trust_layer_components": [
            "AuditLedger with HMAC-SHA256 tamper-evident chain",
            "Application logging (structlog, loguru)",
            "OpenTelemetry tracing integration",
            "Agent action audit trails (who did what, when, why)",
        ],
    },
    14: {
        "title": "Human Oversight (Article 14)",
        "summary": "High-risk AI systems must enable human oversight, including the ability to understand, monitor, and intervene.",
        "requirements": [
            "Enable humans to understand AI system capabilities/limitations",
            "Allow monitoring of AI operation",
            "Provide ability to intervene or halt the system",
            "Maintain human control over automated decisions",
        ],
        "trust_layer_components": [
            "ConsentGate for human approval of high-risk actions",
            "Kill switch via gateway guardrails",
            "Human-in-the-loop patterns (approval gates, interrupts)",
            "Usage limits and budget controls (max_tokens, max_iterations)",
        ],
    },
    15: {
        "title": "Accuracy, Robustness & Cybersecurity (Article 15)",
        "summary": "AI systems must achieve appropriate accuracy, robustness, and cybersecurity throughout their lifecycle.",
        "requirements": [
            "Appropriate level of accuracy for intended purpose",
            "Resilience to errors and inconsistencies",
            "Protection against unauthorized manipulation (adversarial attacks)",
            "Cybersecurity measures appropriate to the risks",
        ],
        "trust_layer_components": [
            "Prompt injection detection (15 weighted patterns)",
            "Output validation via Pydantic parsers",
            "Retry logic with exponential backoff",
            "Guardrails (NeMo, Rebuff, Lakera integration)",
        ],
    },
}


@mcp.tool()
async def explain_article(article: int) -> str:
    """Get a technical explanation of a specific EU AI Act article.

    Explains what Article 9, 10, 11, 12, 14, or 15 requires,
    mapped to specific AIR trust layer components that address each requirement.
    """
    if article not in ARTICLE_EXPLANATIONS:
        return json.dumps({
            "error": f"Article {article} not covered. Supported: 9, 10, 11, 12, 14, 15",
        }, indent=2)

    return json.dumps(ARTICLE_EXPLANATIONS[article], indent=2)


@mcp.tool()
async def generate_compliance_report(code: str) -> str:
    """Generate a full markdown compliance report for AI agent code.

    Scans the code and produces a formatted report with:
    - Framework detection results
    - Trust layer status
    - Per-article findings with severity
    - Fix recommendations
    - Overall compliance score
    """
    scan = _scan_code(code)
    frameworks = scan["frameworks"]
    has_trust = scan["trust_layer_detected"]
    findings = scan["findings"]
    summary = scan["summary"]

    # Build markdown report
    lines = []
    lines.append("# EU AI Act Compliance Report")
    lines.append(f"\n**Generated by**: AIR Blackbox MCP v0.2.0")
    lines.append(f"**Frameworks detected**: {', '.join(frameworks) if frameworks else 'None detected'}")
    lines.append(f"**Trust layer present**: {'Yes' if has_trust else 'No'}")
    lines.append(f"**Compliance score**: {summary['score']} ({summary['passing']} passing, {summary['warnings']} warnings, {summary['failing']} failing)")

    # Group findings by article
    by_article = {}
    for f in findings:
        by_article.setdefault(f["article"], []).append(f)

    article_names = {9: "Risk Management", 10: "Data Governance", 11: "Technical Documentation",
                     12: "Record-Keeping", 14: "Human Oversight", 15: "Accuracy, Robustness & Cybersecurity"}

    for art_num in sorted(by_article.keys()):
        art_findings = by_article[art_num]
        art_name = article_names.get(art_num, f"Article {art_num}")
        lines.append(f"\n## Article {art_num} — {art_name}\n")
        for f in art_findings:
            icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}.get(f["status"], "?")
            lines.append(f"- **[{icon}]** {f['name']} (Severity: {f['severity']})")
            lines.append(f"  - {f['evidence']}")
            if f.get("fix_hint"):
                lines.append(f"  - Fix: {f['fix_hint']}")

    if not has_trust:
        fw = frameworks[0] if frameworks else "langchain"
        # Normalize framework name for template lookup
        fw_lookup = {"llama_index": "haystack", "llamaindex": "haystack", "rag": "haystack",
                     "googleadk": "adk", "claudeagent": "claude", "anthropic": "claude"}.get(fw, fw)
        lines.append("\n## Recommended Next Step\n")
        lines.append(f"Install the AIR trust layer for {fw}:")
        if fw_lookup in TRUST_LAYER_TEMPLATES:
            lines.append(f"```\n{TRUST_LAYER_TEMPLATES[fw_lookup]['install']}\n```")
        else:
            lines.append("```\npip install air-blackbox\n```")

    report = "\n".join(lines)
    return json.dumps({"report": report, "scan_data": scan}, indent=2)



# ==============================================================================
# TIER 5: NEW SDK-POWERED TOOLS (requires air_blackbox SDK)
# ==============================================================================

@mcp.tool()
async def scan_gdpr(code: str) -> str:
    """Scan code for GDPR compliance issues (data processing, consent, etc.).

    Available only with full air-blackbox SDK (pip install air-blackbox[full]).

    Checks for:
    - Personal data handling without consent
    - Data retention policies
    - Right to erasure implementation
    - Data processing agreements
    - Cross-border data transfer safeguards
    """
    try:
        from air_blackbox.compliance.gdpr_scanner import scan_gdpr as _scan_gdpr_sdk
    except ImportError:
        return json.dumps({
            "error": "air_blackbox SDK not installed",
            "install": "pip install air-blackbox[full]",
            "note": "GDPR scanning requires the full SDK. Basic scanning still available without it.",
        }, indent=2)

    try:
        # Write code to temp file for SDK scanner
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        result = _scan_gdpr_sdk(temp_path)
        os.unlink(temp_path)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "note": "GDPR scanner failed. Ensure air_blackbox>=1.6.3 is installed.",
        }, indent=2)


@mcp.tool()
async def scan_bias(code: str) -> str:
    """Scan code for AI bias and fairness issues.

    Available only with full air-blackbox SDK (pip install air-blackbox[full]).

    Checks for:
    - Disparate impact risks in decision logic
    - Training data bias indicators
    - Protected attribute handling
    - Fairness metric awareness
    - Bias mitigation strategies
    """
    try:
        from air_blackbox.compliance.bias_scanner import scan_bias as _scan_bias_sdk
    except ImportError:
        return json.dumps({
            "error": "air_blackbox SDK not installed",
            "install": "pip install air-blackbox[full]",
            "note": "Bias scanning requires the full SDK. Basic scanning still available without it.",
        }, indent=2)

    try:
        # Write code to temp file for SDK scanner
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        result = _scan_bias_sdk(temp_path)
        os.unlink(temp_path)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "note": "Bias scanner failed. Ensure air_blackbox>=1.6.3 is installed.",
        }, indent=2)


@mcp.tool()
async def validate_action(action_type: str, tool_name: str, risk_level: str = "") -> str:
    """Validate an agent action before execution (Article 14 compliance).

    Available only with full air-blackbox SDK (pip install air-blackbox[full]).

    Use this to check if an action should be executed, requires approval,
    or should be blocked. Implements ConsentGate logic.

    Args:
        action_type: 'tool_call', 'file_write', 'api_call', 'shell_exec', etc.
        tool_name: name of the tool/function being called
        risk_level: optional override ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')

    Returns:
        approval status, required policies, and remediation steps
    """
    try:
        from air_blackbox.validate import validate_action as _validate_sdk
    except ImportError:
        return json.dumps({
            "error": "air_blackbox SDK not installed",
            "install": "pip install air-blackbox[full]",
            "note": "Action validation requires the full SDK.",
        }, indent=2)

    try:
        result = _validate_sdk(
            action_type=action_type,
            tool_name=tool_name,
            risk_level=risk_level or None
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "note": "Validation failed. Check tool_name and action_type.",
        }, indent=2)


@mcp.tool()
async def compliance_history(action: str = "list") -> str:
    """View past scan results, trends, and compliance history.

    Available only with full air-blackbox SDK (pip install air-blackbox[full]).

    Actions:
        'list': Show recent scans
        'trend': Analyze compliance trend over time
        'score': Get latest compliance score
        'export': Export full audit trail

    Helps track compliance progress and identify patterns.
    """
    try:
        from air_blackbox.compliance.history import (
            get_history,
            get_trend,
            get_latest_score,
        )
    except ImportError:
        return json.dumps({
            "error": "air_blackbox SDK not installed",
            "install": "pip install air-blackbox[full]",
            "note": "History tracking requires the full SDK.",
        }, indent=2)

    try:
        if action.lower() == "trend":
            result = get_trend()
        elif action.lower() == "score":
            result = get_latest_score()
        else:  # list (default)
            result = get_history()

        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "note": "History lookup failed. Ensure scans have been run.",
        }, indent=2)

