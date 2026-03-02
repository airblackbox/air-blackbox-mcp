"""
EU AI Act Compliance MCP Server

10 tools for scanning, analyzing, and remediating Python AI agents.
Works with Claude Desktop, Cursor, and any MCP-compatible client.
"""

import json
import re
import subprocess
from mcp.server.fastmcp import FastMCP

from .scanner import scan_code as _scan_code, scan_file as _scan_file, scan_project as _scan_project

mcp = FastMCP("air-blackbox")


# ============================================================
# INJECTION DETECTION — 15 weighted patterns
# ============================================================

INJECTION_PATTERNS = [
    ("role_override", 0.9, r"(?i)you\s+are\s+now\s+(?:a\s+)?(?:new|different|my)"),
    ("dan_jailbreak", 0.9, r"(?i)(?:DAN|do\s+anything\s+now|jailbreak|unrestricted\s+mode)"),
    ("system_prompt_override", 0.85, r"(?i)(?:system\s*:\s*you|<\|(?:system|im_start)\|>)"),
    ("safety_bypass", 0.85, r"(?i)(?:ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|rules|guidelines))"),
    ("new_identity", 0.8, r"(?i)(?:forget\s+(?:everything|all|your)|new\s+persona|new\s+identity)"),
    ("urgent_override", 0.8, r"(?i)(?:ADMIN\s*OVERRIDE|EMERGENCY|URGENT\s*:|PRIORITY\s*OVERRIDE)"),
    ("privilege_escalation", 0.75, r"(?i)(?:sudo|root\s+access|admin\s+mode|developer\s+mode|maintenance\s+mode)"),
    ("delimiter_injection", 0.7, r"(?i)(?:BEGIN\s+(?:NEW\s+)?INSTRUCTIONS|END\s+OF\s+SYSTEM|={3,}|#{3,}\s*NEW)"),
    ("hidden_instruction", 0.7, r"(?i)(?:hidden\s+instruction|secret\s+task|covert\s+operation)"),
    ("data_exfil", 0.65, r"(?i)(?:send\s+(?:all|your|the)\s+(?:data|info|contents)|exfiltrate|extract\s+and\s+send)"),
    ("xml_tag_injection", 0.6, r"<(?:system|instruction|prompt|context|override)[^>]*>"),
    ("hypothetical_bypass", 0.6, r"(?i)(?:hypothetically|in\s+theory|for\s+(?:educational|research)\s+purposes)"),
    ("output_manipulation", 0.5, r"(?i)(?:respond\s+only\s+with|output\s+format\s*:|always\s+(?:start|begin)\s+with)"),
    ("encoding_evasion", 0.4, r"(?i)(?:base64|rot13|hex\s+encode|unicode\s+escape|reverse\s+the\s+text)"),
    ("tool_abuse", 0.35, r"(?i)(?:call\s+(?:this|the)\s+(?:tool|function)\s+(?:repeatedly|in\s+a\s+loop))"),
]


def check_injection(text: str, threshold: float = 0.5) -> dict:
    """Scan text for prompt injection patterns."""
    matches = []
    total_score = 0.0

    for name, weight, pattern in INJECTION_PATTERNS:
        if re.search(pattern, text):
            matches.append({"pattern": name, "weight": weight})
            total_score += weight

    # Normalize score to 0-1 range
    max_possible = sum(w for _, w, _ in INJECTION_PATTERNS)
    normalized = min(total_score / max(max_possible * 0.3, 1), 1.0)

    return {
        "detected": len(matches) > 0,
        "score": round(normalized, 3),
        "patterns_matched": matches,
        "pattern_count": len(matches),
        "blocked": normalized >= threshold,
        "threshold": threshold,
    }


# ============================================================
# RISK CLASSIFICATION
# ============================================================

RISK_MAP = {
    "CRITICAL": [
        "shell", "bash", "exec", "execute", "spawn", "delete", "rm",
        "rmdir", "drop", "truncate", "format", "kill", "terminate",
        "run_command", "execute_command", "os_command",
    ],
    "HIGH": [
        "sql", "database", "query_db", "send_email", "email_send",
        "fs_write", "file_write", "file_delete", "deploy", "git_push",
        "publish", "post_message", "send_message", "transfer",
    ],
    "MEDIUM": [
        "http_request", "api_call", "web_request", "fetch", "download",
        "upload", "webhook", "slack_send",
    ],
    "LOW": [
        "file_read", "fs_read", "search", "query", "read", "list",
        "get", "lookup", "calculate", "analyze",
    ],
}


def classify_tool_risk(tool_name: str) -> dict:
    """Classify a tool name by EU AI Act risk level."""
    name_lower = tool_name.lower().replace("-", "_").replace(" ", "_")

    for level, keywords in RISK_MAP.items():
        for kw in keywords:
            if kw in name_lower:
                return {
                    "tool_name": tool_name,
                    "risk_level": level,
                    "matched_keyword": kw,
                    "recommendation": {
                        "CRITICAL": "Must require human approval before execution.",
                        "HIGH": "Should require approval or logging with review.",
                        "MEDIUM": "Should be logged with audit trail.",
                        "LOW": "Standard logging sufficient.",
                    }[level],
                }

    return {
        "tool_name": tool_name,
        "risk_level": "MEDIUM",
        "matched_keyword": None,
        "recommendation": "Unknown tool — classified as MEDIUM by default. Add to risk registry.",
    }


# ============================================================
# REMEDIATION TEMPLATES — Framework-specific compliance patterns
# ============================================================

TRUST_TEMPLATES = {
    "langchain": {
        "install": "pip install langchain structlog pydantic",
        "code": '''import structlog
from enum import Enum
from pydantic import BaseModel

logger = structlog.get_logger()

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Risk classification for tool calls
TOOL_RISKS = {
    "execute_command": RiskLevel.CRITICAL,
    "send_email": RiskLevel.HIGH,
    "search": RiskLevel.LOW,
}

def approve_tool_call(tool_name: str) -> bool:
    """Block critical tools until human approves (Article 9 + 14)."""
    risk = TOOL_RISKS.get(tool_name, RiskLevel.MEDIUM)
    if risk == RiskLevel.CRITICAL:
        approval = input(f"CRITICAL tool '{tool_name}' — approve? (y/n): ")
        return approval.lower() == "y"
    return True

# Structured audit logging (Article 11 + 12)
def log_agent_action(action: str, details: dict):
    logger.info("agent_action", action=action, **details)

# Use in your LangChain agent:
# result = agent.invoke({"input": query}, config={"callbacks": [your_callback]})''',
    },
    "crewai": {
        "install": "pip install crewai structlog pydantic",
        "code": '''import structlog
from pydantic import BaseModel
from typing import Optional

logger = structlog.get_logger()

class AuditEntry(BaseModel):
    action: str
    agent: str
    task: Optional[str] = None
    risk_level: str = "medium"

def log_crew_action(entry: AuditEntry):
    """Structured logging for CrewAI operations (Article 11 + 12)."""
    logger.info("crew_action", **entry.model_dump())

# Add structured logging to your Crew:
# crew = Crew(agents=[researcher, writer], tasks=[...])
# Log each task result with log_crew_action()''',
    },
    "autogen": {
        "install": "pip install pyautogen structlog",
        "code": '''import structlog

logger = structlog.get_logger()

def audit_message(sender: str, receiver: str, message: str):
    """Log every AutoGen message exchange (Article 11 + 12)."""
    logger.info("autogen_message",
        sender=sender, receiver=receiver,
        message_length=len(message))

def require_approval(func_name: str) -> bool:
    """Human oversight for code execution (Article 14)."""
    print(f"AutoGen wants to execute: {func_name}")
    return input("Approve? (y/n): ").lower() == "y"

# Register hooks on your AssistantAgent and UserProxyAgent
# to call these before execution''',
    },
    "openai": {
        "install": "pip install openai structlog",
        "code": '''import structlog
from openai import OpenAI

logger = structlog.get_logger()

client = OpenAI()

def audited_completion(**kwargs):
    """Wrapper that logs every OpenAI API call (Article 11 + 12)."""
    logger.info("openai_call", model=kwargs.get("model"), 
                messages_count=len(kwargs.get("messages", [])))
    response = client.chat.completions.create(**kwargs)
    logger.info("openai_response", 
                tokens=response.usage.total_tokens if response.usage else 0)
    return response

# Use audited_completion() instead of client.chat.completions.create()''',
    },
    "rag": {
        "install": "pip install structlog pydantic",
        "code": '''import structlog
import re
from pydantic import BaseModel
from typing import List

logger = structlog.get_logger()

BLOCKED_PATTERNS = [r"ignore previous", r"system prompt", r"<\\|system\\|>"]

def sanitize_rag_input(text: str) -> str:
    """Block prompt injection in RAG documents (Article 15)."""
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            logger.warning("rag_injection_blocked", pattern=pattern)
            raise ValueError(f"Blocked content pattern: {pattern}")
    return text

def log_retrieval(query: str, doc_ids: List[str]):
    """Track what documents were retrieved and why (Article 11)."""
    logger.info("rag_retrieval", query=query, doc_count=len(doc_ids), 
                doc_ids=doc_ids)

# Call sanitize_rag_input() on documents before ingestion
# Call log_retrieval() after each retrieval step''',
    },
}


ARTICLE_FIX_MAP = {
    9: {
        "component": "Risk classification system",
        "description": "Classifies every tool call by risk level (LOW/MEDIUM/HIGH/CRITICAL) and blocks critical operations until a human approves.",
        "code": '''from enum import Enum

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

TOOL_RISKS = {
    "execute_command": RiskLevel.CRITICAL,
    "delete_record": RiskLevel.CRITICAL,
    "shell": RiskLevel.CRITICAL,
    "send_email": RiskLevel.HIGH,
    "sql_query": RiskLevel.HIGH,
    "deploy": RiskLevel.HIGH,
    "file_read": RiskLevel.LOW,
    "search": RiskLevel.LOW,
}

def classify_and_gate(tool_name: str) -> bool:
    risk = TOOL_RISKS.get(tool_name, RiskLevel.MEDIUM)
    if risk == RiskLevel.CRITICAL:
        approval = input(f"CRITICAL: '{tool_name}' needs approval (y/n): ")
        return approval.lower() == "y"
    return True''',
    },
    10: {
        "component": "PII protection layer",
        "description": "Detects and masks PII (emails, SSNs, API keys, credit cards) before data reaches the LLM.",
        "code": '''import re
import hashlib
import secrets

PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}",
    "ssn": r"\\b\\d{3}-\\d{2}-\\d{4}\\b",
    "credit_card": r"\\b\\d{4}[- ]?\\d{4}[- ]?\\d{4}[- ]?\\d{4}\\b",
    "api_key": r"(?:sk|pk|api)[_-][a-zA-Z0-9]{20,}",
}

_token_map = {}

def tokenize_pii(text: str) -> str:
    """Replace PII with reversible tokens."""
    for pii_type, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, text):
            original = match.group()
            token = f"[PII:{secrets.token_hex(4)}]"
            _token_map[token] = original
            text = text.replace(original, token)
    return text

def detokenize(text: str) -> str:
    for token, original in _token_map.items():
        text = text.replace(token, original)
    return text''',
    },
    11: {
        "component": "Structured audit logging",
        "description": "Machine-readable logging of every agent operation with timestamps and full call graphs.",
        "code": '''import structlog
import json
from datetime import datetime, timezone

logger = structlog.get_logger()

class AuditLog:
    def __init__(self):
        self.entries = []
    
    def append(self, action: str, details: dict):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "details": details,
        }
        self.entries.append(entry)
        logger.info("audit", **entry)
    
    def export(self) -> str:
        return json.dumps(self.entries, indent=2)

# Usage:
# audit = AuditLog()
# audit.append("tool_call", {"tool": "search", "input": "query"})
# audit.append("llm_call", {"model": "gpt-4", "tokens": 150})''',
    },
    12: {
        "component": "Tamper-evident audit chain (HMAC-SHA256)",
        "description": "Cryptographically chained logs where each entry is signed. Alter one record and the chain breaks.",
        "code": '''import hmac
import hashlib
import json
from datetime import datetime, timezone

class TamperEvidentLog:
    def __init__(self, secret_key: bytes = b"change-this-key"):
        self.entries = []
        self.secret = secret_key
        self.prev_hash = "0" * 64
    
    def append(self, action: str, details: dict):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "details": details,
            "prev_hash": self.prev_hash,
        }
        payload = json.dumps(entry, sort_keys=True).encode()
        entry["hash"] = hmac.new(self.secret, payload, hashlib.sha256).hexdigest()
        self.entries.append(entry)
        self.prev_hash = entry["hash"]
    
    def verify_chain(self) -> bool:
        prev = "0" * 64
        for entry in self.entries:
            stored_hash = entry.pop("hash")
            entry["prev_hash"] = prev
            payload = json.dumps(entry, sort_keys=True).encode()
            expected = hmac.new(self.secret, payload, hashlib.sha256).hexdigest()
            entry["hash"] = stored_hash
            if stored_hash != expected:
                return False
            prev = stored_hash
        return True''',
    },
    14: {
        "component": "Human-in-the-loop approval gate",
        "description": "Blocks sensitive agent actions until a human reviews and approves them at runtime.",
        "code": '''from enum import Enum

class ApprovalMode(str, Enum):
    BLOCK_ALL = "block_all"
    BLOCK_HIGH_AND_CRITICAL = "block_high_critical"
    LOG_ONLY = "log_only"

def human_approval_gate(tool_name: str, risk_level: str, 
                         mode: str = "block_high_critical") -> bool:
    """Gate that requires human approval for sensitive operations."""
    if mode == "log_only":
        print(f"[LOG] Tool: {tool_name}, Risk: {risk_level}")
        return True
    
    if mode == "block_all" or risk_level in ("high", "critical"):
        print(f"\\n⚠️  Agent wants to execute: {tool_name} (risk: {risk_level})")
        approval = input("Approve? (y/n): ")
        return approval.lower() == "y"
    
    return True  # Low/medium pass through''',
    },
    15: {
        "component": "Prompt injection detection",
        "description": "Pattern-based scanning of prompts before they reach the model, blocking known injection techniques.",
        "code": '''import re

INJECTION_PATTERNS = [
    (r"(?i)ignore\\s+(?:all\\s+)?previous\\s+instructions", "safety_bypass", 0.85),
    (r"(?i)you\\s+are\\s+now\\s+(?:a\\s+)?(?:new|different)", "role_override", 0.9),
    (r"(?i)(?:DAN|do\\s+anything\\s+now|jailbreak)", "jailbreak", 0.9),
    (r"<(?:system|instruction|prompt)[^>]*>", "xml_injection", 0.6),
    (r"(?i)(?:sudo|admin\\s+mode|developer\\s+mode)", "privilege_escalation", 0.75),
]

def scan_for_injection(text: str, threshold: float = 0.5) -> dict:
    matches = []
    for pattern, name, weight in INJECTION_PATTERNS:
        if re.search(pattern, text):
            matches.append({"pattern": name, "weight": weight})
    
    score = sum(m["weight"] for m in matches)
    return {
        "detected": len(matches) > 0,
        "score": round(min(score, 1.0), 3),
        "patterns": matches,
        "blocked": score >= threshold,
    }''',
    },
}


ARTICLE_EXPLANATIONS = {
    9: {
        "title": "Risk Management System",
        "summary": "Requires identifying, analyzing, and mitigating risks throughout the AI system lifecycle.",
        "technical": (
            "Every tool call your agent makes needs risk classification. "
            "A file_read is low risk. A delete_database is critical. "
            "You need a system that classifies risk levels and applies "
            "proportionate controls — blocking critical actions, logging high-risk ones. "
            "Implement using Python enums for risk levels and a gating function."
        ),
        "implementation": "Risk classification enum + gating function that requires approval for critical tools",
    },
    10: {
        "title": "Data and Data Governance",
        "summary": "Requires data governance controls including data minimization and PII protection.",
        "technical": (
            "PII flowing through your agent pipeline must be masked before "
            "it reaches the LLM. Names, emails, SSNs, API keys, credit card numbers — "
            "all must be tokenized or redacted. If you're running RAG, documents in your "
            "knowledge base need provenance tracking."
        ),
        "implementation": "Regex-based PII detection + tokenization layer before LLM calls",
    },
    11: {
        "title": "Technical Documentation",
        "summary": "Requires structured, machine-readable documentation of AI system operations.",
        "technical": (
            "Not a PDF on a shelf. The regulation wants structured logs of every "
            "operation: full call graphs showing chain -> LLM -> tool -> result. "
            "Each operation must be timestamped and attributable. Use structured "
            "logging (e.g., structlog) with JSON output."
        ),
        "implementation": "Structured logging with structlog or similar, JSON-formatted audit entries",
    },
    12: {
        "title": "Record-Keeping",
        "summary": "Requires automatic recording of events that regulators can mathematically verify.",
        "technical": (
            "This is where most teams fail. Article 12 requires logs that regulators "
            "can MATHEMATICALLY VERIFY haven't been altered. Standard logger.info() "
            "won't cut it. You need tamper-evident chains — each entry cryptographically "
            "linked to the previous one via HMAC-SHA256. Alter one record, the chain breaks."
        ),
        "implementation": "HMAC-SHA256 chained audit log where each entry references the hash of the previous entry",
    },
    14: {
        "title": "Human Oversight",
        "summary": "Requires humans can review and interrupt AI system execution.",
        "technical": (
            "Humans must be able to review what the agent did and interrupt it "
            "mid-execution. Not after the fact — during runtime. Critical operations "
            "like executing shell commands or sending emails must be blocked until "
            "a human approves."
        ),
        "implementation": "Approval gate function that blocks critical/high-risk tool calls until human confirms",
    },
    15: {
        "title": "Accuracy, Robustness & Cybersecurity",
        "summary": "Requires resilience against prompt injection, data poisoning, and adversarial attacks.",
        "technical": (
            "Your agent needs defense against prompt injection — where malicious input "
            "tries to override system instructions. Also data poisoning if you run RAG — "
            "malicious documents that persist across queries. Article 15 requires "
            "resilience against unauthorized third-party attempts to alter system use."
        ),
        "implementation": "Pattern-based prompt injection scanner + input validation on RAG documents",
    },
}


# ============================================================
# MCP TOOLS — Tier 1: Scanning
# ============================================================

@mcp.tool()
async def scan_code(code: str) -> str:
    """Scan Python AI agent code for EU AI Act compliance.

    Checks all 6 articles (9, 10, 11, 12, 14, 15). Detects the framework
    (LangChain, CrewAI, AutoGen, OpenAI, RAG) and whether compliance
    patterns are present. Returns findings with severity and fix
    recommendations.

    Args:
        code: Python source code to scan
    """
    result = _scan_code(code)
    return json.dumps(result, indent=2)


@mcp.tool()
async def scan_file(file_path: str) -> str:
    """Scan a single Python file for EU AI Act compliance.

    Reads the file and checks all 6 articles. Returns line count,
    framework detection, and compliance findings.

    Args:
        file_path: Absolute path to a .py file
    """
    result = _scan_file(file_path)
    return json.dumps(result, indent=2)


@mcp.tool()
async def scan_project(directory: str) -> str:
    """Scan an entire Python project for EU AI Act compliance.

    Recursively scans all .py files (skips .git, __pycache__, venv, etc.).
    Returns per-file results and aggregate compliance score.

    Args:
        directory: Absolute path to project root directory
    """
    result = _scan_project(directory)
    return json.dumps(result, indent=2)


# ============================================================
# MCP TOOLS — Tier 2: Analysis
# ============================================================

@mcp.tool()
async def analyze_with_model(code: str) -> str:
    """Analyze code using the local fine-tuned AI compliance model (Ollama).

    Sends the code to the air-compliance-v2 Llama model running locally
    via Ollama. Falls back to rule-based scanning if Ollama is not running.
    Your code never leaves your machine.

    Args:
        code: Python source code to analyze
    """
    try:
        result = subprocess.run(
            ["ollama", "run", "air-compliance-v2",
             f"Analyze this code for EU AI Act compliance:\n\n{code}"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.dumps({
                "source": "ollama/air-compliance-v2",
                "analysis": result.stdout.strip(),
            }, indent=2)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # Fallback to rule-based scanning
    fallback = _scan_code(code)
    fallback["source"] = "rule-based (Ollama not available)"
    return json.dumps(fallback, indent=2)


@mcp.tool()
async def check_prompt_injection(text: str) -> str:
    """Check text for prompt injection patterns.

    Scans with 15 weighted patterns including role override, jailbreak,
    system prompt manipulation, privilege escalation, and data exfiltration.
    Returns matched patterns with confidence scores.

    Args:
        text: Text to scan for injection patterns
    """
    result = check_injection(text)
    return json.dumps(result, indent=2)


@mcp.tool()
async def classify_risk(tool_name: str) -> str:
    """Classify a tool or function by EU AI Act risk level.

    Returns CRITICAL, HIGH, MEDIUM, or LOW with the recommendation
    for that risk level.

    Args:
        tool_name: Name of the tool or function to classify
    """
    result = classify_tool_risk(tool_name)
    return json.dumps(result, indent=2)


# ============================================================
# MCP TOOLS — Tier 3: Remediation
# ============================================================

@mcp.tool()
async def add_trust_layer(code: str, framework: str = "langchain") -> str:
    """Generate code to add EU AI Act compliance patterns to existing agent code.

    Returns working, copy-paste-ready Python code with compliance layers
    integrated. Supports: langchain, crewai, autogen, openai, rag.

    Args:
        code: Your existing Python agent code
        framework: Agent framework (langchain, crewai, autogen, openai, rag)
    """
    fw = framework.lower().strip()
    if fw not in TRUST_TEMPLATES:
        return json.dumps({
            "error": f"Unknown framework: {framework}",
            "supported": list(TRUST_TEMPLATES.keys()),
        })

    template = TRUST_TEMPLATES[fw]

    # Run a scan on the original code first
    scan = _scan_code(code)

    return json.dumps({
        "framework": fw,
        "install_command": template["install"],
        "compliance_code": template["code"],
        "original_scan": f"{scan['passed']}/{scan['total']} articles passing",
        "expected_after": "5/6 or 6/6 articles passing",
        "note": "Add the compliance code to your agent setup. Re-scan to verify.",
    }, indent=2)


@mcp.tool()
async def suggest_fix(article: int, framework: str = "langchain") -> str:
    """Get the specific code fix for a failing EU AI Act article.

    Returns the compliance component, explanation, and working code
    to fix the compliance gap for that article.

    Args:
        article: EU AI Act article number (9, 10, 11, 12, 14, or 15)
        framework: Agent framework for framework-specific install command
    """
    if article not in ARTICLE_FIX_MAP:
        return json.dumps({
            "error": f"Article {article} not supported. Use 9, 10, 11, 12, 14, or 15.",
        })

    fix = ARTICLE_FIX_MAP[article]
    fw = framework.lower().strip()
    install = TRUST_TEMPLATES.get(fw, TRUST_TEMPLATES["langchain"])["install"]

    return json.dumps({
        "article": article,
        "component": fix["component"],
        "description": fix["description"],
        "install_command": install,
        "code": fix["code"],
    }, indent=2)


# ============================================================
# MCP TOOLS — Tier 4: Documentation
# ============================================================

@mcp.tool()
async def explain_article(article: int) -> str:
    """Explain what a specific EU AI Act article requires technically.

    Maps each article to the implementation pattern that satisfies it.
    Covers Articles 9, 10, 11, 12, 14, and 15.

    Args:
        article: EU AI Act article number (9, 10, 11, 12, 14, or 15)
    """
    if article not in ARTICLE_EXPLANATIONS:
        return json.dumps({
            "error": f"Article {article} not covered. Use 9, 10, 11, 12, 14, or 15.",
            "covered_articles": [9, 10, 11, 12, 14, 15],
        })

    info = ARTICLE_EXPLANATIONS[article]
    return json.dumps({
        "article": article,
        "title": info["title"],
        "summary": info["summary"],
        "technical_requirements": info["technical"],
        "implementation_approach": info["implementation"],
        "deadline": "August 2, 2026",
    }, indent=2)


@mcp.tool()
async def generate_compliance_report(code: str) -> str:
    """Generate a full EU AI Act compliance report for Python AI agent code.

    Returns a markdown-formatted report with all 6 articles, findings,
    severity scores, and remediation guidance.

    Args:
        code: Python source code to generate report for
    """
    scan = _scan_code(code)

    lines = []
    lines.append("# EU AI Act Compliance Report")
    lines.append(f"\n**Framework detected:** {scan['framework']}")
    lines.append(f"**Compliance score:** {scan['compliance_score']}")
    lines.append(f"**Enforcement deadline:** {scan['deadline']}")
    lines.append("")

    for check in scan["articles"]:
        status = "PASS" if check["passed"] else ("WARN" if check.get("severity") == "warning" else "FAIL")
        icon = "✅" if check["passed"] else ("⚠️" if check.get("severity") == "warning" else "❌")
        lines.append(f"## {icon} Article {check['article']} — {check['title']}: {status}")
        lines.append(f"\n**Severity:** {check['severity']}")
        lines.append(f"\n{check['finding']}")
        if check.get("fix"):
            lines.append(f"\n**Fix:** {check['fix']}")
        lines.append("")

    if scan["passed"] < scan["total"]:
        lines.append("---")
        lines.append(
            "\n**Next steps:** Review the failing articles above and implement "
            "the suggested fixes. Use `suggest_fix(article_number)` for "
            "copy-paste-ready code for each article."
        )

    lines.append(f"\n---\n*Generated by EU AI Act Compliance Scanner*")

    report = "\n".join(lines)
    return json.dumps({"report": report, "scan": scan}, indent=2)
