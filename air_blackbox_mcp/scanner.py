"""
Scanning logic for AIR Blackbox MCP server.

Detects frameworks, trust layers, and checks EU AI Act compliance
across Articles 9, 10, 11, 12, 14, and 15.
"""
import os
import re
from dataclasses import dataclass


# ── Framework detection ──────────────────────────────────────────

FRAMEWORK_PATTERNS = {
    "langchain": [r"from langchain", r"import langchain", r"ChatOpenAI", r"LLMChain", r"AgentExecutor"],
    "crewai": [r"from crewai", r"import crewai", r"Crew\(", r"Agent\(.*role="],
    "autogen": [r"from autogen", r"import autogen", r"AssistantAgent", r"UserProxyAgent"],
    "openai": [r"from openai", r"import openai", r"openai\.chat", r"client\.chat\.completions"],
    "haystack": [r"from haystack", r"import haystack", r"Pipeline\("],
    "llamaindex": [r"from llama_index", r"import llama_index", r"VectorStoreIndex", r"ServiceContext"],
    "semantic_kernel": [r"from semantic_kernel", r"import semantic_kernel", r"Kernel\("],
    "rag": [r"VectorStore", r"Retriever", r"retrieve.*document", r"embedding", r"chunk"],
}

TRUST_LAYER_PATTERNS = [
    r"air_langchain_trust", r"air_crewai_trust", r"air_autogen_trust",
    r"air_openai_trust", r"air_rag_trust", r"AirTrust",
    r"AuditLedger", r"ConsentGate", r"DataVault",
]


def detect_framework(code: str) -> list[str]:
    """Detect which AI frameworks are used in the code."""
    found = []
    for name, patterns in FRAMEWORK_PATTERNS.items():
        for p in patterns:
            if re.search(p, code):
                found.append(name)
                break
    return found


def detect_trust_layer(code: str) -> bool:
    """Check if any AIR trust layer components are present."""
    for p in TRUST_LAYER_PATTERNS:
        if re.search(p, code):
            return True
    return False


# ── Article checks ───────────────────────────────────────────────

@dataclass
class Finding:
    article: int
    name: str
    status: str  # pass, warn, fail
    severity: str  # HIGH, MEDIUM, LOW
    evidence: str
    fix_hint: str = ""


def scan_code(code: str) -> dict:
    """Run all EU AI Act compliance checks on a code string.

    Returns a dict with frameworks, trust_layer, findings, and summary.
    """
    frameworks = detect_framework(code)
    has_trust = detect_trust_layer(code)
    findings = []

    findings.extend(_check_art9(code))
    findings.extend(_check_art10(code))
    findings.extend(_check_art11(code))
    findings.extend(_check_art12(code))
    findings.extend(_check_art14(code))
    findings.extend(_check_art15(code))

    total = len(findings)
    passing = sum(1 for f in findings if f.status == "pass")
    warnings = sum(1 for f in findings if f.status == "warn")
    failing = sum(1 for f in findings if f.status == "fail")

    return {
        "frameworks": frameworks,
        "trust_layer_detected": has_trust,
        "findings": [_to_dict(f) for f in findings],
        "summary": {
            "total_checks": total,
            "passing": passing,
            "warnings": warnings,
            "failing": failing,
            "score": f"{passing}/{total}",
        },
    }


def scan_file(file_path: str) -> dict:
    """Read and scan a single Python file."""
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()
    result = scan_code(code)
    result["file"] = file_path
    result["lines"] = len(code.splitlines())
    return result


def scan_project(directory: str) -> dict:
    """Recursively scan all .py files in a directory."""
    if not os.path.isdir(directory):
        return {"error": f"Directory not found: {directory}"}

    skip_dirs = {
        "node_modules", ".git", "__pycache__", ".venv", "venv",
        "env", ".env", ".tox", ".mypy_cache", ".pytest_cache",
        "dist", "build", "egg-info", ".eggs", "site-packages",
        "deprecated", "archived",
    }

    py_files = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.endswith(".egg-info")]
        for fname in files:
            if fname.endswith(".py"):
                py_files.append(os.path.join(root, fname))

    if not py_files:
        return {"error": f"No Python files found in {directory}", "files_scanned": 0}

    # Merge all code for a project-level scan
    all_code = []
    file_results = []
    for fp in py_files:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
            all_code.append(code)
            file_results.append({"file": os.path.relpath(fp, directory), "lines": len(code.splitlines())})
        except Exception:
            continue

    merged = "\n\n".join(all_code)
    result = scan_code(merged)
    result["directory"] = directory
    result["files_scanned"] = len(file_results)
    result["files"] = file_results[:20]  # Cap at 20 to keep response size manageable
    return result


def _to_dict(f: Finding) -> dict:
    return {
        "article": f.article,
        "name": f.name,
        "status": f.status,
        "severity": f.severity,
        "evidence": f.evidence,
        "fix_hint": f.fix_hint,
    }


# ── Article 9: Risk Management ──────────────────────────────────

def _check_art9(code: str) -> list[Finding]:
    findings = []
    # Error handling around LLM calls
    llm_patterns = r'\.chat\.completions\.create\(|\.invoke\(|\.run\(|\.generate\(|\.predict\('
    has_llm = bool(re.search(llm_patterns, code))
    has_try = bool(re.search(r'\btry\b.*?\bexcept\b', code, re.DOTALL))
    if has_llm:
        findings.append(Finding(article=9, name="LLM call error handling",
            severity="HIGH" if not has_try else "LOW",
            status="pass" if has_try else "fail",
            evidence="LLM calls wrapped in try/except" if has_try else "LLM calls found without error handling",
            fix_hint="Wrap LLM API calls in try/except to handle failures gracefully"))
    # Fallback patterns
    fallback_pats = r'fallback|retry|backoff|with_fallbacks|with_retry|tenacity|max_retries'
    has_fallback = bool(re.search(fallback_pats, code, re.IGNORECASE))
    findings.append(Finding(article=9, name="Fallback/recovery patterns",
        severity="MEDIUM", status="pass" if has_fallback else "warn",
        evidence="Fallback/retry patterns detected" if has_fallback else "No fallback or retry patterns found",
        fix_hint="Add fallback logic for LLM failures (retry, default response, alternative model)"))
    return findings


# ── Article 10: Data Governance ──────────────────────────────────

def _check_art10(code: str) -> list[Finding]:
    findings = []
    # Input validation
    input_pats = r'pydantic|BaseModel|validator|field_validator|validate_input|TypedDict|dataclass|InputGuard|sanitize'
    has_input = bool(re.search(input_pats, code))
    findings.append(Finding(article=10, name="Input validation / schema enforcement",
        severity="MEDIUM", status="pass" if has_input else "warn",
        evidence="Input validation detected (Pydantic/dataclass/TypedDict)" if has_input else "No structured input validation found",
        fix_hint="Use Pydantic models or dataclasses to validate inputs before LLM calls"))
    # PII handling
    pii_pats = r'pii|redact|mask_(?:data|pii|email)|anonymize|tokenize_pii|presidio|scrub|sensitive_data|gdpr|personal_data'
    has_pii = bool(re.search(pii_pats, code, re.IGNORECASE))
    findings.append(Finding(article=10, name="PII handling",
        severity="HIGH" if not has_pii else "LOW",
        status="pass" if has_pii else "warn",
        evidence="PII-aware patterns found" if has_pii else "No PII detection or masking patterns found",
        fix_hint="Add PII detection before sending data to LLM providers"))
    return findings


# ── Article 11: Technical Documentation ──────────────────────────

def _check_art11(code: str) -> list[Finding]:
    findings = []
    # Docstrings
    lines = code.split("\n")
    total_defs = 0
    documented = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if (s.startswith("def ") or s.startswith("class ")) and not s.startswith("def _"):
            total_defs += 1
            for j in range(i + 1, min(i + 4, len(lines))):
                nl = lines[j].strip()
                if nl == "":
                    continue
                if nl.startswith('"""') or nl.startswith("'''"):
                    documented += 1
                break
    if total_defs > 0:
        pct = documented / total_defs * 100
        findings.append(Finding(article=11, name="Code documentation (docstrings)",
            severity="MEDIUM" if pct < 30 else "LOW",
            status="pass" if pct >= 60 else "warn" if pct >= 30 else "fail",
            evidence=f"{documented}/{total_defs} public functions/classes have docstrings ({pct:.0f}%)",
            fix_hint="Add docstrings to public functions explaining purpose and parameters"))
    # Type hints
    total_fn = 0
    typed_fn = 0
    for line in lines:
        s = line.strip()
        if s.startswith("def ") and not s.startswith("def _"):
            total_fn += 1
            if "->" in s or re.search(r':\s*(str|int|float|bool|list|dict|List|Dict|Optional|Any)', s):
                typed_fn += 1
    if total_fn > 0:
        pct = typed_fn / total_fn * 100
        findings.append(Finding(article=11, name="Type annotations",
            severity="LOW",
            status="pass" if pct >= 50 else "warn" if pct >= 20 else "fail",
            evidence=f"{typed_fn}/{total_fn} public functions have type hints ({pct:.0f}%)",
            fix_hint="Add type hints to function signatures"))
    return findings


# ── Article 12: Record-Keeping ───────────────────────────────────

def _check_art12(code: str) -> list[Finding]:
    findings = []
    # Logging - require BOTH import AND actual log calls near AI/LLM code
    log_import_pats = r'import logging|from logging|import structlog|from structlog|import loguru|from loguru'
    log_call_pats = r'logger\.\w+\(|logging\.\w+\(|structlog\.\w+\(|log\.\w+\('
    has_import = bool(re.search(log_import_pats, code))
    has_calls = bool(re.search(log_call_pats, code))
    # Check if logging is used near AI/LLM patterns (within context)
    ai_pats = r'\.invoke\(|\.run\(|\.generate\(|\.predict\(|completions\.create|\.kickoff\(|agent|llm|model'
    has_ai = bool(re.search(ai_pats, code, re.IGNORECASE))
    # Only pass if logging is both imported AND called. If AI code exists
    # but logging is only imported (never called), that's a warn.
    if has_import and has_calls:
        status = "pass"
        evidence = "Logging framework imported and actively used"
    elif has_import and not has_calls:
        status = "warn"
        evidence = "Logging imported but no log calls found - logging may be unused"
    else:
        status = "fail"
        evidence = "No logging framework found"
    findings.append(Finding(article=12, name="Application logging",
        severity="HIGH" if status == "fail" else "MEDIUM" if status == "warn" else "LOW",
        status=status,
        evidence=evidence,
        fix_hint="Add import logging and log key decisions, errors, and LLM interactions"))
    # Tracing / observability
    trace_pats = r'opentelemetry|otel|trace_id|span_id|run_id|request_id|langsmith|langfuse|helicone|instrumentation|dispatcher|event_handler|TracerProvider|callbacks'
    has_trace = bool(re.search(trace_pats, code, re.IGNORECASE))
    findings.append(Finding(article=12, name="Tracing / observability",
        severity="MEDIUM",
        status="pass" if has_trace else "warn",
        evidence="Tracing/observability integration found" if has_trace else "No tracing or observability integration detected",
        fix_hint="Add OpenTelemetry, LangSmith, or similar to track AI decisions"))
    # Action audit trail
    audit_pats = r'action_log|audit_trail|audit_log|log_action|record_action|event_log|CONTENT_TRACING_ENABLED|agent_events|crew_events|emit_event'
    has_audit = bool(re.search(audit_pats, code, re.IGNORECASE))
    findings.append(Finding(article=12, name="Agent action audit trail",
        severity="MEDIUM",
        status="pass" if has_audit else "warn",
        evidence="Action-level audit logging found" if has_audit else "No action-level audit trail detected",
        fix_hint="Log every agent action with user_id, timestamp, action_type, and target"))
    return findings


# ── Article 14: Human Oversight ──────────────────────────────────

def _check_art14(code: str) -> list[Finding]:
    findings = []
    # Human in the loop
    hitl_pats = r'human_in_the_loop|human_approval|require_approval|require_confirmation|confirmation_gate|ask_human|human_input|HumanApprovalCallbackHandler|confirmation_strategy|allow_delegation|interrupt_before|interrupt_after'
    has_hitl = bool(re.search(hitl_pats, code, re.IGNORECASE))
    findings.append(Finding(article=14, name="Human-in-the-loop patterns",
        severity="HIGH" if not has_hitl else "LOW",
        status="pass" if has_hitl else "warn",
        evidence="Human oversight patterns found" if has_hitl else "No human approval gates detected",
        fix_hint="Add human approval gates for high-risk actions"))
    # Rate limiting / budget
    rate_pats = r'rate_limit|max_tokens|max_iterations|max_steps|budget|token_limit|cost_limit|throttle|max_rpm'
    has_rate = bool(re.search(rate_pats, code, re.IGNORECASE))
    findings.append(Finding(article=14, name="Usage limits / budget controls",
        severity="MEDIUM",
        status="pass" if has_rate else "warn",
        evidence="Rate limiting or budget controls found" if has_rate else "No rate limiting or budget controls detected",
        fix_hint="Set max_tokens, max_iterations, or budget limits to prevent runaway agents"))
    # Identity binding
    id_pats = r'user_id|authorized_by|delegated_by|on_behalf_of|auth_context|identity_token|Fingerprint|AgentCard'
    has_id = bool(re.search(id_pats, code, re.IGNORECASE))
    findings.append(Finding(article=14, name="Agent-to-user identity binding",
        severity="MEDIUM",
        status="pass" if has_id else "warn",
        evidence="User identity binding found" if has_id else "Agent actions not tied to authorizing user",
        fix_hint="Track user_id alongside every agent action"))
    # Action boundaries
    bound_pats = r'allowed_tools|tool_whitelist|blocked_tools|allowed_actions|permission_gate|restricted_actions|tool_filter|enabled_tools'
    has_bound = bool(re.search(bound_pats, code, re.IGNORECASE))
    findings.append(Finding(article=14, name="Agent action boundaries",
        severity="MEDIUM",
        status="pass" if has_bound else "warn",
        evidence="Action boundary controls found" if has_bound else "Agent has unrestricted tool access",
        fix_hint="Define allowed_tools or action boundaries to limit agent capabilities"))
    return findings


# ── Article 15: Accuracy, Robustness & Cybersecurity ─────────────

def _check_art15(code: str) -> list[Finding]:
    findings = []
    # Injection defense - require actual implementation, not just mentions in comments
    # Strip comments first to avoid false positives on "# TODO: add guardrail"
    code_no_comments = re.sub(r'#[^\n]*', '', code)
    code_no_docstrings = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', '', code_no_comments)

    # Strong signals: actual imports or instantiations of security libraries
    strong_inj_pats = r'from\s+\w*(?:guardrail|nemo|rebuff|lakera|prompt_guard)\w*\s+import|import\s+\w*(?:guardrail|nemo|rebuff|lakera|prompt_guard)\w*|NeMoGuardrails\(|Rebuff\(|Lakera\(|PromptGuard\(|ContentFilter\(|ModerationClient\('
    has_strong = bool(re.search(strong_inj_pats, code_no_docstrings, re.IGNORECASE))

    # Moderate signals: function definitions or calls that implement defense
    moderate_inj_pats = r'def\s+(?:check|detect|filter|block|sanitize).*(?:inject|prompt|input)|content_filter\s*\(|moderation\s*\(|safety_check\s*\(|check_injection\s*\(|sanitize_input\s*\(|validate_prompt\s*\('
    has_moderate = bool(re.search(moderate_inj_pats, code_no_docstrings, re.IGNORECASE))

    # Weak signals: just the words in actual code (not comments) - only warn
    weak_inj_pats = r'guardrail|content_filter|prompt.?injection|trust_policy'
    has_weak = bool(re.search(weak_inj_pats, code_no_docstrings, re.IGNORECASE))

    if has_strong:
        status = "pass"
        evidence = "Injection defense library imported and instantiated"
    elif has_moderate:
        status = "pass"
        evidence = "Custom injection defense implementation detected"
    elif has_weak:
        status = "warn"
        evidence = "Security-related terms found in code but no active defense implementation"
    else:
        status = "warn"
        evidence = "No prompt injection defense detected"

    findings.append(Finding(article=15, name="Prompt injection defense",
        severity="HIGH" if status == "warn" and not has_weak else "MEDIUM" if status == "warn" else "LOW",
        status=status,
        evidence=evidence,
        fix_hint="Add input sanitization or guardrails to detect prompt injection"))
    # Output validation
    out_pats = r'output_parser|OutputParser|PydanticOutputParser|JsonOutputParser|validate_output|response_model|structured_output|output_pydantic|output_json'
    has_out = bool(re.search(out_pats, code))
    findings.append(Finding(article=15, name="LLM output validation",
        severity="MEDIUM",
        status="pass" if has_out else "warn",
        evidence="Output validation/parsing found" if has_out else "No structured output validation detected",
        fix_hint="Use output parsers to validate LLM responses before acting on them"))
    # Retry logic
    retry_pats = r'retry|backoff|tenacity|max_retries|exponential_backoff|with_retry|Retry\('
    has_retry = bool(re.search(retry_pats, code, re.IGNORECASE))
    findings.append(Finding(article=15, name="Retry / backoff logic",
        severity="LOW",
        status="pass" if has_retry else "warn",
        evidence="Retry/backoff patterns found" if has_retry else "No retry or backoff logic detected",
        fix_hint="Add retry with exponential backoff for LLM API calls"))
    # Unsafe input
    danger_pats = r'f".*\{.*input.*\}.*"|\.format\(.*input'
    has_danger = bool(re.search(danger_pats, code))
    if has_danger:
        findings.append(Finding(article=15, name="Unsafe input handling",
            severity="HIGH", status="warn",
            evidence="Raw user input may be injected directly into prompts",
            fix_hint="Validate and sanitize user input before injecting into LLM prompts"))
    return findings


# ── Injection detection ──────────────────────────────────────────

INJECTION_PATTERNS = [
    ("role_override", 0.9, r"(?i)(ignore|disregard|forget).*(?:previous|above|prior).*(?:instruction|rule|prompt)"),
    ("dan_jailbreak", 0.9, r"(?i)(DAN|do anything now|jailbreak|unlock|god mode)"),
    ("system_prompt_override", 0.85, r"(?i)(system prompt|system message|system instruction).*(?:is|was|should be|override)"),
    ("safety_bypass", 0.85, r"(?i)(bypass|disable|turn off|ignore).*(?:safety|filter|guard|restriction|rule)"),
    ("new_identity", 0.8, r"(?i)(you are now|act as|pretend to be|roleplay as|your new role)"),
    ("urgent_override", 0.8, r"(?i)(emergency|urgent|critical|override|admin).*(?:command|order|instruction|access)"),
    ("privilege_escalation", 0.75, r"(?i)(sudo|admin|root|superuser|elevated|privilege)"),
    ("delimiter_injection", 0.7, r"(?i)(```|<\|im_sep\|>|<\|endoftext\|>|\[INST\]|\[/INST\])"),
    ("hidden_instruction", 0.7, r"(?i)(hidden|secret|covert|invisible).*(?:instruction|command|prompt|message)"),
    ("data_exfil", 0.65, r"(?i)(send|post|email|upload|exfiltrate|leak).*(?:data|info|secret|key|password|token)"),
    ("xml_tag_injection", 0.6, r"</?(?:system|assistant|user|human|ai|instruction)[^>]*>"),
    ("hypothetical_bypass", 0.6, r"(?i)(hypothetically|theoretically|in theory|what if).*(?:could you|would you|can you)"),
    ("output_manipulation", 0.5, r"(?i)(format your|structure your|begin your|start your).*(?:response|output|answer).*(?:with|as|by)"),
    ("encoding_evasion", 0.4, r"(?i)(base64|hex|rot13|encode|decode|translate).*(?:this|following|instruction)"),
    ("tool_abuse", 0.35, r"(?i)(call|invoke|execute|run).*(?:tool|function|api).*(?:repeatedly|loop|infinite|many times)"),
]


def check_injection(text: str) -> dict:
    """Scan text for prompt injection patterns.

    Returns detected patterns, confidence score, and block recommendation.
    """
    detected = []
    max_weight = 0.0

    for name, weight, pattern in INJECTION_PATTERNS:
        if re.search(pattern, text):
            detected.append({"pattern": name, "weight": weight})
            max_weight = max(max_weight, weight)

    blocked = max_weight >= 0.7
    return {
        "detected_patterns": detected,
        "confidence": round(max_weight, 2),
        "would_block": blocked,
        "pattern_count": len(detected),
        "verdict": "BLOCKED" if blocked else "SUSPICIOUS" if detected else "CLEAN",
    }


# ── Risk classification ──────────────────────────────────────────

RISK_MAP = {
    "CRITICAL": ["shell", "bash", "exec", "delete", "rm", "spawn", "execute", "eval", "os.system", "subprocess"],
    "HIGH": ["sql", "database", "send_email", "fs_write", "deploy", "git_push", "write_file", "modify_file"],
    "MEDIUM": ["http_request", "api_call", "fetch", "request", "download", "upload"],
    "LOW": ["file_read", "search", "query", "list", "get", "read", "lookup"],
}


def classify_risk(tool_name: str) -> dict:
    """Classify a tool/function name by EU AI Act risk level."""
    name_lower = tool_name.lower()
    for level, keywords in RISK_MAP.items():
        for kw in keywords:
            if kw in name_lower:
                return {
                    "tool": tool_name,
                    "risk_level": level,
                    "matched_keyword": kw,
                    "recommendation": _risk_recommendation(level),
                }
    return {
        "tool": tool_name,
        "risk_level": "UNKNOWN",
        "matched_keyword": None,
        "recommendation": "Review this tool manually to assess risk level.",
    }


def _risk_recommendation(level: str) -> str:
    recs = {
        "CRITICAL": "Requires human approval before every execution. Add ConsentGate with CRITICAL policy.",
        "HIGH": "Requires human approval or strict audit logging. Add ConsentGate with HIGH policy.",
        "MEDIUM": "Should be logged with full context. Add AuditLedger tracking.",
        "LOW": "Standard logging sufficient. No approval gate required.",
    }
    return recs.get(level, "Review manually.")
